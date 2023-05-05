import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.cuda import cuda_var
from utils.gumbel import gumbel_sample


class SimpleFeedForwardIK(nn.Module):
    NAME = "simple-feed-forward-ik"

    def __init__(self, config, constants, bootstrap_model=None):
        super(SimpleFeedForwardIK, self).__init__()

        self.budget = 3
        self.config = config
        self.constants = constants
        self.temperature = 1.0

        if config["feature_type"] == "feature":
            self.obs_encoder = nn.Sequential(nn.Linear(config["obs_dim"], self.budget))

            self.prev_encoder = nn.Sequential(nn.Linear(config["obs_dim"], self.budget))

            # Phi
            self.phi_embedding = nn.Embedding(self.budget, self.budget)
            self.phi_embedding.weight.data.copy_(
                torch.from_numpy(np.eye(self.budget)).float()
            )
            self.phi_embedding.weight.requires_grad = False

            self.abstract_state_emb = self.budget
            self.obs_emb_dim = self.config["obs_dim"]

            # Model head
            self.classifier = nn.Sequential(
                nn.Linear(self.budget + self.budget, constants["n_hidden"]),
                nn.LeakyReLU(),
                nn.Linear(constants["n_hidden"], config["num_actions"]),
            )

        elif config["feature_type"] == "image":
            raise NotImplementedError()

        else:
            raise NotImplementedError()

        if torch.cuda.is_available():
            self.cuda()

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

    def __gen_logits__(
        self, prev_observations, observations, discretized, type="logsoftmax"
    ):
        if self.config["feature_type"] == "image":
            raise NotImplementedError()

        prev_encoding = self.prev_encoder(prev_observations)
        obs_encoding = self.obs_encoder(observations)

        if discretized:
            # Compute probability using Gumbel softmax
            prob, log_prob = gumbel_sample(obs_encoding, self.temperature)
            mean_entropy = -torch.mean(torch.sum(prob * log_prob, dim=1))
            argmax_indices = log_prob.max(1)[1]
            obs_encoding = torch.matmul(prob, self.phi_embedding.weight)
        else:
            mean_entropy = None
            argmax_indices = None
            prob = None

        x = torch.cat([prev_encoding, obs_encoding], dim=1)
        logits = self.classifier(x)

        if type == "logsoftmax":
            result = F.log_softmax(logits, dim=1)
        elif type == "softmax":
            result = F.softmax(logits, dim=1)
        else:
            raise AssertionError("Unhandled type ", type)

        return result, {
            "mean_entropy": mean_entropy,
            "assigned_states": argmax_indices,
            "prob": prob,
        }

    def gen_log_prob(self, prev_observations, observations, discretized):
        return self.__gen_logits__(
            prev_observations, observations, discretized, type="logsoftmax"
        )

    def gen_prob(self, prev_observations, observations, discretized):
        return self.__gen_logits__(
            prev_observations, observations, discretized, type="softmax"
        )

    def encode_observations(self, observations):
        observations = cuda_var(torch.from_numpy(np.array(observations))).float()

        if self.config["feature_type"] == "image":
            raise NotImplementedError()

        elif self.config["feature_type"] == "feature":
            assert len(observations.size()) == 1
            observations = observations.view(1, -1)
        else:
            raise NotImplementedError()

        log_prob = F.log_softmax(self.obs_encoder(observations), dim=1)
        argmax_indices = log_prob.max(1)[1]

        return int(argmax_indices[0])

    @staticmethod
    def _freeze_param(parameters):
        for param in parameters:
            param.requires_grad = False

    def load_from_another_instance(self, other_model, lock_params=False):
        assert type(self) == type(
            other_model
        ), "Class must be the same. Found %r and %r" % (type(self), type(other_model))

        self.prev_encoder.load_state_dict(other_model.prev_encoder.state_dict())
        self.obs_encoder.load_state_dict(other_model.obs_encoder.state_dict())
        self.classifier.load_state_dict(other_model.classifier.state_dict())

        if self.config["feature_type"] == "image":
            raise NotImplementedError()

        if lock_params:
            self._freeze_param(self.prev_encoder.parameters())
            self._freeze_param(self.obs_encoder.parameters())
            self._freeze_param(self.classifier.parameters())

            if self.config["feature_type"] == "image":
                self._freeze_param(self.img_encoder_conv.parameters())

    def save(self, folder_name, model_name=None):
        if model_name is None:
            torch.save(self.state_dict(), folder_name + "encoder_model")
        else:
            torch.save(self.state_dict(), folder_name + model_name)

    def load(self, folder_name, model_name=None):
        if model_name is None:
            self.load_state_dict(torch.load(folder_name + "encoder_model"))
        else:
            self.load_state_dict(torch.load(folder_name + model_name))
