import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.cuda import cuda_var
from utils.gumbel import gumbel_sample


class FactoRLEncoder(nn.Module):
    def __init__(self, budget, factor_obs_dim, config, constants):
        super(FactoRLEncoder, self).__init__()

        self.config = config
        self.constants = constants
        self.budget = budget
        self.temperature = 1.0

        if config["feature_type"] == "feature":
            self.obs_encoder = nn.Sequential(nn.Linear(factor_obs_dim, self.budget))

            self.prev_encoder = nn.Sequential(nn.Linear(config["obs_dim"], 3))

            # Phi
            self.phi_embedding = nn.Embedding(self.budget, self.budget)
            self.phi_embedding.weight.data.copy_(torch.from_numpy(np.eye(self.budget)).float())
            self.phi_embedding.weight.requires_grad = False

            # action embedding
            self.action_emb = nn.Embedding(config["num_actions"], config["num_actions"])
            self.action_emb.weight.data.copy_(torch.from_numpy(np.eye(config["num_actions"])).float())
            self.action_emb.weight.requires_grad = False

            self.abstract_state_emb = self.budget
            self.obs_emb_dim = self.config["obs_dim"]
            self.act_emb_dim = self.config["num_actions"]

            # Model head
            self.classifier = nn.Sequential(
                nn.Linear(3 + config["num_actions"] + self.budget, constants["n_hidden"]),
                nn.LeakyReLU(),
                nn.Linear(constants["n_hidden"], 2),
            )

        else:
            raise AssertionError("Unhandled feature type")

        if torch.cuda.is_available():
            self.cuda()

    def gen_logits_(self, prev_observations, actions, observations, discretized, type="logsoftmax"):
        """
        :param prev_obs:    Pytorch float tensor of size batch x dim1
        :param action:      Pytorch long tensor of size batch
        :param obs_patch:   Pytorch float tensor of size batch x dim2
        :return:
        """

        if self.config["feature_type"] == "image":
            raise AssertionError()

        prev_encoding = self.prev_encoder(prev_observations)
        action_x = self.action_emb(actions).squeeze()
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

        x = torch.cat([prev_encoding, action_x, obs_encoding], dim=1)
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

    def gen_log_prob(self, prev_observations, actions, observations, discretized):
        return self.gen_logits_(prev_observations, actions, observations, discretized, type="logsoftmax")

    def encode_observations(self, observations):
        observations = cuda_var(torch.from_numpy(observations)).float()

        if self.config["feature_type"] == "image":
            observations = observations.view(-1, self.channels, self.height, self.width)
            observations = self.img_encoder_conv(observations).view(1, -1)
        elif self.config["feature_type"] == "feature":
            assert len(observations.size()) == 1
            observations = observations.view(1, -1)
        else:
            raise NotImplementedError()

        log_prob = F.log_softmax(self.obs_encoder(observations), dim=1)

        argmax_indices = log_prob.max(1)[1]

        return int(argmax_indices[0])
