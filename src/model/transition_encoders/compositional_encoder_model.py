import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from learning.learning_utils.clustering_algorithm import ClusteringModel


class CompositionalEncoderModel(nn.Module):
    """Model for learning the backward kinematic inseparability"""

    NAME = "compbackwardmodel"

    def __init__(self, config, constants, bootstrap_model=None):
        super(CompositionalEncoderModel, self).__init__()

        self.budget = constants["num_homing_policy"]
        self.config = config
        self.constants = constants
        self.temperature = 1.0

        if config["feature_type"] == "feature":
            self.obs_encoder = nn.Sequential(nn.Linear(config["obs_dim"], constants["n_hidden"]))

            self.prev_encoder = nn.Sequential(nn.Linear(config["obs_dim"], constants["n_hidden"]))

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
            self.obs_action = nn.Sequential(
                nn.Linear(constants["n_hidden"] + config["num_actions"], constants["n_hidden"]),
                nn.LeakyReLU(),
                nn.Linear(constants["n_hidden"], constants["n_hidden"]),
            )

        elif config["feature_type"] == "image":
            self.n_frames, self.height, self.width, self.channels = config["obs_dim"]

            self.img_encoder_conv = nn.Sequential(
                nn.Conv2d(self.n_frames * self.channels, 16, 8, 4),
                nn.LeakyReLU(),
                nn.Conv2d(16, 16, 4, 2),
                nn.LeakyReLU(),
                nn.Conv2d(16, 16, 4, 2),
                nn.LeakyReLU(),
            )

            # Number below is flattened dimension of the output of CNN for a single datapoint
            num_hidden = 384

            # Encoder for previous image
            self.prev_encoder = nn.Sequential(nn.Linear(num_hidden, constants["n_hidden"]))

            # Encoder for current image
            self.obs_encoder = nn.Sequential(nn.Linear(num_hidden, constants["n_hidden"]))

            # action embedding
            self.action_emb = nn.Embedding(config["num_actions"], config["num_actions"])
            self.action_emb.weight.data.copy_(torch.from_numpy(np.eye(config["num_actions"])).float())
            self.action_emb.weight.requires_grad = False

            self.abstract_state_emb = self.budget
            self.obs_emb_dim = self.config["obs_dim"]
            self.act_emb_dim = self.config["num_actions"]

            # Model head
            self.obs_action = nn.Sequential(
                nn.Linear(constants["n_hidden"] + config["num_actions"], constants["n_hidden"]),
                nn.LeakyReLU(),
                nn.Linear(constants["n_hidden"], constants["n_hidden"]),
            )

        else:
            raise NotImplementedError()

        if torch.cuda.is_available():
            self.cuda()

        if bootstrap_model is not None:
            if isinstance(bootstrap_model, ClusteringModel):
                self.load_state_dict(bootstrap_model.feature_fn.model.state_dict())
            else:
                self.load_state_dict(bootstrap_model.state_dict())

    def __gen_logits__(self, prev_observations, actions, observations, discretized, type="logsoftmax"):
        if self.config["feature_type"] == "image":
            observations = observations.view(-1, self.channels, self.height, self.width)
            prev_observations = prev_observations.view(-1, self.channels, self.height, self.width)
            batch_size = prev_observations.size(0)
            x = torch.cat([prev_observations, observations], dim=0)
            x = self.img_encoder_conv(x)
            prev_observations = x[:batch_size, :, :, :].view(batch_size, -1)
            observations = x[batch_size:, :, :, :].view(batch_size, -1)

        prev_encoding = self.prev_encoder(prev_observations)
        action_x = self.action_emb(actions).squeeze()
        prev_obs_act = torch.cat([prev_encoding, action_x], dim=1)
        prev_obs_act_encoding = self.obs_action(prev_obs_act)  # Batch x Hidden Dim

        obs_encoding = self.obs_encoder(observations)  # Batch x Hidden Dim

        if discretized:
            raise NotImplementedError()
        else:
            logits = torch.sum(obs_encoding * prev_obs_act_encoding, dim=1).view(-1)  # Batch
            batch_size = logits.size(0)

            # Probability of 0 is given by sigmoid(logits) and Probability of 1 is given by sigmoid(-logits)
            if type == "logsoftmax":
                log_prob_1 = F.logsigmoid(logits).view(batch_size, 1)
                log_prob_0 = F.logsigmoid(-logits).view(batch_size, 1)
                result = torch.cat([log_prob_0, log_prob_1], dim=1)

            elif type == "softmax":
                prob_1 = torch.sigmoid(logits).view(batch_size, 1)
                prob_0 = (1.0 - prob_1).view(batch_size, 1)
                result = torch.cat([prob_0, prob_1], dim=1)

            else:
                raise AssertionError("Unhandled type ", type)

            return result, {"mean_entropy": None, "assigned_states": None, "prob": None}

    def __gen_scores__(self, prev_observations, actions, observations):
        if self.config["feature_type"] == "image":
            observations = observations.view(-1, self.channels, self.height, self.width)
            prev_observations = prev_observations.view(-1, self.channels, self.height, self.width)
            batch_size = prev_observations.size(0)
            x = torch.cat([prev_observations, observations], dim=0)
            x = self.img_encoder_conv(x)
            prev_observations = x[:batch_size, :, :, :].view(batch_size, -1)
            observations = x[batch_size:, :, :, :].view(batch_size, -1)

        prev_encoding = self.prev_encoder(prev_observations)
        action_x = self.action_emb(actions).squeeze()
        prev_obs_act = torch.cat([prev_encoding, action_x], dim=1)
        prev_obs_act_encoding = self.obs_action(prev_obs_act)  # Batch x Hidden Dim

        obs_encoding = self.obs_encoder(observations)  # Batch x Hidden Dim

        scores = torch.matmul(prev_obs_act_encoding, torch.transpose(obs_encoding, 0, 1))

        return scores

    @staticmethod
    def __gen_batch_logits_from_encodings__(prev_obs_act_encoding, obs_encoding, type="logsigmoid"):
        """
        :param prev_obs_act_encoding: a pytorch variable of size Batch_1 x Hidden_dim
        :param obs_encoding: a pytorch variable of size Batch_2 x Hidden num_factors
        :return: Computed logits of size Batch_2 x Batch_1 representing probability of y=1
        """

        logits = torch.matmul(obs_encoding, prev_obs_act_encoding.transpose(0, 1))  # Batch_2 x Batch_1

        # Probability of 0 is given by sigmoid(logits) and Probability of 1 is given by sigmoid(-logits)
        if type == "logsigmoid":
            result = F.logsigmoid(logits)
        elif type == "sigmoid":
            result = torch.sigmoid(logits)
        else:
            raise AssertionError("Unhandled type ", type)

        return result, {"mean_entropy": None, "assigned_states": None, "prob": None}

    def encode_prev_obs_action(self, prev_observations, actions):
        if self.config["feature_type"] == "image":
            prev_observations = prev_observations.view(-1, self.channels, self.height, self.width)
            batch_size = prev_observations.size(0)
            prev_observations = self.img_encoder_conv(prev_observations).view(batch_size, -1)

        prev_encoding = self.prev_encoder(prev_observations)
        action_x = self.action_emb(actions).squeeze()

        if len(action_x.size()) == 1:
            action_x = action_x.view(1, -1)

        prev_obs_act = torch.cat([prev_encoding, action_x], dim=1)
        prev_obs_act_encoded = self.obs_action(prev_obs_act)  # Batch x Hidden Dim

        return prev_obs_act_encoded

    def encode_curr_obs(self, observations):
        if self.config["feature_type"] == "image":
            observations = observations.view(-1, self.channels, self.height, self.width)
            batch_size = observations.size(0)
            observations = self.img_encoder_conv(observations).view(batch_size, -1)

        obs_encoding = self.obs_encoder(observations)

        return obs_encoding

    def gen_log_prob(self, prev_observations, actions, observations, discretized):
        return self.__gen_logits__(prev_observations, actions, observations, discretized, type="logsoftmax")

    def gen_scores(self, prev_observations, actions, observations):
        return self.__gen_scores__(prev_observations, actions, observations)

    def gen_batch_log_prob_from_encodings(self, prev_obs_act_encoding, obs_encoding):
        return self.__gen_batch_logits_from_encodings__(prev_obs_act_encoding, obs_encoding, type="logsigmoid")

    def gen_prob(self, prev_observations, actions, observations, discretized):
        return self.__gen_logits__(prev_observations, actions, observations, discretized, type="softmax")

    def gen_batch_prob_from_encodings(self, prev_obs_act_encoding, obs_encoding):
        return self.__gen_batch_logits_from_encodings__(prev_obs_act_encoding, obs_encoding, type="sigmoid")

    def encode_observations(self, observations):
        raise NotImplementedError()

    @staticmethod
    def _freeze_param(parameters):
        for param in parameters:
            param.requires_grad = False

    def load_from_another_instance(self, other_model, lock_params=False):
        assert type(self) == type(other_model), "Class must be the same. Found %r and %r" % (type(self), type(other_model))

        self.prev_encoder.load_state_dict(other_model.prev_encoder.state_dict())
        self.obs_encoder.load_state_dict(other_model.obs_encoder.state_dict())
        self.classifier.load_state_dict(other_model.classifier.state_dict())

        if self.config["feature_type"] == "image":
            self.img_encoder_conv.load_state_dict(other_model.img_encoder_conv.state_dict())

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
