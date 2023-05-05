import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.cuda import cuda_var


class ConvMClassifier(nn.Module):
    """Model for learning the forward kinematic inseparability"""

    NAME = "convm"

    def __init__(self, num_class, config, constants, bootstrap_model=None):
        super(ConvMClassifier, self).__init__()

        self.num_class = num_class
        self.config = config
        self.constants = constants
        self.channel = config["obs_dim"][2]

        if config["feature_type"] == "feature":
            raise NotImplementedError()

        elif config["feature_type"] == "image":
            self.encoding = cuda_var(
                self.positionalencoding2d(
                    d_model=4, height=config["obs_dim"][0], width=config["obs_dim"][1]
                )
            )

            self.input_shape = config["obs_dim"]
            n = self.input_shape[0]
            m = self.input_shape[1]
            self.embedding_size = ((n - 1) // 4 - 1) * ((m - 1) // 4 - 1) * 32

            self.image_conv = nn.Sequential(
                nn.Conv2d(self.channel + 4, 16, (8, 8), 8),
                nn.LeakyReLU(),
                nn.Conv2d(16, 32, (4, 4), 4),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(128, num_class),
            )

            if bootstrap_model is not None:
                self.load_state_dict(bootstrap_model.state_dict())

            if torch.cuda.is_available():
                self.cuda()

        else:
            raise NotImplementedError()

        if torch.cuda.is_available():
            self.cuda()

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

    @staticmethod
    def positionalencoding2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dimension (got dim={:d})".format(d_model)
            )
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pos_w = torch.arange(0.0, width).unsqueeze(1)
        pos_h = torch.arange(0.0, height).unsqueeze(1)
        pe[0:d_model:2, :, :] = (
            torch.sin(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )
        pe[1:d_model:2, :, :] = (
            torch.cos(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )
        pe[d_model::2, :, :] = (
            torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )
        pe[d_model + 1 :: 2, :, :] = (
            torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )

        return pe

    def _gen_logits(self, observations, return_log_prob=True):
        if self.config["feature_type"] == "feature":
            raise AssertionError(
                "Conv classifier can only operate on images and not 1-D features."
            )

        logits = self.obs_encoder(observations)

        if return_log_prob:
            return F.log_softmax(logits, dim=1), dict()
        else:
            return F.softmax(logits, dim=1), dict()

    def obs_preprocess(self, obs):
        x = obs.reshape(-1, *self.input_shape)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    def obs_encoder(self, obs):
        obs = self.obs_preprocess(obs)
        my_encoding = self.encoding[None, :, :, :].repeat_interleave(obs.size(0), dim=0)
        obs = torch.cat([obs, my_encoding], dim=1)
        x = self.image_conv(obs)
        return F.log_softmax(x, dim=1)

    def gen_log_prob(self, observations):
        return self._gen_logits(observations, return_log_prob=True)

    def gen_prob(self, observations):
        return self._gen_logits(observations, return_log_prob=False)

    def save(self, folder_name, model_name=None):
        if model_name is None:
            torch.save(self.state_dict(), folder_name + ConvMClassifier.NAME)
        else:
            torch.save(self.state_dict(), folder_name + model_name)

    def load(self, folder_name, model_name=None):
        if model_name is None:
            self.load_state_dict(torch.load(folder_name + ConvMClassifier.NAME))
        else:
            self.load_state_dict(torch.load(folder_name + model_name))
