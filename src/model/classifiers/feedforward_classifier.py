import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardClassifier(nn.Module):
    """Model for learning the forward kinematic inseparability"""

    NAME = "ff"

    def __init__(self, num_class, config, constants, bootstrap_model=None):
        super(FeedForwardClassifier, self).__init__()

        self.num_class = num_class
        self.config = config
        self.constants = constants

        if config["feature_type"] == "feature":
            self.obs_encoder = nn.Sequential(
                nn.Linear(config["obs_dim"], constants["n_hidden"]),
                nn.LeakyReLU(),
                nn.Linear(constants["n_hidden"], constants["n_hidden"]),
                nn.LeakyReLU(),
                nn.Linear(constants["n_hidden"], self.num_class),
            )

        elif config["feature_type"] == "image":
            raise NotImplementedError()

        else:
            raise NotImplementedError()

        if torch.cuda.is_available():
            self.cuda()

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

    def _gen_logits(self, observations, return_log_prob=True):
        if self.config["feature_type"] == "image":
            raise AssertionError("Cannot handle images right now")

        logits = self.obs_encoder(observations)

        if return_log_prob:
            return F.log_softmax(logits, dim=1), dict()
        else:
            return F.softmax(logits, dim=1), dict()

    def gen_log_prob(self, observations):
        return self._gen_logits(observations, return_log_prob=True)

    def gen_prob(self, observations):
        return self._gen_logits(observations, return_log_prob=False)

    def save(self, folder_name, model_name=None):
        if model_name is None:
            torch.save(self.state_dict(), folder_name + FeedForwardClassifier.NAME)
        else:
            torch.save(self.state_dict(), folder_name + model_name)

    def load(self, folder_name, model_name=None):
        if model_name is None:
            self.load_state_dict(torch.load(folder_name + FeedForwardClassifier.NAME))
        else:
            self.load_state_dict(torch.load(folder_name + model_name))
