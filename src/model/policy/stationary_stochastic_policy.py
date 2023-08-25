import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.cerebral_env_meta.action_type import ActionType


class StationaryStochasticPolicy(nn.Module, ActionType):
    def __init__(self, constants, config):
        super(StationaryStochasticPolicy, self).__init__()
        super(ActionType, self).__init__()

        self.layer1 = nn.Linear(config["obs_dim"], 56)
        self.layer2 = nn.Linear(56, 56)
        self.layer3 = nn.Linear(56, config["num_actions"])

    def gen_prob(self, observations):
        x = F.relu(self.layer1(observations))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x))

        return x

    def action_type(self):
        raise ActionType.Discrete

    def sample_action(self, observations):
        prob = self.gen_prob(observations)
        dist = torch.distributions.Categorical(prob)
        return torch.multinomial(dist.probs, 1, True)

    def get_argmax_action(self, observations):
        prob = self.gen_prob(observations)
        return prob.max(1)[1]

    def save(self, folder_name, model_name=None):
        if model_name is None:
            torch.save(self.state_dict(), folder_name + "stationary_policy")
        else:
            torch.save(self.state_dict(), folder_name + model_name)

    def load(self, folder_name, model_name=None):
        if model_name is None:
            self.load_state_dict(torch.load(folder_name + "stationary_policy"))
        else:
            self.load_state_dict(torch.load(folder_name + model_name))
