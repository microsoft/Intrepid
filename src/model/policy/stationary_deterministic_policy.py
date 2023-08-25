import torch
import torch.nn as nn

from environments.intrepid_env_meta.action_type import ActionType


class StationaryDeterministicPolicy(nn.Module, ActionType):

    def __init__(self, config, constants):
        super(StationaryDeterministicPolicy, self).__init__()
        super(ActionType, self).__init__()

        self.config = config
        self.constants = constants

        if config["feature_type"] == "feature":

            if self.constants["policy_type"] == "linear":
                self.layer = nn.Sequential(
                    nn.Linear(config["obs_dim"], config["num_actions"])
                )
            elif self.constants["policy_type"] == "non-linear":
                # TODO currently not supporting more features (e.g., number of layers etc.)
                self.layer = nn.Sequential(
                    nn.Linear(config["obs_dim"], 48),
                    nn.LeakyReLU(),
                    nn.Linear(48, config["num_actions"]),
                )
            else:
                raise AssertionError("Unhandled policy_type %r" % self.constants["policy_type"])

        elif config["feature_type"] == "image":

            # When the feature type is an image, the observation dimension stores the frame, height, width and channel.
            self.n_frames, self.height, self.width, self.channels = config["obs_dim"]

            self.img_encoder_conv = nn.Sequential(
                nn.Conv2d(self.n_frames * self.channels, 32, 8, 4),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.LeakyReLU(),
                nn.Conv2d(64, 64, 4, 2),
                nn.LeakyReLU()
            )

            self.layer = nn.Sequential(
                nn.Linear(576, constants["n_hidden"]),
                nn.LeakyReLU(),
                nn.Linear(constants["n_hidden"], config["num_actions"])
            )

        else:
            raise AssertionError("Unhandled config type ", config["feature_type"])

        if torch.cuda.is_available():
            self.cuda()

    def action_type(self):
        raise ActionType.Discrete

    def gen_q_val(self, observations):

        if self.config["feature_type"] == "image":
            x = observations.view(-1, self.channels, self.height, self.width)
            observations = self.img_encoder_conv(x).view(-1, 576)

        x = self.layer(observations)

        return x

    def sample_action(self, observations):

        # For a deterministic policy, sampling is same as taking argmax
        return self.get_argmax_action(observations)

    def get_argmax_action(self, observations):

        prob = self.gen_q_val(observations)
        return int(prob.max(1)[1])

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
