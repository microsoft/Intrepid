import pickle

from environments.intrepid_env_meta.action_type import ActionType


class StationaryConstantPolicy(ActionType):
    """A policy that always takes the same action deterministically regardless of input"""

    def __init__(self, action):
        super(ActionType, self).__init__()
        self.action = action

    def action_type(self):
        raise NotImplementedError()

    def gen_q_val(self, observations):
        raise NotImplementedError()

    def sample_action(self, observations):
        return self.action

    def get_argmax_action(self, observations):
        return self.action

    def save(self, folder_name, model_name=None):
        with open(folder_name + model_name, "wb") as fobj:
            pickle.dump(self.action, fobj)

    def load(self, folder_name, model_name=None):
        with open(folder_name + model_name, "rb") as fobj:
            self.action = pickle.load(fobj)
