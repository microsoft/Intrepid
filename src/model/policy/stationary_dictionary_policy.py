import random

from environments.cerebral_env_meta.action_type import ActionType


class StationaryDictionaryPolicy(ActionType):
    def __init__(self, q_val_dictionary, actions):
        super(ActionType, self).__init__()

        self.q_val_dictionary = q_val_dictionary
        self.actions = actions

    def action_type(self):
        raise NotImplementedError()

    def gen_q_val(self, observations):
        raise NotImplementedError()

    def sample_action(self, state):
        action = self.get_argmax_action(state)
        assert isinstance(action, int), "Action should be of type int. Found %r of type %r" % (action, type(action))
        return action

    def get_argmax_action(self, state):
        state = tuple(state)
        if state in self.q_val_dictionary:
            return int(self.q_val_dictionary[state].argmax())
        else:
            return random.choice(self.actions)

    def save(self, folder_name, model_name=None):
        raise NotImplementedError()

    def load(self, folder_name, model_name=None):
        raise NotImplementedError()
