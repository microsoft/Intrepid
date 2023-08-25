from environments.intrepid_env_meta.action_type import ActionType
from model.policy.abstract_stationary import AbstractStationaryPolicy


class StationaryActionConditionPolicy(AbstractStationaryPolicy):
    """A policy that takes action by evaluating an input condition"""

    def __init__(self, action_condition):
        super(StationaryActionConditionPolicy, self).__init__()
        self.action_condition = action_condition

    def action_type(self):
        raise NotImplementedError()

    def gen_q_val(self, observations):
        raise NotImplementedError()

    def sample_action(self, observations):
        return self.action_condition(observations)

    def get_argmax_action(self, observations):
        return self.action_condition(observations)

    def save(self, folder_name, model_name=None):
        raise NotImplementedError()

    def load(self, folder_name, model_name=None):
        raise NotImplementedError()
