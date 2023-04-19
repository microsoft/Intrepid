import random
import numpy as np
import torch.nn as nn

from environments.cerebral_env_meta.action_type import ActionType


class NonStationaryComposedPolicy(nn.Module, ActionType):

    def __init__(self, encoder_fn, q_values, config):
        super(NonStationaryComposedPolicy, self).__init__()
        super(ActionType, self).__init__()

        self.encoder_fn = encoder_fn
        self.q_values = q_values
        self.action_space = config["actions"]

    def action_type(self):
        raise ActionType.Discrete

    def sample_action(self, observations, time_step):

        return self.get_argmax_action(observations, time_step)

    def get_argmax_action(self, observations, time_step):

        if self.encoder_fn is None:
            return random.choice(self.action_space)

        if type(self.encoder_fn) == list or type(self.encoder_fn) == dict:
            latent_state = self.encoder_fn[time_step].encode_observations(observations)
        else:
            latent_state = self.encoder_fn.encode_observations(observations)

        if (time_step, latent_state) in self.q_values:

            q_values = self.q_values[(time_step, latent_state)]

            return np.random.choice(np.flatnonzero(q_values == q_values.max()))
        else:
            return random.choice(self.action_space)

    def save(self, folder_name, model_name=None):

        raise NotImplementedError()

    def load(self, folder_name, model_name=None):

        raise NotImplementedError()
