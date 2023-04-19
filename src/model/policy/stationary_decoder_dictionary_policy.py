from environments.cerebral_env_meta.action_type import ActionType
from model.policy.stationary_dictionary_policy import StationaryDictionaryPolicy


class StationaryDecoderLatentPolicy(ActionType):

    def __init__(self, decoder, q_val_dictionary, actions):
        super(ActionType, self).__init__()

        self.decoder = decoder
        self.latent_policy = StationaryDictionaryPolicy(q_val_dictionary, actions)

    def action_type(self):
        raise NotImplementedError()

    def gen_q_val(self, observations):
        raise NotImplementedError()

    def sample_action(self, obs):
        obs = obs[0]
        latent_state = self.decoder.encode_observations(obs)
        return self.latent_policy.sample_action(latent_state)

    def get_argmax_action(self, obs):

        latent_state = self.decoder.encode_observations(obs)
        return self.latent_policy.get_argmax_action(latent_state)

    def save(self, folder_name, model_name=None):
        raise NotImplementedError()

    def load(self, folder_name, model_name=None):
        raise NotImplementedError()
