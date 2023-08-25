import random

from learning.learning_utils.episode import Episode
from environments.cerebral_env_meta.environment_keys import EnvKeys


class Roll:
    def __init__(self, env, actions):
        self._env = env
        self.eps = None
        self.actions = actions

    def roll_in(self, policy, t):
        """
        Roll-in in the environment using the above policy till time step t
        :param policy: A policy for taking actions
        :param t: Number of actions taken by the policy.
        :return:
        """

        obs, info = self._env.reset()
        self.eps = Episode(state=info[EnvKeys.ENDO_STATE], observation=obs, gamma=1.0)

        for h in range(0, t):
            action = policy.sample_action(obs, h)
            obs, reward, done, info = self._env.step(action)
            self.eps.add(
                action=action,
                reward=reward,
                new_obs=obs,
                new_state=info[EnvKeys.ENDO_STATE],
            )

        return self

    def take_random(self, k):
        for t in range(0, k):
            action = random.choice(self.actions)
            self.take_action(action)

        return self

    def take_action(self, action):
        obs, reward, done, info = self._env.step(action)

        self.eps.add(
            action=action,
            reward=reward,
            new_obs=obs,
            new_state=info[EnvKeys.ENDO_STATE],
        )

        return self

    def roll_out(self, policy, t):
        raise NotImplementedError()

    def terminate(self):
        """Terminate the roll-out"""

        self.eps.terminate()
        return self

    def retrieve(self, pattern=None):
        """Retrieve the details"""

        if pattern is not None:
            raise NotImplementedError()
        else:
            return self.eps
