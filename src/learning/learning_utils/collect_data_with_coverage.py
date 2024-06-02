import random
import numpy as np

from learning.datastructures.episode import Episode
from environments.intrepid_env_meta.environment_keys import EnvKeys


class CollectDatawithCoverage:
    """
    A useful class for collecting episodes without the aid of a trained model
    """

    RANDOM, OPT, MIX, MIXIN, MIXIN2 = range(5)

    def __init__(self, exp_setup):
        # Environment parameters
        self.horizon = exp_setup.config["horizon"]
        self.actions = exp_setup.config["actions"]

        if exp_setup.constants["data_type"] == "random":
            self.data_type = CollectDatawithCoverage.RANDOM

        elif exp_setup.constants["data_type"] == "opt":
            self.data_type = CollectDatawithCoverage.OPT

        elif exp_setup.constants["data_type"] == "mix":
            self.data_type = CollectDatawithCoverage.MIX

        elif exp_setup.constants["data_type"] == "mixin":
            self.data_type = CollectDatawithCoverage.MIXIN

        elif exp_setup.constants["data_type"] == "mixin2":
            self.data_type = CollectDatawithCoverage.MIXIN2

        else:
            raise AssertionError("Unhandled data type %s" % exp_setup.constants["data_type"])

    def _collect_random_episode(self, env):
        obs, info = env.reset()
        eps = Episode(observation=obs, state=info[EnvKeys.ENDO_STATE])

        for h in range(0, self.horizon):
            action = np.random.choice(self.actions)
            new_obs, reward, done, info = env.step(action)

            eps.add(
                action=action,
                reward=reward,
                new_obs=new_obs,
                new_state=info[EnvKeys.ENDO_STATE],
            )
            if done:
                break

        eps.terminate()

        return eps

    def _collect_opt_episode(self, env):
        obs, info = env.reset()
        eps = Episode(observation=obs, state=info[EnvKeys.ENDO_STATE])

        for h in range(0, self.horizon):
            # We assume that environment gives optimal action
            action = env.get_optimal_action()
            new_obs, reward, done, info = env.step(action)

            eps.add(
                action=action,
                reward=reward,
                new_obs=new_obs,
                new_state=info[EnvKeys.ENDO_STATE],
            )
            if done:
                break

        eps.terminate()

        return eps

    def _collect_opt_random_episode(self, env, deviation_step):
        obs, info = env.reset()
        eps = Episode(observation=obs, state=info[EnvKeys.ENDO_STATE])

        for h in range(0, self.horizon):
            # We assume that environment gives optimal action

            if h >= deviation_step:
                action = np.random.choice(self.actions)
            else:
                action = env.get_optimal_action()
            new_obs, reward, done, info = env.step(action)

            eps.add(
                action=action,
                reward=reward,
                new_obs=new_obs,
                new_state=info[EnvKeys.ENDO_STATE],
            )
            if done:
                break

        eps.terminate()

        return eps

    def _collect_random_goal_with_deviation(self, env):
        obs, info = env.reset()

        p = random.random()

        # Goal is sampled immediately after resetting as the goal is within a certain distance from the agent
        if p < 0.5:
            goal_pos = env.env.get_current_goal()
            deviation_step = self.horizon  # This implies we never deviate from the optimal policy
        elif 0.5 <= p < 0.75:
            # Sample goal
            goal_pos = env.env.sample_goal()
            deviation_step = self.horizon  # This implies we never deviate from the optimal policy
        elif 0.75 <= p < 0.875:
            goal_pos = env.env.get_current_goal()
            deviation_step = random.randint(0, self.horizon - 1)
        else:
            # Sample goal
            goal_pos = env.env.sample_goal()
            deviation_step = random.randint(0, self.horizon - 1)

        eps = Episode(observation=obs, state=info[EnvKeys.ENDO_STATE])

        for h in range(0, self.horizon):
            # We assume that environment gives optimal action

            if h >= deviation_step:
                action = np.random.choice(self.actions)
            else:
                action = env.env.get_goal_pos_action(goal_pos)
            new_obs, reward, done, info = env.step(action)

            eps.add(
                action=action,
                reward=reward,
                new_obs=new_obs,
                new_state=info[EnvKeys.ENDO_STATE],
            )
            if done:
                break

        eps.terminate()

        return eps

    def collect_episodes(self, env, dataset_size, data_type=None):
        if data_type is None:
            data_type = self.data_type

        # Collect data
        # TODO: convert to TensorDataset
        episodes = []

        for _ in range(0, dataset_size):
            if data_type == CollectDatawithCoverage.RANDOM:
                eps = self._collect_random_episode(env)

            elif data_type == CollectDatawithCoverage.OPT:
                eps = self._collect_opt_episode(env)

            elif data_type == CollectDatawithCoverage.MIX:
                if random.random() < 0.5:
                    eps = self._collect_random_episode(env)
                else:
                    eps = self._collect_opt_episode(env)

            elif data_type == CollectDatawithCoverage.MIXIN:
                if random.random() < 0.5:
                    deviation_step = self.horizon  # This implies we never deviate from the optimal policy
                else:
                    deviation_step = random.randint(0, self.horizon - 1)
                eps = self._collect_opt_random_episode(env, deviation_step)

            elif data_type == CollectDatawithCoverage.MIXIN2:
                eps = self._collect_random_goal_with_deviation(env)

            else:
                raise AssertionError("Unhandled data collection strategy %r" % self.data_type)

            episodes.append(eps)

        return episodes
