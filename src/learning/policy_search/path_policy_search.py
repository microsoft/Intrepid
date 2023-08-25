import time
import os
import pickle

from model.policy.stationary_constant_policy import StationaryConstantPolicy
from environments.intrepid_env_meta.environment_wrapper import GenerateEnvironmentWrapper
from learning.policy_search.abstract_policy_search import AbstractPolicySearch


class PathPolicySearch(AbstractPolicySearch):
    """ Solve a reinforcement learning problem using Path Policy Search. The path policy search finds path
    that maximizes reward. It returns a policy which always executes a sequence of actions. """

    def __init__(self, config, constants):
        AbstractPolicySearch.__init__(self)

        self.config = config
        self.constants = constants
        self.gamma = config["gamma"]
        self.num_eval_samples = constants["eval_homing_policy_sample_size"]

    @staticmethod
    def _generate_pps_dataset(homing_policy_dataset, reward_func, horizon, reward_id):
        """ Create a dataset of the form dictionary over last time homing policies with values as another dictionary
         with actions as key and (total reward, count) as value """

        dataset = dict()

        for dp in homing_policy_dataset:

            if reward_func is None:
                raise AssertionError("Cannot use Path Policy Search without a given reward function. Use PSDP instead.")
            else:
                # TODO we should know what type of model we have in the reward function so
                # we can define the appropriate reward function
                if "cluster_center" not in dp.meta_dict:
                    reward = reward_func(dp.get_next_obs(), horizon)
                else:
                    reward = 1.0 if dp.meta_dict["cluster_center"] == reward_id else 0.0

            action = dp.get_action()
            prev_index = -1 if dp.get_policy_index() is None else dp.get_policy_index()

            if prev_index in dataset:
                if action in dataset[prev_index]:
                    prev_value = dataset[prev_index][action]
                    dataset[prev_index][action] = (prev_value[0] + reward, prev_value[1] + 1)
                else:
                    dataset[prev_index][action] = (reward, 1)
            else:
                dataset[prev_index] = {action: (reward, 1)}

        return dataset

    @staticmethod
    def do_train(config, constants, homing_policy_dataset, env_info, policy_folder_name, actions, horizon,
                 encoder_reward_args, homing_policies, logger, debug):

        pps = PathPolicySearch(config, constants)

        env = GenerateEnvironmentWrapper.make_env(env_info[0], config, env_info[1])

        if encoder_reward_args is None:
            reward_func = None
            reward_id = -1
        else:
            encoding_function, reward_id = encoder_reward_args
            reward_func = lambda observation, time: 1 \
                if time == horizon and encoding_function.encode_observations(observation) == reward_id else 0

        tensorboard = None

        learned_policy, mean_reward, info =  \
            pps.train(homing_policy_dataset, env, actions, horizon, reward_func, homing_policies, logger,
                      tensorboard, debug, reward_id)

        # Save the learned policy to disk
        PathPolicySearch._save_policy(learned_policy, policy_folder_name, horizon, info["prev_policy_index"])

    @staticmethod
    def _save_policy(learned_policy, policy_folder_name, horizon, policy_index):
        """ Save the learned policy to disk. Since we only learned the last step therefore we save the value
         of the last step along with the index of the homing policy for previous step. """

        if not os.path.exists(policy_folder_name):
            os.makedirs(policy_folder_name)
        learned_policy[horizon].save(folder_name=policy_folder_name, model_name="step_%d" % horizon)

        with open(policy_folder_name + "prev_policy_index", 'wb') as fobj:
            pickle.dump(policy_index, fobj)

    def read_policy(self, policy_folder_name, horizon, previous_step_homing_policy, delete=False):
        """ Read the policy from the disk """

        homing_policy = dict()
        with open(policy_folder_name + "prev_policy_index", 'rb') as fobj:
            policy_index = pickle.load(fobj)

        for j in range(1, horizon):
            homing_policy[j] = previous_step_homing_policy[policy_index][j]

        policy = StationaryConstantPolicy(action=None)
        policy.load(folder_name=policy_folder_name, model_name="step_%d" % horizon)
        homing_policy[horizon] = policy

        if delete:
            # Delete the file after reading for saving disk space
            os.remove(policy_folder_name + "step_%d" % horizon)

        return homing_policy

    def train(self, replay_memory, env, actions, horizon, reward_func, homing_policies, logger, tensorboard,
              debug=False, reward_id=-1):
        """ Performs the learning """

        logger.log("Doing PPS on Horizon %r." % horizon)

        # Find which action maximizes the total reward at the last time step
        sample_start = time.time()
        dataset = self._generate_pps_dataset(replay_memory, reward_func, horizon, reward_id)
        sample_time = time.time() - sample_start

        learned_policy = dict()
        best_homing_policy_index = -1
        best_action = None
        best_reward = float('-inf')

        for homing_policy_index in dataset:
            for action in dataset[homing_policy_index]:
                val = dataset[homing_policy_index][action]
                mean_step_reward = val[0] / float(max(1, val[1]))
                if mean_step_reward > best_reward:
                    best_reward = mean_step_reward
                    best_homing_policy_index = homing_policy_index
                    best_action = action

        assert best_action is not None and (horizon == 1 or best_homing_policy_index != -1), \
            "Failed to find optimal path"

        learned_policy[horizon] = StationaryConstantPolicy(action=best_action)

        if horizon > 1:
            for i in range(1, horizon):
                learned_policy[i] = homing_policies[horizon - 1][best_homing_policy_index][i]

        mean_reward = self._evaluate__learned_policy(env, horizon, reward_func, learned_policy,
                                                     logger, num_rollouts=self.num_eval_samples,
                                                     actions=self.config["actions"])

        pps_step_time = time.time() - sample_start
        logger.log("PPS(%d): Mean Reward: %r, Time: %r sec (sample time %r sec)." %
                   (reward_id, mean_reward, round(pps_step_time, 2), round(sample_time, 2)))

        return learned_policy, mean_reward, {"prev_policy_index": best_homing_policy_index}
