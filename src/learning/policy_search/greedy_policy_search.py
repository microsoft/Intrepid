import time
import os
import pickle
import torch
import numpy as np

from utils.cuda import cuda_var
from model.policy.stationary_deterministic_policy import StationaryDeterministicPolicy
from environments.cerebral_env_meta.environment_wrapper import GenerateEnvironmentWrapper
from learning.policy_search.abstract_policy_search import AbstractPolicySearch
from learning.learning_utils.contextual_bandit_oracle import ContextualBanditOracle


class GreedyPolicySearch(AbstractPolicySearch):
    """ Solve a reinforcement learning problem using Greedy Policy Search. The GPS is linear in horizon.
     GPS works by finding the optimal homing policy upto horizon-1 followed by optimal continuation. """

    def __init__(self, config, constants):
        AbstractPolicySearch.__init__(self)

        self.config = config
        self.constants = constants
        self.gamma = config["gamma"]
        self.num_eval_samples = constants["eval_homing_policy_sample_size"]
        self.batch_size = constants["cb_oracle_batch_size"]
        self.contextual_bandit_oracle = ContextualBanditOracle(config, constants)

    def _best_prefix_homing_policy(self, stationary_policy, dataset, num_homing_policy):

        homing_policy_reward = np.zeros(num_homing_policy)
        homing_policy_reward_count = np.zeros(num_homing_policy)

        dataset_size = len(dataset)
        batches = [dataset[i:i + self.batch_size] for i in range(0, dataset_size, self.batch_size)]

        for batch in batches:

            observation_batch = cuda_var(torch.cat([torch.from_numpy(np.array(point[0])).view(1, -1)
                                                    for point in batch], dim=0)).float()
            predicted_rewards = stationary_policy.gen_q_val(observation_batch)  # Batch x Actions
            predicted_max_rewards = predicted_rewards.max(1)[0]  # Batch

            for i, dp in enumerate(batch):
                policy_index = dp[6]
                predicted_reward = float(predicted_max_rewards[i])

                homing_policy_reward[policy_index] += predicted_reward
                homing_policy_reward_count[policy_index] += 1.0

        for i in range(0, num_homing_policy):
            homing_policy_reward[i] = homing_policy_reward[i] / max(1.0, homing_policy_reward_count[i])

        return homing_policy_reward.argmax()

    @staticmethod
    def _generate_contextual_bandit_dataset(homing_policy_dataset, reward_func, horizon):

        contextual_bandit_dataset = []

        for dp in homing_policy_dataset:

            if reward_func is None:
                raise AssertionError("Cannot use Path Policy Search without a given reward function. Use PSDP instead.")
            else:
                reward = reward_func(dp.get_next_obs(), horizon)

            total_reward = reward
            cb_dp = (dp.get_curr_obs(), dp.get_action_prob(), dp.get_action(),
                     total_reward, dp.get_curr_state(), dp.get_next_state(), dp.get_policy_index())

            contextual_bandit_dataset.append(cb_dp)

        return contextual_bandit_dataset

    @staticmethod
    def do_train(config, constants, homing_policy_dataset, env_info, policy_folder_name, actions, horizon,
                 encoder_reward_args, homing_policies, logger, debug):

        gps = GreedyPolicySearch(config, constants)

        env = GenerateEnvironmentWrapper.make_env(env_info[0], config, env_info[1])

        if encoder_reward_args is None:
            reward_func = None
            reward_id = -1
        else:
            encoding_function, reward_id = encoder_reward_args
            reward_func = lambda obs, time: 1 if time == horizon and \
                                                 encoding_function.encode_observations(obs) == reward_id else 0

        tensorboard = None

        learned_policy, mean_reward, info =  \
            gps.train(homing_policy_dataset, env, actions, horizon, reward_func, homing_policies, logger,
                      tensorboard, debug, reward_id)

        # Save the learned policy to disk
        GreedyPolicySearch._save_policy(learned_policy, policy_folder_name, horizon, info["prev_policy_index"])

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

        policy = StationaryDeterministicPolicy(self.config, self.constants)
        policy.load(folder_name=policy_folder_name, model_name="step_%d" % horizon)
        homing_policy[horizon] = policy

        if delete:
            # Delete the file after reading for saving disk space
            os.remove(policy_folder_name + "step_%d" % horizon)

        return homing_policy

    def train(self, homing_policy_dataset, env, actions, horizon, reward_func, homing_policies, logger, tensorboard,
                  debug=False, reward_id=-1):
        """ Performs the learning """

        logger.log("Doing GPS on Horizon %r." % horizon)

        # Learn the optimal policy for the last step
        sample_start = time.time()
        contextual_bandit_dataset = self._generate_contextual_bandit_dataset(homing_policy_dataset, reward_func,
                                                                             horizon)
        sample_time = time.time() - sample_start

        # Call contextual bandit oracle
        oracle_start = time.time()
        optimal_policy_step, info = self.contextual_bandit_oracle.learn_optimal_policy(contextual_bandit_dataset,
                                                                                       logger, tensorboard, debug)
        oracle_time = time.time() - oracle_start

        learned_policy = dict()
        learned_policy[horizon] = optimal_policy_step

        if horizon > 1:
            # Find the best prefix homing policy
            best_homing_policy_index = self._best_prefix_homing_policy(optimal_policy_step, contextual_bandit_dataset,
                                                                       len(homing_policies[horizon - 1]))
            for i in range(1, horizon):
                learned_policy[i] = homing_policies[horizon - 1][best_homing_policy_index][i]
        else:
            best_homing_policy_index = -1

        if debug:
            self.log_intermediate_results(contextual_bandit_dataset, optimal_policy_step, env, actions, horizon,
                                          horizon, homing_policies, learned_policy,
                                          reward_func, logger, self.num_eval_samples, 32)

        # mean_reward = self._evaluate__learned_policy(env, horizon, reward_func, learned_policy,
        #                                              logger, num_rollouts=self.num_eval_samples,
        #                                              actions=self.config["actions"])
        mean_reward = 0.0   # TODO currently disabled for accounting

        gps_step_time = time.time() - sample_start
        logger.log("GPS(%d): Mean Reward: %r, Best-Test-Loss: %r, Best-Train-Loss: %r, Time: %r sec. "
                   "{Samples: %r, CB-Oracle: %r}" % (reward_id, mean_reward, round(info["cb-best-test-loss"], 4),
                                                     round(info["cb-best-train-loss"], 4), round(gps_step_time, 2),
                                                     round((sample_time * 100.0) / gps_step_time, 2),
                                                     round((oracle_time * 100.0) / gps_step_time, 2)))

        return learned_policy, mean_reward, {"prev_policy_index": best_homing_policy_index}
