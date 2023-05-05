import time
import os

from environments.cerebral_env_meta.environment_wrapper import (
    GenerateEnvironmentWrapper,
)
from learning.policy_search.abstract_policy_search import AbstractPolicySearch
from learning.learning_utils.contextual_bandit_oracle import ContextualBanditOracle
from model.policy.stationary_deterministic_policy import StationaryDeterministicPolicy


class PSDP(AbstractPolicySearch):
    """Solve a reinforcement learning problem using PSDP (Bagnell et al.)"""

    def __init__(self, config, constants):
        AbstractPolicySearch.__init__(self)

        self.config = config
        self.constants = constants
        self.gamma = config["gamma"]
        self.num_train_samples = constants["psdp_training_num_samples"]
        self.num_eval_samples = constants["eval_homing_policy_sample_size"]
        self.contextual_bandit_oracle = ContextualBanditOracle(config, constants)

    @staticmethod
    def do_train(
        config,
        constants,
        homing_policy_dataset,
        env_info,
        policy_folder_name,
        actions,
        horizon,
        encoder_reward_args,
        homing_policies,
        logger,
        debug,
    ):
        psdp = PSDP(config, constants)
        env = GenerateEnvironmentWrapper.make_env(env_info[0], config, env_info[1])

        if encoder_reward_args is None:
            reward_func = None
        else:
            encoding_function, reward_id = encoder_reward_args
            reward_func = (
                lambda obs, h: 1
                if h == horizon
                and encoding_function.encode_observations(obs) == reward_id
                else 0
            )

        learned_policy, mean_reward, _ = psdp.train(
            homing_policy_dataset,
            env,
            actions,
            horizon,
            reward_func,
            homing_policies,
            logger,
            None,
            debug,
        )

        # Save the learned policy to disk
        PSDP._save_policy(learned_policy, policy_folder_name, horizon)

    @staticmethod
    def _save_policy(learned_policy, policy_folder_name, horizon):
        """Save the learned policy to disk"""

        for i in range(1, horizon + 1):
            if not os.path.exists(policy_folder_name):
                os.makedirs(policy_folder_name)
            learned_policy[i].save(
                folder_name=policy_folder_name, model_name="step_%d" % i
            )

    def read_policy(
        self, policy_folder_name, horizon, previous_step_homing_policy, delete=False
    ):
        """Read the policy from the disk"""

        homing_policy = dict()
        for j in range(1, horizon + 1):
            policy = StationaryDeterministicPolicy(self.config, self.constants)
            policy.load(folder_name=policy_folder_name, model_name="step_%d" % j)

            if delete:
                # Delete the file after reading for saving disk space
                os.remove(policy_folder_name + "step_%d" % j)

            homing_policy[j] = policy

        return homing_policy

    def train(
        self,
        replay_memory,
        env,
        actions,
        horizon,
        reward_func,
        homing_policies,
        logger,
        tensorboard,
        debug=False,
    ):
        """Performs the learning"""

        (
            time_start,
            sample_time,
            oracle_time,
        ) = (
            time.time(),
            0,
            0,
        )
        learned_policy = dict()

        logger.log("Doing PSDP with Horizon %r." % horizon)

        sum_rewards = 0.0

        for step in range(horizon, 0, -1):
            # Learn the optimal policy for this step
            sample_start = time.time()
            contextual_bandit_dataset = [
                self._gather_sample(
                    env,
                    actions,
                    horizon,
                    step,
                    homing_policies,
                    learned_policy,
                    reward_func,
                )
                for _ in range(0, self.num_train_samples)
            ]
            sample_time += time.time() - sample_start

            for dp in contextual_bandit_dataset:
                sum_rewards += dp[3]

            # Call contextual bandit oracle
            oracle_start = time.time()
            (
                optimal_policy_step,
                info,
            ) = self.contextual_bandit_oracle.learn_optimal_policy(
                contextual_bandit_dataset, logger, tensorboard, debug
            )
            oracle_time += time.time() - oracle_start

            learned_policy[step] = optimal_policy_step

            if debug:
                logger.log(
                    "Step %d CB: Best Test Loss %f, Best Train Loss %f, and Stopping max_epoch %d"
                    % (
                        step,
                        round(info["cb-best-test-loss"], 4),
                        round(info["cb-best-train-loss"], 4),
                        info["stopping_epoch"],
                    )
                )
                self.log_intermediate_results(
                    contextual_bandit_dataset,
                    optimal_policy_step,
                    env,
                    actions,
                    horizon,
                    step,
                    homing_policies,
                    learned_policy,
                    reward_func,
                    logger,
                    self.num_eval_samples,
                    self.constants["cb_oracle_batch_size"],
                )

        mean_reward = self._evaluate__learned_policy(
            env,
            horizon,
            reward_func,
            learned_policy,
            logger,
            num_rollouts=self.num_eval_samples,
            actions=self.config["actions"],
        )

        psdp_step_time = time.time() - time_start
        logger.log(
            "PSDP cost is %r sec. {Samples: %r, CB-Oracle: %r}"
            % (
                psdp_step_time,
                (sample_time * 100.0) / psdp_step_time,
                (oracle_time * 100.0) / psdp_step_time,
            )
        )

        total_episodes = horizon * self.num_train_samples

        return (
            learned_policy,
            mean_reward,
            {"sum_rewards": sum_rewards, "total_episodes": total_episodes},
        )
