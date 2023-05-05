import pdb
import time
import numpy as np

from utils.beautify_time import beautify
from learning.core_learner.safe_ppo_learner import SafePPOLearner
from learning.learning_utils.linear_disag_model import LinearDisagModel


class Sabre:
    # Currently, we only handle linear safety features

    def __init__(self, exp_setup):
        self.n = exp_setup.constants["sabre_n"]
        self.b = exp_setup.constants["sabre_b"]
        self.m = exp_setup.constants["sabre_m"]
        self.num_eval = exp_setup.constants["sabre_eval"]
        self.finetune = exp_setup.constants["sabre_finetune"] > 0

        self.horizon = exp_setup.config["horizon"]
        self.actions = exp_setup.config["actions"]
        self.num_actions = len(self.actions)
        self.safe_action = exp_setup.config["stop_action"]
        self.logger = exp_setup.logger
        self.experiment = exp_setup.experiment

        self.rl_method = SafePPOLearner(exp_setup)

    def _generate_episodes(
        self, env, policy, policy_disag_model, num_eps, evaluate=False
    ):
        returns = []
        counts = {
            "SafeAction": 0,
            "UnsafeAction": 0,
            "UnsafeSafeAction": 0,
            "Switch": 0,
            "Continue": 0,
        }
        states = []

        total_act = 0
        total_masked_act = 0
        total_safe_masked_act = 0
        total_safe_act = 0

        for i in range(num_eps):
            state, info = env.reset()
            states.append(info["state"])
            return_ = 0.0

            # traj = "%r" % (info["state"],)

            for t in range(1, self.horizon + 1):
                safety_ftrs = env.get_safety_ftrs(info["state"])

                # Calculate the masks for every state the agent visits once, and store it
                masks = np.array(
                    [
                        not policy_disag_model.is_surely_safe(
                            safety_ftr=safety_ftrs[action], action=action
                        )
                        for action in range(self.num_actions)
                    ]
                ).astype(np.float32)

                total_masked_act += masks.sum()
                total_act += self.num_actions

                for action in range(self.num_actions):
                    if env.safety_query(safety_ftrs[action], save=False):
                        total_safe_act += 1
                        # An action which is safe but is masked
                        if masks[action] == 1:
                            total_safe_masked_act += 1

                # Running ppo policy:
                action, _ = policy.act(state, masks, evaluate)

                if action == env.stop_action:
                    act_str = "SafeAction"
                elif action == env.unsafe_actions[t - 1]:
                    if info["state"][0] != 3:
                        act_str = "UnsafeAction"
                        self.logger.log("Warning: Taking an unsafe action")
                    else:
                        act_str = "UnsafeSafeAction"
                elif action == env.switch_actions[t - 1]:
                    act_str = "Switch"
                else:
                    act_str = "Continue"
                counts[act_str] += 1
                # print("Episode %i: State=%r, Action=%s, Masks=%r" % (i, info["state"], act_str, masks))

                next_state, reward, done, info = env.step(action.item())

                if t < self.horizon:
                    states.append(info["state"])

                # traj += "-%r" % (info["state"],)

                state = next_state
                return_ += reward

                if done:
                    break
            # self.logger.log("Episode %d: Traj %s" % (i, traj))

            returns.append(return_)

        meta_dict = {
            "Total actions": total_act,
            "Total masked actions pct": (total_masked_act * 100.0)
            / float(max(1, total_act)),
            "Total Safe actions": total_safe_act,
            "Total Masked Safe actions pct": (total_safe_masked_act * 100.0)
            / float(max(1, total_safe_act)),
        }

        return states, returns, counts, meta_dict

    def _create_new_safety_dataset(
        self, env, policy, policy_disag_model, reward_disag_model, inc=True
    ):
        dataset = []
        oracle_calls = 0
        states, _, counts, _ = self._generate_episodes(
            env, policy, policy_disag_model, self.m
        )
        self.logger.log("SABRE batch: %r" % counts)

        # state_count = dict()
        # for state in states:
        #     if state not in state_count:
        #         state_count[state] = 0
        #     state_count[state] += 1
        #
        # for state in state_count:
        #     self.logger.log("State %r: Visited %d many times" % (state, state_count[state]))

        for state in states:
            safety_ftrs = env.get_safety_ftrs(state)
            for action in self.actions:
                if reward_disag_model.in_region_of_disag(safety_ftrs[action], action):
                    # query the environment's safety model
                    gold_safety_label = env.safety_query(safety_ftrs[action])
                    y = 1.0 if gold_safety_label else -1.0
                    oracle_calls += 1
                    dataset.append((safety_ftrs[action], y, action))

                    if inc:
                        # Incremental update to reward disagreement model
                        reward_disag_model.update(safety_ftrs[action], y)

        return dataset, oracle_calls

    def _evaluate(self, env, policy, policy_disag_model):
        _, returns, counts, meta_dict = self._generate_episodes(
            env, policy, policy_disag_model, self.num_eval, evaluate=True
        )

        results = {
            "Num_Eval": len(returns),
            "Avg_Return": np.mean(returns),
            "Std": np.std(returns),
            "Max": np.max(returns),
            "Min": np.min(returns),
        }

        for key, val in counts.items():
            results[key] = val

        self.logger.log(
            "SABRE: Evaluated Policy on %d Episodes. Mean return %f (Std %f, Max %f, Min %f)"
            % (
                results["Num_Eval"],
                results["Avg_Return"],
                results["Std"],
                results["Max"],
                results["Min"],
            )
        )

        self.logger.log("SABRE: Evaluate Policy took actions %r" % counts)

        self.logger.log(
            "SABRE: %s"
            % " ".join(
                "%s: %r" % (key, val) for (key, val) in sorted(meta_dict.items())
            )
        )

        return results

    def train(self, env, exp_id, reward_only=False):
        self.logger.log(
            "Starting SABRE (Experiment ID: %d, Reward Only %r)" % (exp_id, reward_only)
        )

        dataset = []
        total_oracle_calls = 0
        policy = None

        for i in range(self.n):
            self.logger.log("SABRE: [Policy Loop %d out of %d]" % (i + 1, self.n))

            # Define policy disagreement model
            policy_disag_model = LinearDisagModel(dataset, self.safe_action)

            for b in range(self.b):
                self.logger.log("SABRE: [Reward Loop %d out of %d]" % (b + 1, self.b))

                # Define reward disagreement model
                reward_disag_model = LinearDisagModel(dataset, self.safe_action)

                # num_safe, pct = reward_disag_model.test(dataset[500:])
                # self.logger("Testing: Total safe features were %d of which %f %% were marked safe" % (num_safe, pct))

                # Call Blackbox RL algorithm to find a policy
                time_s = time.time()
                self.logger.log("SABRE: Calling Blackbox RL method]")
                policy = self.rl_method.do_train(
                    env=env,
                    policy_disag_model=policy_disag_model,
                    reward_disag_model=None if reward_only else reward_disag_model,
                    policy=policy if self.finetune or reward_only else None,
                    max_episodes=1000,
                )  # Run safety reward
                self.logger.log(
                    "SABRE: Blackbox RL Call Completed. Time taken %s"
                    % beautify(time.time() - time_s)
                )

                # Create new safety dataset
                time_s = time.time()
                self.logger.log("SABRE: Creating New Safety Dataset.")
                new_dataset, oracle_calls_ = self._create_new_safety_dataset(
                    env, policy, policy_disag_model, reward_disag_model
                )
                dataset.extend(new_dataset)
                total_oracle_calls += oracle_calls_

                self.logger.log(
                    "SABRE: Dataset Created. %d New points were added. Total dataset size %d, Time taken %s"
                    % (len(new_dataset), len(dataset), beautify(time.time() - time_s))
                )

                self.logger.log(
                    "SABRE: %d New calls to Safety Oracle were made. Total calls made %d"
                    % (oracle_calls_, total_oracle_calls)
                )

                self.logger.log(
                    "SABRE: Total Lp Calls %d, Total Optimization Failures %d, Lp Time taken %s"
                    % (
                        LinearDisagModel.TOTAL_CALL,
                        LinearDisagModel.TOTAL_OPT_FAILURE,
                        beautify(LinearDisagModel.TOTAL_TIME),
                    )
                )

            # TODO: remove evaluating each outer iteration after debugging.
            # if i < self.n - 1:
            #     self._evaluate(env, policy, policy_disag_model)

        policy_disag_model = LinearDisagModel(dataset, self.safe_action)

        run_eps = 10000 - self.n * self.b * (1000 + self.m)

        # Learn a policy to optimize the environment reward
        policy = self.rl_method.do_train(
            env=env,
            policy_disag_model=policy_disag_model,
            reward_disag_model=None,
            policy=policy if reward_only else None,
            max_episodes=run_eps,
        )  # Run on environment reward

        # Evaluate the policy
        results = self._evaluate(env, policy, policy_disag_model)

        self.logger.log(
            "Sabre experiment over. Total oracle calls %d, "
            "Total number of unsafe actions taken %d"
            % (total_oracle_calls, env.num_unsafe_actions)
        )

        results["Total_Lp_Calls"] = LinearDisagModel.TOTAL_CALL
        results["Total_Lp_Failure"] = LinearDisagModel.TOTAL_OPT_FAILURE
        results["Total_Lp_Time"] = LinearDisagModel.TOTAL_TIME
        results["Total_Oracle_Calls"] = total_oracle_calls
        results["Total_Unsafe_Actions"] = env.num_unsafe_actions

        # Save disagreement model
        np.savez(
            "%s/policy_disag_model.npy" % self.experiment,
            A=policy_disag_model.A,
            b=policy_disag_model.b,
        )

        return results

    def do_train_from_disag(self, env, model_fname):
        policy_disag_model = LinearDisagModel.load_from_file(
            model_fname, self.safe_action
        )

        # Learn a policy to optimize the environment reward
        policy = self.rl_method.do_train(
            env=env, policy_disag_model=policy_disag_model, reward_disag_model=None
        )  # Run on environment reward

        # Evaluate the policy
        results = self._evaluate(env, policy, policy_disag_model)

        self.logger.log(
            "Sabre experiment over. Total oracle calls %d, "
            "Total number of unsafe actions taken %d" % (0, env.num_unsafe_actions)
        )

        results["Total_Lp_Calls"] = LinearDisagModel.TOTAL_CALL
        results["Total_Lp_Failure"] = LinearDisagModel.TOTAL_OPT_FAILURE
        results["Total_Lp_Time"] = LinearDisagModel.TOTAL_TIME
        results["Total_Oracle_Calls"] = 0
        results["Total_Unsafe_Actions"] = env.num_unsafe_actions

        return results
