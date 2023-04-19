import time
import torch
import random
import numpy as np
import learning.learning_utils.policy_evaluate as policy_evaluate

from collections import deque
from utils.cuda import cuda_var
from utils.average import AverageUtil
from utils.tensorboard import Tensorboard
from model.policy.open_loop import OpenLoopPolicy
from learning.tabular_rl.value_iteration import ValueIteration
from learning.core_learner.ppe_util import PPEDebugger, ErrorUtil
from environments.cerebral_env_meta.environment_keys import EnvKeys
from learning.learning_utils.count_probability import CountProbability
from learning.tabular_rl.det_tabular_mdp_builder import DetTabularMDPBuilder
from learning.learning_utils.generic_train_classifier import GenericTrainClassifier


class PPE:
    """
        PPE algorithm (Efroni et al., 2021)
    """

    def __init__(self, exp_setup):

        self.config = exp_setup.config
        self.constants = exp_setup.constants
        self.logger = exp_setup.logger
        self.debug = exp_setup.debug
        self.experiment = exp_setup.experiment

        self.classifier_type = self.constants["classifier_type"]

        # Train encoding function
        self.train_classifier = GenericTrainClassifier(exp_setup)

        self.ppe_debugger = PPEDebugger(exp_setup=exp_setup)

    @staticmethod
    def _generate_dataset(env, path_map, num_samples):

        path_id_to_state_map = dict()

        dataset = []
        num_samples_per_path = num_samples // len(path_map)

        assert num_samples_per_path >= 1, "Too few samples to cover all %d many paths" % len(path_map)

        for path_id, path in path_map.items():
            for _ in range(0, num_samples_per_path):
                obs, endo_state, reward = PPE._follow_path(env, path)
                dataset.append((obs, path_id, endo_state, reward))
                path_id_to_state_map[path_id] = endo_state

        for _ in range(10):
            random.shuffle(dataset)

        return dataset, path_id_to_state_map

    @staticmethod
    def _follow_path(env, path):

        obs, info = env.reset(generate_obs=False)
        reward = None
        path_len = path.num_timesteps()

        for h in range(0, path_len):
            action = path.sample_action(obs, h)
            obs, reward, done, info = env.step(action, generate_obs=(h == path_len - 1))

        return obs, info[EnvKeys.ENDO_STATE], reward

    @staticmethod
    def log_dataset(dataset, prob, logger):

        prob = prob * prob.size(0)

        state_stats = dict()
        empirical_prob = dict()
        hashes = dict()

        for ix, dp in enumerate(dataset):
            obs, path_id, endo_state, _ = dp
            if endo_state not in state_stats:
                state_stats[endo_state] = CountProbability()
                empirical_prob[endo_state] = []
                hashes[endo_state] = deque(maxlen=2)

            state_stats[endo_state].add(path_id)
            empirical_prob[endo_state].append(prob[ix].view(1, -1))
            hashes[endo_state].append(hash(str(obs)))

        logger.log("Dataset Statistics.")
        for endo_state, state_prob in state_stats.items():

            empirical_prob_ = torch.cat(empirical_prob[endo_state], dim=0).mean(0)
            num_paths = empirical_prob_.size(0)
            entries = set(state_prob.get_entries())
            learned_prob_str = "{%s" % ("; ".join(["%r: %f" % (entry_, empirical_prob_[entry_].item())
                                                   for entry_ in sorted(entries)]))
            suffix = []
            for ix in range(0, num_paths):
                val = empirical_prob_[ix].item()
                if ix in entries or val < 0.05:
                    continue
                else:
                    suffix.append("%r: %f" % (ix, val))

            learned_prob_str = learned_prob_str + "--" + ("; ".join(suffix)) + "}"

            logger.log("%r. Hashes %r" % (endo_state, hashes[endo_state]))
            logger.log("   using state %s" % state_prob)
            logger.log("   learned prob %s" % learned_prob_str)

        return state_stats

    @staticmethod
    def _get_reward_stats(dataset):

        mean_reward_path = dict()
        max_reward = -float('inf')
        for dp in dataset:

            _, path_id, _, reward = dp
            max_reward = max(max_reward, reward)

            if path_id in mean_reward_path:
                mean_reward_path[path_id].acc(reward)
            else:
                mean_reward_path[path_id] = AverageUtil(reward)

        return mean_reward_path, max_reward

    @staticmethod
    def _extract_reward_policy(model, q_val, init_state):

        # Given initial state, q_value and deterministic model
        # Extract the optimal reward-sensitive open loop policy
        state = init_state
        actions = []
        for h in range(0, model.horizon):

            action_id = q_val[(h, state)].argmax()
            action = model.actions[action_id]
            actions.append(action)

            next_states = model.get_transitions(state, action)
            assert len(next_states) == 1, "Deterministic model cannot allow transition to more than one state."
            state, _ = next_states[0]

        return OpenLoopPolicy(actions=actions)

    def train(self, env, exp_id=1, opt_reward=False):
        """ Execute Path Prediction Algorithm on an environment using

        :param env: Environment
        :param exp_id: Experiment ID, used to separate different independent runs
        :param opt_reward: If True then optimize the environment reward, else run in reward-free mode
        :return:
        """

        horizon = self.config["horizon"]
        actions = self.config["actions"]
        num_samples = self.constants["encoder_training_num_samples"]
        tensorboard = Tensorboard(log_dir=self.config["save_path"])

        homing_policies = dict()            # Contains a set of homing policies for every time step

        model = DetTabularMDPBuilder(actions=actions, horizon=horizon, gamma=1.0)      # Model for tabular MDP
        model.add_state(state=(0, 0), timestep=0)
        max_reward = -float('inf')

        abstract_to_state_map = {0: "sinit"}
        error_util = ErrorUtil()
        debugging_metrics = dict()

        for step in range(1, horizon + 1):

            self.logger.log("Running Path Prediction Algorithm: Step %r out of %r " % (step, horizon))

            homing_policies[step] = []      # Homing policies for this time step

            prev_paths = [OpenLoopPolicy(path_id=0)] if step == 1 else homing_policies[step - 1]
            path_map = dict()
            for prev_id, prev_path in enumerate(prev_paths):
                for action_id, action in enumerate(actions):
                    path_id = prev_id * len(actions) + action_id
                    path_map[path_id] = prev_path.extend(action, path_id=path_id)
                    self.logger.log("Path ID %d: %r -> %s" % (path_id, abstract_to_state_map[prev_path.path_id],
                                                              env.act_to_str(action)))

            # num_samples = max(10 * len(path_map), 5000)
            # self.logger.log("Number of samples changed to %d to have at least 100 samples per path" % num_samples)

            # Step 1: Create dataset for learning the encoding function. A single datapoint consists of
            #         an observation x paired with an id of roll-in policy chosen from policy cover of previous time
            #         step and an action taken at the last time step. Other miscellaneous information such as identity
            #         of endogenous state are provided.
            time_start = time.time()
            dataset, path_id_to_state_map = self._generate_dataset(env, path_map, num_samples)
            self.logger.log("Encoder: %r samples collected in %r sec" % (num_samples, time.time() - time_start))

            # Step 2: Train a classifier on the dataset to predict id of roll-in policy
            #         given an observation for this time step
            time_encoder_start = time.time()
            classifier, _ = self.train_classifier.do_train(
                model_type="classifier",
                model_name=self.classifier_type,
                num_class=len(path_map),
                dataset=dataset,
                logger=self.logger,
                tensorboard=tensorboard
            )
            # self.util.save_encoder_model(classifier, experiment, trial, step, "backward")
            self.logger.log("Encoder: Training time %r" % (time.time() - time_encoder_start))

            prob = self.train_classifier.get_class_mean_prob(classifier, dataset)     # Dim x num_class

            state_stats = None if not self.debug else self.log_dataset(dataset, prob, self.logger)

            # Step 3: Remove redundant paths
            path_common = {path_id: {path_id} for path_id in range(0, len(path_map))}

            elim_param = 5.0 / (8 * len(actions) * len(prev_paths))
            self.logger.log("Setting elim_param to %f as N=%d and K=%d" % (elim_param, len(path_map), len(actions)))

            for i in range(0, len(path_map)):

                if i not in path_common:
                    continue

                for j in range(i + 1, len(path_map)):

                    if j not in path_common:
                        continue
                    else:
                        sim = abs(prob[:, i] - prob[:, j]).sum().item()
                        if sim < elim_param:
                            # Eliminate the higher index and merge the two sets of equivalent paths
                            path_common[i] = path_common[i].union(path_common[j])
                            del path_common[j]

            abstract_to_state_map = dict()

            # Update the latent model
            for path_id, other_path_ids in path_common.items():

                abstract_to_state_map[path_id] = path_id_to_state_map[path_id]

                model.add_state(state=(step, path_id), timestep=step)
                homing_policies[step].append(path_map[path_id])
                prev_state = (step - 1, path_map[path_id].parent_path_id)
                model.add_transition(prev_state, path_map[path_id].action, (step, path_id))

                for other_id in other_path_ids:
                    if other_id == path_id:
                        continue
                    other_state = (step - 1, path_map[other_id].parent_path_id)
                    model.add_transition(other_state, path_map[other_id].action, (step, path_id))

            # Add reward statistics to the MDP
            mean_reward_path, max_reward_ = self._get_reward_stats(dataset)
            max_reward = max(max_reward, max_reward_)
            for path_id, reward in mean_reward_path.items():
                path = path_map[path_id]
                state = (step - 1, path.parent_path_id)
                model.add_reward(state, path.action, reward.get_mean())

            num_states = len(set([dp[2] for dp in dataset]))
            self.logger.log("PPE (Time step %d): Found %d many real states while exploring" % (step, num_states))
            self.logger.log("PPE (Time step %d): Added %d many paths to policy cover" % (step, len(path_common)))
            self.logger.log("PPE (Time step %d): Error %s" % (step, error_util))
            self.logger.log("PPE (Time step %d): Total time taken %r" % (step, time.time() - time_start))

            # Step 5 (Optional): Debug the data and model. This can take a long time and so should be performed
            # ony when debugging the algorithm.
            if self.debug:
                state_decoder = PPEDecoder(self.config, classifier, path_common)
                debugging_metrics[step] = self.ppe_debugger.debug(path_map, prob, path_common, env,
                                                                  abstract_to_state_map, step, error_util, dataset,
                                                                  state_stats, elim_param, state_decoder,
                                                                  homing_policies)

        # Finalize the model so no further changes can be directly done via API
        model.finalize()

        if not opt_reward:

            return {
                "coverage": horizon,
                "error1": error_util.error1,
                "error2": error_util.error2,
                "debugging_metrics": debugging_metrics
            }
        else:

            self.logger.log("Reward Sensitive Learning: Computing the optimal policy for the environment reward.")

            # Compute the optimal policy
            plan_time = time.time()
            value_it = ValueIteration()
            q_val = value_it.do_value_iteration(tabular_mdp=model, min_reward_val=0.0)
            expected_ret = q_val[(0, (0, 0))].max()
            v_star = env.get_optimal_value()

            self.logger.log("Value Iteration performed. Expected policy value is %f. V* is %r. "
                            "Max observed reward is %f" % (expected_ret, v_star, max_reward))
            self.logger.log("Reward Sensitive Learning: Time %r" % (time.time() - plan_time))

            learned_policy = self._extract_reward_policy(model, q_val, init_state=(0, 0))
            self.logger.log("Actual: Total number of episodes used %d. Total return %f. Mean return %f" %
                       (env.num_eps, env.sum_return, env.get_mean_return()))

            # Evaluate the optimal policy
            results = policy_evaluate.evaluate(env, learned_policy, horizon, self.logger,
                                               env.num_eps, env.sum_return, regret=True)

            results.update({
                "coverage": horizon,
                "q_val": expected_ret,
                "v_star": v_star,
                "max_reward": max_reward,
                "error1": error_util.error1,
                "error2": error_util.error2,
                "debugging_metrics": debugging_metrics
            })

            return results


class PPEDecoder:

    def __init__(self, config, classifier, path_common):
        self.config = config
        self.classifier = classifier
        self. path_common = path_common

    def encode_observations(self, observations):

        observations = cuda_var(torch.from_numpy(np.array(observations))).float().view(1, -1)
        prob = self.classifier.gen_prob(observations)[0][0]       # should be equal to number of classes

        best_id = -1
        best_val = -1

        for i in self.path_common:
            if prob[i] > best_val:
                best_val = prob[i]
                best_id = i

        return best_id


class PPEDecoder2:

    def __init__(self, config, classifier, epsilon):
        self.config = config
        self.classifier = classifier
        self.epsilon = epsilon

    def encode_observations(self, observations):

        observations = cuda_var(torch.from_numpy(np.array(observations))).float().view(1, -1)
        prob = self.classifier.gen_prob(observations)[0][0]       # should be equal to number of classes
        ixs = (prob - (prob.max() - self.epsilon)) >= 0.0

        choices = set()
        for i in range(0, ixs.size(0)):
            if ixs[i].item():
                choices.add(i)

        return min(choices)
