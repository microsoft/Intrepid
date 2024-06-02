import torch
import random
import itertools
import numpy as np
import torch.optim as optim
import utils.generic_policy as gp

from learning.learning_utils.entropy_decay_policy import EntropyDecayPolicy
from learning.datastructures.count_probability import CountProbability
from learning.datastructures.count_conditional_probability import CountConditionalProbability
from learning.learning_utils.encoder_sampler_wrapper import EncoderSamplerWrapper
from learning.learning_utils.factorl_graph_identification import FactoRLGraphIdentification
from learning.datastructures.transition import TransitionDatapoint
from model.policy.stationary_decoder_dictionary_policy import StationaryDecoderLatentPolicy
from model.transition_encoders.factorl_encoder import FactoRLEncoder
from utils.average import AverageUtil
from utils.cuda import cuda_var
from utils.tensorboard import Tensorboard


class TrainModel:

    def __init__(self, config, constants):

        self.config = config
        self.constants = constants
        self.epoch = 10     # constants["encoder_training_epoch"]
        self.learning_rate = constants["encoder_training_lr"]
        self.batch_size = constants["encoder_training_batch_size"]
        self.validation_size_portion = constants["validation_data_percent"]
        self.entropy_coeff = constants["entropy_reg_coeff"]
        self.num_homing_policies = constants["num_homing_policy"]
        self.patience = constants["patience"]
        self.max_retrials = constants["max_try"]
        self.expected_optima = constants["expected_optima"]  # If the model reaches this loss then we exit

        self.entropy_coeff = constants["entropy_reg_coeff"]

    def _calc_loss_from_dataset(self, model, batch, epoch, discretized, test_set_errors=None, past_entropy=None):

        prev_observations = cuda_var(torch.cat([torch.from_numpy(np.array(point.get_curr_obs())).view(1, -1)
                                                for point in batch], dim=0)).float()
        actions = cuda_var(torch.cat([torch.from_numpy(np.array(point.get_action())).view(1, -1)
                                      for point in batch], dim=0)).long()
        observations = cuda_var(torch.cat([torch.from_numpy(np.array(point.get_next_obs())).view(1, -1)
                                           for point in batch], dim=0)).float()
        gold_labels = cuda_var(torch.cat([torch.from_numpy(np.array(point.is_valid())).view(1, -1)
                                          for point in batch], dim=0)).long()

        # Compute loss
        log_probs, meta_dict = model.gen_log_prob(prev_observations=prev_observations,
                                                  actions=actions,
                                                  observations=observations,
                                                  discretized=discretized)  # outputs a matrix of size batch x 2
        classification_loss = -torch.mean(log_probs.gather(1, gold_labels.view(-1, 1)))

        self.entropy_decay_policy = EntropyDecayPolicy(self.constants, epoch)
        decay_coeff = self.entropy_decay_policy.get_entropy_coeff(epoch, test_set_errors, past_entropy)

        if discretized:
            # For discretized models, there is an internal classification step representation by a probability
            # distribution that can be controlled using entropy bonus
            # NOT SUPPORTED AT THE MOMENT
            loss = classification_loss - self.entropy_coeff * decay_coeff * meta_dict["mean_entropy"]
        else:
            loss = classification_loss

        info_dict = dict()

        info_dict["classification_loss"] = classification_loss

        if discretized:
            info_dict["mean_entropy"] = meta_dict["mean_entropy"]
            info_dict["entropy_coeff"] = self.entropy_coeff * decay_coeff
        else:
            info_dict["mean_entropy"] = 0.0
            info_dict["entropy_coeff"] = 0.0

        return loss, info_dict

    def train_model(self, dataset, factor_obs_dim, logger, discretized, debug, tensorboard):

        # torch.manual_seed(ctr)
        print("Solving dataset with stats %r" % (len(dataset)))

        # Current model
        model = FactoRLEncoder(2, factor_obs_dim, self.config, self.constants)

        # Model for storing the best model as measured by performance on the test set
        best_model = FactoRLEncoder(2, factor_obs_dim, self.config, self.constants)

        param_with_grad = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params=param_with_grad, lr=self.learning_rate)

        random.shuffle(dataset)
        dataset_size = len(dataset)
        batches = [dataset[i:i + self.batch_size] for i in range(0, dataset_size, self.batch_size)]

        train_batch = int((1.0 - self.validation_size_portion) * len(batches))
        train_batches = batches[:train_batch]
        test_batches = batches[train_batch:]

        best_test_loss, best_epoch, train_loss = 0.69, -1, 0.69  # 0.69 is -log(2)
        num_train_examples, num_test_examples = 0, 0
        patience_counter = 0

        test_set_errors, past_entropy = [], []

        for epoch_ in range(1, self.epoch + 1):

            train_loss, mean_entropy, num_train_examples = 0.0, 0.0, 0
            for train_batch in train_batches:

                loss, info_dict = self._calc_loss_from_dataset(model, train_batch, epoch_, discretized,
                                                               test_set_errors, past_entropy)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 40)
                optimizer.step()

                loss = float(loss)
                tensorboard.log_scalar("Encoding_Loss ", loss)

                for key in info_dict:
                    tensorboard.log_scalar(key, info_dict[key])

                batch_size = len(train_batch)
                train_loss = train_loss + float(info_dict["classification_loss"]) * batch_size
                mean_entropy = mean_entropy + float(info_dict["mean_entropy"]) * batch_size
                num_train_examples = num_train_examples + batch_size

            train_loss = train_loss / float(max(1, num_train_examples))
            mean_entropy = mean_entropy / float(max(1, num_train_examples))

            # Evaluate on test batches
            test_loss = 0
            num_test_examples = 0
            for test_batch in test_batches:
                _, info_dict = self._calc_loss_from_dataset(model, test_batch, epoch_, discretized,
                                                            test_set_errors, past_entropy)

                batch_size = len(test_batch)
                test_loss = test_loss + float(info_dict["classification_loss"]) * batch_size
                num_test_examples = num_test_examples + batch_size

            test_loss = test_loss / float(max(1, num_test_examples))
            logger.debug("Train Loss after max_epoch %r is %r, mean entropy %r, entropy coeff %r" %
                         (epoch_, round(train_loss, 2), round(mean_entropy, 2), info_dict["entropy_coeff"]))
            logger.debug("Test Loss after max_epoch %r is %r" % (epoch_, round(test_loss, 2)))

            test_set_errors.append(test_loss)
            past_entropy.append(mean_entropy)

            if test_loss < best_test_loss:
                patience_counter = 0
                best_test_loss = test_loss
                best_epoch = epoch_
                best_model.load_state_dict(model.state_dict())
            else:
                # Check patience condition
                patience_counter += 1  # number of max_epoch since last increase
                if best_test_loss < self.expected_optima or test_loss > 0.8:  # Found good solution or diverged
                    break

                if patience_counter == self.patience:
                    logger.log("Patience Condition Triggered: No improvement for %r epochs" % patience_counter)
                    break

        logger.log("FactoRL(Discretized: %r), Train/Test = %d/%d, Best Tune Loss %r at max_epoch %r, "
                   "Train Loss after %r epochs is %r " % (discretized, num_train_examples,
                                                          num_test_examples, round(best_test_loss, 2),
                                                          best_epoch, epoch_, round(train_loss, 2)))

        return best_model, best_test_loss

    @staticmethod
    def test_model(decoder, children_factor, dataset, k, logger):
        """ We have a decoder for the inferred children_factor. """

        correspondence = np.zeros((2, 2))           # TODO generalize to k-nary in future
        for dp in dataset:
            atoms = np.array([dp.next_obs[i] for i in children_factor])
            inf_factored_val = decoder.encode_observations(atoms)
            gold_factored_val = dp.next_state[k]
            correspondence[gold_factored_val, inf_factored_val] += 1

        # Normalized correspondence matrix. (i, j)^{th} entry contains P(inferred factor val is i|gold factor val is j)
        gold_val_counts = correspondence.sum(axis=1)
        gold_val_counts = np.expand_dims(gold_val_counts, 1).astype(np.float32)     # 2 x 1
        correspondence /= gold_val_counts                                           # Normalizes row

        logger.log("Learned Decoder Correspondence Matrix \n %r" % correspondence)

        # The model should map the states to either [[1, 0], [0, 1]] or [[0, 1], [1, 0]]
        # We define a score based on minimum of determinant of the correspondence matrix from these two matrices

        standard1 = np.eye(2)
        standard2 = np.array([[0, 1.0], [1.0, 0]])     # Decoder is label invariant

        min_diff = min(np.linalg.det(standard1 - correspondence), np.linalg.det(standard2 - correspondence))

        return min_diff


class CompoundDecoder:
    """ Decoder that takes a set of decoders for factors and uses it to decode the whole state """

    def __init__(self, decoders, children_factors):
        self.decoders = decoders
        self.children_factors = children_factors
        self.num_factors = len(self.children_factors)

    def get_num_factor(self):
        return self.num_factors

    def encode_observations(self, obs):

        learned_state = np.zeros(self.num_factors)
        for i, children_factor in enumerate(self.children_factors):
            var = np.array([obs[i] for i in children_factor], dtype=np.float32)
            val = self.decoders[i].encode_observations(var)
            learned_state[i] = val

        return learned_state


class CompoundModel:
    """
        Transition function class to map a factorized state and action to another factorized state
    """

    def __init__(self, model, factors):
        self.model = model
        self.factors = factors
        self.num_factors = len(factors)

    def get_num_factor(self):
        return self.num_factors

    def sample(self, state, action):

        new_state = []
        for i in range(self.num_factors):
            factor_val = tuple([state[pt] for pt in self.factors[i]])

            key = (factor_val, action)

            if key in self.model[i]:
                v = self.model[i][key].sample()
            else:
                # We don't have data for this input so output random bits
                # so we output equal probability
                v = random.randint(0, 1)

            new_state.append(v)

        return tuple(new_state)

    def get_probability(self, state, action, next_state):

        prod = 1.0
        for i, v in enumerate(next_state):
            factor_val = tuple([state[pt] for pt in self.factors[i]])

            key = (factor_val, action)

            if key in self.model[i]:
                prod_ = self.model[i][key].get_prob_entry(v)
            else:
                # We don't have data for this input so output random bits
                # so we output equal probability
                prod_ = 0.5

            prod *= prod_

        return prod


class FactoRL:
    """ A novel model-based reinforcement learning algorithm with PAC guarantees for Rich Observation Factored MDP. """

    SUCC_STR = "succ"

    # Maximum decoder error as measured by the smallest determinant value of the difference between the
    # correspondence matrix to the permutation matrix. The correspondence matrix M is defined as the matrix between
    # true state-factor values and learned state-factor values and M_{ij} denotes the probability that decoder maps
    # state-factor value i to learned value j. If the learned factor is perfect then for the case where factor values
    # are binary, the correspondence matrix M should either be identity or [[0, 1], [1, 0]]
    MAX_DECODER_ERROR = 0.1

    # Model error is measured by the total variation distance between the estimated model which is estimated by
    # each individual factor, and the total model estimated directly over state. This approach sadly wont work when
    # there are many factors, in which case, one should compare it with the gold model extracted from the environment.
    MAX_MODEL_ERROR = 0.1

    # When learning policy cover, we should ignore policies if their probability of succeeding to reach the desired
    # configuration is below this value. This should ideally be a function of the reachability parameter. For
    # deterministic environments, the reachability parameter is 1 so a value such as 0.5 is reasonable.
    MIN_POLICY_PROB = 0.5

    # Number of roll-outs to test policy cover
    POLICY_COVER_ROLLOUT = 10

    def __init__(self, exp_setup):

        self.config = exp_setup.config
        self.constants = exp_setup.constants

        self.logger = exp_setup.logger
        self.experiment = exp_setup.experiment
        self.debug = exp_setup.debug
        self.env_name = exp_setup.env_name

        # Sampler for generating data for training the encoding function
        self.encoder_sampler = EncoderSamplerWrapper(exp_setup.constants)

        self.state_dim = self.config["state_dim"]
        self.obs_dim = self.config["obs_dim"]
        assert self.obs_dim > 0, "observation dimension should be positive but found %d" % self.obs_dim
        self.actions = self.config["actions"]
        self.num_actions = len(self.actions)
        self.horizon = self.config["horizon"]
        self.max_parents = self.constants["max_parents"]
        self.num_factor_vals = self.constants["num_factor_vals"]
        self.num_samples = self.constants["encoder_training_num_samples"]
        self.train_model_util = TrainModel(exp_setup.config, exp_setup.constants)

    @staticmethod
    def _extract_atoms(obs, children_factor):
        """ Given a n-dimensional obs and a set of factors, return the selected vector """
        return np.array([obs[i] for i in children_factor])

    @staticmethod
    def _gather_sample(env, actions, step, homing_policies, selection_weights=None):
        """ Gather sample using ALL_RANDOM style """

        start_obs, meta = env.reset()
        if step > 1:

            if selection_weights is None:
                # Select a homing policy for the previous time step randomly uniformly
                ix = random.randint(0, len(homing_policies[step - 1]) - 1)
                policy = homing_policies[step - 1][ix]
            else:
                # Select a homing policy for the previous time step using the given weights
                # policy = random.choices(homing_policies[step - 1], weights=selection_weights, k=1)[0]
                ix = gp.sample_action_from_prob(selection_weights)
                policy = homing_policies[step - 1][ix]
            obs = start_obs

            for step_ in range(1, step):
                obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
                action = policy[step_ - 1].sample_action(obs_var)
                obs, reward, done, meta = env.step(action)

            current_obs = obs
        else:
            ix = None
            current_obs = start_obs

        if meta is not None and "state" in meta:
            curr_state = meta["state"]
        else:
            curr_state = None

        deviation_action = random.choice(actions)
        action_prob = 1.0 / float(max(1, len(actions)))

        next_obs, reward, done, meta = env.step(deviation_action)
        new_meta = meta

        if new_meta is not None and "state" in new_meta:
            next_state = new_meta["state"]
        else:
            next_state = None

        data_point = TransitionDatapoint(curr_obs=current_obs,
                                         action=deviation_action,
                                         next_obs=next_obs,
                                         y=1,
                                         curr_state=curr_state,
                                         next_state=next_state,
                                         action_prob=action_prob,
                                         policy_index=ix,
                                         step=step,
                                         reward=reward)

        return data_point

    @staticmethod
    def _gather_policy_cover_frontier(env, step, homing_policies, selection_weights=None):
        """ Gather sample using ALL_RANDOM style """

        start_obs, meta = env.reset()
        if step > 1:

            if selection_weights is None:
                # Select a homing policy for the previous time step randomly uniformly
                ix = random.randint(0, len(homing_policies[step - 1]) - 1)
                policy = homing_policies[step - 1][ix]
            else:
                # Select a homing policy for the previous time step using the given weights
                # policy = random.choices(homing_policies[step - 1], weights=selection_weights, k=1)[0]
                ix = gp.sample_action_from_prob(selection_weights)
                policy = homing_policies[step - 1][ix]
            obs = start_obs

            for step_ in range(1, step):
                obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
                action = policy[step_ - 1].sample_action(obs_var)
                obs, reward, done, meta = env.step(action)

            current_obs = obs
        else:
            current_obs = start_obs

        if meta is not None and "state" in meta:
            curr_state = meta["state"]
        else:
            curr_state = None

        return curr_state, current_obs

    @staticmethod
    def _generate_combinations(d, k):

        if type(d) == int:
            my_set = range(0, d)
        else:
            my_set = d

        it = itertools.combinations(my_set, 0)
        for k_ in range(1, k + 1):
            it_k_ = itertools.combinations(my_set, k_)
            it = itertools.chain(it, it_k_)

        return it

    def _generate_prod_values(self, d):
        return itertools.product(*([list(range(0, self.num_factor_vals))] * d))

    @staticmethod
    def _merge(indices_tup1, val1, indices_tup2, val2):

        it = list(sorted(itertools.chain(zip(indices_tup1, val1), zip(indices_tup2, val2)), key=lambda x: x[0]))

        joined_indices = tuple([item[0] for item in it])
        joined_val = tuple([item[1] for item in it])

        return joined_indices, joined_val

    def _est_model(self, dataset, curr_decoder, next_decoder, logger, use_gold_state=False):
        """
                                Estimate T(curr | prev, a)
        :param dataset:         Dataset of transitions
        :param curr_decoder:    Decoder for current time step
        :param next_decoder:    Decoder for next time step
        :return:    Estimated model in latent state, list of visited state, and dataset labeled with decoded state
        """

        # Pick all combinations of factors and their values in the reachable_set and every action and estimate the model
        labeled_dataset = []
        states_visited = set()
        for dp in dataset:
            curr_state = dp.curr_state if use_gold_state else curr_decoder.encode_observations(dp.curr_obs)
            next_state = dp.next_state if use_gold_state else next_decoder.encode_observations(dp.next_obs)
            labeled_dataset.append((curr_state, dp.action, next_state))
            states_visited.add(tuple(next_state))

        model = [dict() for _ in range(curr_decoder.num_factors)]
        num_factor = curr_decoder.get_num_factor()

        # Take every combination of 2k set and their values
        factor_combinations = list(self._generate_combinations(num_factor, 2 * self.max_parents))

        for labeled_dp in labeled_dataset:

            for factor_combination in factor_combinations:

                factor_val = tuple([labeled_dp[0][pt] for pt in factor_combination])     # \hat{S}[I]
                action = labeled_dp[1]
                # print("Stored model for ", action)

                for i in range(0, num_factor):

                    key = (factor_val, action)

                    if factor_combination not in model[i]:
                        model[i][factor_combination] = dict()

                    if key not in model[i][factor_combination]:
                        model[i][factor_combination][key] = CountProbability()

                    model[i][factor_combination][key].add(labeled_dp[2][i])

        # Find parent
        parent_candidates = list(self._generate_combinations(num_factor, self.max_parents))

        # List of parent factors and transition models for every factor
        parent_factors = []
        factor_model = []

        for i in range(0, num_factor):

            # Find parent for this bit
            best_parent = None
            best_width = float('inf')

            for parent_candidate in parent_candidates:

                width = 0.0
                control_candidates = list(
                    self._generate_combinations(set(range(0, num_factor)) - set(parent_candidate), self.max_parents))

                len_pc = len(parent_candidate)
                for parent_val in self._generate_prod_values(len_pc):

                    for control_group_1 in control_candidates:

                        len_cg1 = len(control_group_1)
                        for control_group_1_val in self._generate_prod_values(len_cg1):

                            for control_group_2 in control_candidates:

                                len_cg2 = len(control_group_2)
                                for control_group_2_val in self._generate_prod_values(len_cg2):

                                    for action in self.actions:

                                        joined_group_1, joined_val_1 = self._merge(parent_candidate, parent_val,
                                                                                   control_group_1, control_group_1_val)

                                        joined_group_2, joined_val_2 = self._merge(parent_candidate, parent_val,
                                                                                   control_group_2, control_group_2_val)

                                        key1 = (joined_val_1, action)
                                        key2 = (joined_val_2, action)

                                        # pdb.set_trace()

                                        if key1 not in model[i][joined_group_1] or key2 not in model[i][joined_group_2]:
                                            # print("Key1 Not found: Set %r, Value %r, Action %r, Index %d" % key1)
                                            # print("Key2 Not found: Set %r, Value %r, Action %r, Index %d" % key2)
                                            continue

                                        # print("Key1: Set %r, Value %r, Action %r, Index %d" % key1)
                                        # print("Key2: Set %r, Value %r, Action %r, Index %d" % key2)
                                        width_ = model[i][joined_group_1][key1].total_variation(
                                            model[i][joined_group_2][key2])
                                        # pdb.set_trace()
                                        width = max(width, width_)

                if width < best_width:
                    best_parent = parent_candidate
                    best_width = width

            parent_factors.append(best_parent)
            factor_model.append(model[i][best_parent])

            logger.log("Best parent for factor %d => %r" % (i, best_parent))

            # if i == 0:
            #     print("Values taken by s[0] ", set([dp[0][0] for dp in labeled_dataset]))
            #     print("Action set ", set([dp[1] for dp in labeled_dataset]))
            #     # print("For factor 0 and parent=() various combinations considered were ", model[0][()].keys())
            #     # print("For factor 0 and parent=(0) various combinations considered were ", model[0][(0,)].keys())

        estimated_model = CompoundModel(model=factor_model, factors=parent_factors)

        return estimated_model, states_visited, labeled_dataset

    @staticmethod
    def _test_model_error(labeled_dataset, estimated_model, logger):

        direct_prob = CountConditionalProbability()

        for (state, action, next_state) in labeled_dataset:
            direct_prob.add(entry=tuple(next_state), condition=(tuple(state), action))

        diff = 0.0
        for (state, action, next_state) in labeled_dataset:
            direct_prob_ = direct_prob.get_prob_entry(entry=tuple(next_state), condition=(tuple(state), action))
            model_prob_ = estimated_model.get_probability(tuple(state), action, tuple(next_state))
            diff += abs(model_prob_ - direct_prob_)

        diff /= float(max(1, len(labeled_dataset)))

        return diff

    def _plan(self, estimated_model, reward_fn, step, states_visited, decoder):

        learned_policy = [None] * (step + 1)
        q_values = [dict() for _ in range(step + 1)]

        # num_factors = estimated_model[0].get_num_factor()
        # state_set = list(self._generate_prod_values(num_factors))

        for h in range(step, -1, -1):

            # for state in state_set:
            for state in states_visited[h]:

                q_values[h][state] = np.zeros(self.num_actions)

                reward = reward_fn(state, h)

                # print("H = %d of %d, State = %r, Reward %f, Num entries %d" % (h, step, state, reward, len(q_values[h])))

                for action in self.actions:

                    if h == step:
                        q_values[h][state][action] = reward
                    else:

                        future_return = 0.0
                        # for next_state in state_set:
                        for next_state in states_visited[h + 1]:

                            prob_val = estimated_model[h].get_probability(state, action, next_state)
                            future_return += prob_val * q_values[h + 1][next_state].max()

                        q_values[h][state][action] = reward + future_return

            learned_policy[h] = StationaryDecoderLatentPolicy(decoder[h], q_values[h], self.actions)

        mean_total_reward = AverageUtil()
        for _ in range(0, FactoRL.POLICY_COVER_ROLLOUT):
            total_return, took_action = self._rollout(learned_policy, estimated_model, reward_fn, step)
            mean_total_reward.acc(total_return)

        # print("Mean total reward ", mean_total_reward)
        # pdb.set_trace()

        return learned_policy, mean_total_reward.get_mean(), took_action

    @staticmethod
    def _rollout(learned_policy, estimated_model, reward_fn, step):

        state_it = tuple([0] * estimated_model[0].get_num_factor())
        total_reward = 0.0
        # print("State: ", state_it)
        took_action = None

        for h in range(step + 1):

            reward = reward_fn(state_it, h)
            total_reward += reward
            action = learned_policy[h].latent_policy.get_argmax_action(state_it)
            # print("Step %d: Taking action %r and received reward %r" % (h, action, reward))

            if h == 0:
                took_action = action

            if h < step:
                state_it = estimated_model[h].sample(state_it, action)
                # print("New State ", state_it)

        return total_reward, took_action

    def _learn_policy_cover(self, estimated_model, step, states_visited, decoder, logger):
        """ Learn a policy cover time step step using the estimated model """

        num_factor = estimated_model[step - 1].get_num_factor()

        learned_policies = []
        factor_candidates = list(self._generate_combinations(num_factor, 2 * self.max_parents))

        action_stats = [0] * self.num_actions

        for factor_candidate in factor_candidates:

            len_pc = len(factor_candidate)
            for parent_val in self._generate_prod_values(len_pc):

                reward_fn = lambda state_, step_: 1.0 if step_ == step and all(
                    [state_[k] == parent_val[i] for i, k in enumerate(factor_candidate)]) else 0.0

                # print("Trying to learn Policy for Factored Candidate %r and Parent Val is %r" % (factor_candidate, parent_val))
                learned_policy, policy_val, took_action = self._plan(estimated_model, reward_fn,
                                                                     step, states_visited, decoder)

                if policy_val > FactoRL.MIN_POLICY_PROB:
                    logger.log("Added Policy for Reaching state[%r] = %r at time step %d" %
                               (factor_candidate, parent_val, step))
                    action_stats[took_action] += 1.0
                    learned_policies.append(learned_policy)

        # pdb.set_trace()

        return learned_policies

    def _test_policy_cover(self, env, policy_cover, step, logger):

        dataset = []

        for _ in range(2 * self.num_samples):
            dataset.append(
                self._gather_policy_cover_frontier(env, step, policy_cover))

        state_count = dict()

        for dp in dataset:

            state_key = tuple(dp[0])

            if state_key not in state_count:
                state_count[state_key] = 1
            else:
                state_count[state_key] += 1

        logger.log("Policy cover test: %d many unique states found at time step %d (starting from step=1)"
                   % (len(state_count), step + 1))

    def _create_nce_dataset(self, dataset, children_factor):

        nce_dataset = []

        for dp in dataset[:self.num_samples]:
            pos = dp.make_copy()
            pos.next_obs = self._extract_atoms(pos.next_obs, children_factor)
            nce_dataset.append(pos)

            neg = dp.make_copy()

            # Make neg_data fake by replacing last observation and state with a randomly chosen example
            j = random.randint(0, self.num_samples - 1)
            chosen_sample = dataset[j]

            neg.y = 0  # Marked as fake
            # Replaced last observation
            neg.next_obs = self._extract_atoms(chosen_sample.next_obs, children_factor)
            neg.next_state = chosen_sample.next_state  # Replaced last state
            neg.reward = None  # Reward for fake transition makes little sense
            nce_dataset.append(neg)

        return nce_dataset

    @staticmethod
    def _log_dataset_stats(dataset, logger):

        transition_count = dict()
        state_count = dict()

        for dp in dataset:

            if dp.is_valid():

                transition_key = (tuple(dp.curr_state), dp.action, tuple(dp.next_state))
                state_key = tuple(dp.next_state)

                if transition_key not in transition_count:
                    transition_count[transition_key] = 1
                else:
                    transition_count[transition_key] += 1

                if state_key not in state_count:
                    state_count[state_key] = 1
                else:
                    state_count[state_key] += 1

        logger.log("%d many unique states found" % len(state_count))
        logger.log("%d many unique transitions found" % len(transition_count))

    def train(self, env, exp_id=1, opt_reward=False, fail_early=True):

        """ Execute FactoRL algorithm on an environment using
        :param env:
        :param exp_id:
        :param opt_reward:
        :return:
        """

        use_gold_factor = False
        use_gold_decoder = False

        self.logger.log("Running FactoRL on %s with setting use_gold_factor: %r and use_gold_decoder: %r" %
                   (self.env_name, use_gold_factor, use_gold_decoder))
        self.logger.log("%s: Fixed actions %r" % (self.env_name, env.env.fixed_actions))

        # Important quantities that are learned for each time step separately
        children_factors = dict()           # Inferred children factors in the observation's atom space
        decoder = dict()                    # Decoder for mapping observation to latent state
        estimated_model = dict()            # Estimated model in the latent state space
        policy_cover = dict()               # Set of exploration policies to visit subset of state factors
        states_visited = dict()             # Set of learned abstract (or real) states visited at a given time

        tensorboard = Tensorboard(log_dir=self.config["save_path"])

        # Initialization for time step 0
        decoder[0] = ZeroDecoder(self.state_dim)
        states_visited[0] = {tuple([0] * self.state_dim)}   # Only state reachable at the zeroth time step is (0)
        policy_cover[0] = []                                # Empty policy

        for step in range(1, self.horizon + 1):

            # Step 1: Collect dataset
            self.logger.log("FactoRL: Time step %d out of %d" % (step, self.horizon))

            dataset = []
            self.logger.log("Phase 1: Data collection at time step %d." % step)
            for _ in range(2 * self.num_samples):
                dataset.append(
                    self._gather_sample(env, self.actions, step, policy_cover))
            self._log_dataset_stats(dataset, self.logger)

            # Step 2: Find the children factor
            self.logger.log("Phase 2: Learn Mapping from Atoms to Parent Factor at time step %d." % step)
            gold_factors = env.env.get_children_factors(step)
            graph_identifier = FactoRLGraphIdentification(self.config, self.constants)

            inferred_factors = gold_factors if use_gold_factor else \
                graph_identifier.get_factors(dataset[:self.num_samples], self.logger, tensorboard)

            factors_succ, infer_to_gold_map = graph_identifier.test_factors(factors_gold=gold_factors,
                                                                            factors_inferred=inferred_factors)
            children_factors[step] = inferred_factors
            if factors_succ:
                self.logger.log("Phase 2 succeeded. Learned children factors %r matching gold factors %r" %
                                (inferred_factors, gold_factors))
            else:
                self. logger.log("Phase 2 failed. Learned children factors %r while gold factors were %r " %
                                 (inferred_factors, gold_factors))

                if fail_early:
                    return {FactoRL.SUCC_STR: 0.0}

            # Step 3: Train decoder
            self.logger.log("Phase 3: Learn a Decoder for Each Children Factor Set at time step %d." % step)
            decoders = []
            for i, children_factor in enumerate(inferred_factors):

                self.logger.log("Learning Decoder Number %d, corresponding to factor set %r" % (i + 1, children_factor))

                # Prepare dataset for noise contrastive learning
                nce_dataset = self._create_nce_dataset(dataset, children_factor)

                # Decode each set of atoms
                factor_obs_dim = len(children_factor)
                decoder_, _ = self.train_model_util.train_model(nce_dataset, factor_obs_dim, self.logger,
                                                                True, self.debug, tensorboard)
                decoders.append(decoder_)

                # When we generalize then we should map i to the state-id
                min_diff = self.train_model_util.test_model(decoder_,
                                                            children_factor,
                                                            dataset,
                                                            infer_to_gold_map[i],
                                                            self.logger)

                if min_diff < FactoRL.MAX_DECODER_ERROR:
                    self.logger.log("Step 3 [Decoder number %d] succeeded with decoding error=%f < %f" %
                               (i, min_diff, FactoRL.MAX_DECODER_ERROR))
                else:
                    self.logger.log("Step 3 [Decoder number %d] failed with decoding error=%f > %f" %
                               (i, min_diff, FactoRL.MAX_DECODER_ERROR))

                    if fail_early:
                        return {FactoRL.SUCC_STR: 0.0}

            # decoders = [GoldDecoder() for _ in children_factors[step]]
            decoder[step] = CompoundDecoder(decoders, children_factors[step])

            # Step 4: Model the latent dynamics
            self.logger.log("Phase 4: Estimate the Latent Dynamics for time step %d." % step)
            estimated_model_, states_visited_, labeled_dataset = self._est_model(dataset=dataset[self.num_samples:],
                                                                                 curr_decoder=decoder[step - 1],
                                                                                 next_decoder=decoder[step],
                                                                                 logger=self.logger,
                                                                                 use_gold_state=False)

            # Compute model error
            avg_error = self._test_model_error(labeled_dataset, estimated_model_, self.logger)

            if avg_error <= FactoRL.MAX_MODEL_ERROR:
                self.logger.log("Step 4 [Model Estimation] succeeded. Avg. model error was %f <= %f" %
                                (avg_error, FactoRL.MAX_MODEL_ERROR))
            elif fail_early:
                self.logger.log("Step 4 [Model Estimation] failed. Avg. model error was %f > %f" %
                                (avg_error, FactoRL.MAX_MODEL_ERROR))

                return {FactoRL.SUCC_STR: 0.0}

            estimated_model[step - 1] = estimated_model_
            states_visited[step] = states_visited_

            # Step 5: Planning
            self.logger.log("Phase 5: Learning a policy cover for time step %d." % step)
            policy_cover[step] = self._learn_policy_cover(estimated_model, step, states_visited, decoder, self.logger)
            self._test_policy_cover(env, policy_cover, step + 1, self.logger)

        return {FactoRL.SUCC_STR: 1.0}


class ZeroDecoder:

    def __init__(self, state_dim):
        self.num_factors = state_dim

    def get_num_factor(self):
        return self.num_factors

    def encode_observations(self, obs):
        return tuple([0] * self.num_factors)
