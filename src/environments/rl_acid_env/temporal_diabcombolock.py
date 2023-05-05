from environments.rl_acid_env.noise_gen import *
from environments.rl_acid_env.rl_acid_wrapper import RLAcidWrapper
from model.policy.stationary_constant_policy import StationaryConstantPolicy


class TemporalDiabCombinationLock(RLAcidWrapper):
    env_name = "temporal_diabcombolock"

    def __init__(self, config):
        """
        :param config: Configuration of the environment
        """
        RLAcidWrapper.__init__(self, config)

        self.noise_type = self.get_noise(config["noise_type"])
        self.adapt_config(config)

        self.horizon = config["horizon"]
        self.num_actions = config["num_actions"]
        self.optimal_reward_a = config["optimal_reward_a"]
        self.optimal_reward_b = config["optimal_reward_b"]
        self.optimal_reward_prob = 1.0
        self.rng = np.random.RandomState(10 * config["seed"])
        self.anti_shaping_reward = config["anti_shaping_reward"]
        self.anti_shaping_reward2 = config["anti_shaping_reward2"]
        self.tolerance = 0.1

        self.exo_flip_prob = config["exo_flip_prob"]
        self.exo_dim = (
            self.horizon if config["exo_dim"] == -1 else config["exo_dim"]
        )  # Dimension of exogenous noise

        minimum_good_state_return = (
            min(self.optimal_reward_a, self.optimal_reward_b) * self.optimal_reward_prob
        )

        assert self.anti_shaping_reward * 0.5 < minimum_good_state_return, (
            "Anti shaping reward shouldn't exceed return on reaching any good state %r"
            % minimum_good_state_return
        )

        assert self.num_actions >= 3, "At least three actions are needed"
        self.actions = list(range(0, self.num_actions))

        self.opt_a = self.rng.randint(low=0, high=self.num_actions, size=self.horizon)
        self.opt_b = []
        for h in range(self.horizon):
            actions = list(range(0, self.num_actions))
            actions.remove(self.opt_a[h])  # opt_a[h] and opt_b[h] are kept different
            self.opt_b.append(self.rng.choice(actions))

        if self.noise_type == RLAcidWrapper.GAUSSIAN:
            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            self.dim = self.horizon + 4 + self.exo_dim

        elif self.noise_type == RLAcidWrapper.BERNOULLI:
            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1. We further add noise of size horizon.
            self.dim = (
                self.horizon + 4 + self.exo_dim + self.horizon
            )  # Add noise of length horizon

        elif self.noise_type == RLAcidWrapper.HADAMHARD:
            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = self.horizon + 4 + self.exo_dim
            self.hadamhard_matrix = generated_hadamhard_matrix(lower_bound)
            self.dim = self.hadamhard_matrix.shape[0]

        elif self.noise_type == RLAcidWrapper.HADAMHARDG:
            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = self.horizon + 4 + self.exo_dim
            self.hadamhard_matrix = generated_hadamhard_matrix(lower_bound)
            self.dim = self.hadamhard_matrix.shape[0]

        else:
            raise AssertionError("Unhandled noise type %r" % self.noise_type)

        config["obs_dim"] = self.dim

    def get_env_name(self):
        return self.env_name

    def get_actions(self):
        return self.actions

    def get_num_actions(self):
        return self.num_actions

    def get_horizon(self):
        return self.horizon

    def transition(self, x, a):
        # New exogenous is created by bit-wise OR operator between new exo and a flip vector
        new_exo = x[2] ^ self.rng.binomial(1, self.exo_flip_prob, (self.exo_dim,))

        if x[1] == 0:  # First time step is a special case
            if a == self.opt_a[x[1]]:
                return 0, 1, new_exo

            elif a == self.opt_b[x[1]]:
                # We require opt_a[t] and opt_b[t] to be different, hence there is always a way to reach (1, 1, new_exo)
                return 1, 1, new_exo
            else:
                return 2, 1, new_exo
        else:
            if x[0] == 0 and a == self.opt_a[x[1]]:
                # If in "good" state of type 1 and took right action then go to "good" state of type 1
                return 0, x[1] + 1, new_exo
            elif x[0] == 1 and a == self.opt_b[x[1]]:
                # If in "good" state of type 2 and took right action then go to "good" state of type 2
                return 1, x[1] + 1, new_exo
            else:
                # Else go to "bad" state
                return 2, x[1] + 1, new_exo

    def calc_step(self, state, action):
        if state[1] == 0:  # First time step is a special case
            if action == self.opt_a[state[1]]:
                return 0, 1

            elif action == self.opt_b[state[1]]:
                # We require opt_a[t] and opt_b[t] to be different, hence there is always a way to reach (1, 1, new_exo)
                return 1, 1
            else:
                return 2, 1
        else:
            if state[0] == 0 and action == self.opt_a[state[1]]:
                # If in "good" state of type 1 and took right action then go to "good" state of type 1
                return 0, state[1] + 1
            elif state[0] == 1 and action == self.opt_b[state[1]]:
                # If in "good" state of type 2 and took right action then go to "good" state of type 2
                return 1, state[1] + 1
            else:
                # Else go to "bad" state
                return 2, state[1] + 1

    def make_obs(self, x):
        if self.noise_type == RLAcidWrapper.BERNOULLI:
            v = np.zeros(self.dim, dtype=float)
            v[x[0]] = 1.0
            v[3 + x[1]] = 1.0
            v[self.horizon + 4 : self.horizon + 4 + self.exo_dim] = x[2]
            v[self.horizon + 4 + self.exo_dim :] = self.rng.binomial(
                1, 0.5, self.horizon
            )

        elif self.noise_type == RLAcidWrapper.GAUSSIAN:
            v = np.zeros(self.dim, dtype=float)
            v[x[0]] = 1.0
            v[3 + x[1]] = 1.0
            v[self.horizon + 4 :] = x[2]
            v = v + self.rng.normal(loc=0.0, scale=0.1, size=v.shape)

        elif self.noise_type == RLAcidWrapper.HADAMHARD:
            v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
            v[x[0]] = 1.0
            v[3 + x[1]] = 1.0
            v[self.horizon + 4 :] = x[2]
            v = np.matmul(self.hadamhard_matrix, v)

        elif self.noise_type == RLAcidWrapper.HADAMHARDG:
            v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
            v[x[0]] = 1.0
            v[3 + x[1]] = 1.0
            v[self.horizon + 4 :] = x[2]
            v = v + self.rng.normal(loc=0.0, scale=0.1, size=v.shape)
            v = np.matmul(self.hadamhard_matrix, v)

        else:
            raise AssertionError("Unhandled noise type %r" % self.noise_type)

        return v

    def start(self):
        # Endogenous state is (0, 0) and we sample exogenous state has Bernoulli iid vector
        exogenous = self.rng.binomial(1, 0.5, (self.exo_dim,))

        return 0, 0, exogenous

    def reward(self, x, a, next_x):
        # If the agent reaches the final good states then give it the optimal reward.
        if x[0] == 0 and x[1] == self.horizon - 1 and a == self.opt_a[x[1]]:
            return self.optimal_reward_a * self.rng.binomial(
                1, self.optimal_reward_prob
            )

        if x[0] == 1 and x[1] == self.horizon - 1 and a == self.opt_b[x[1]]:
            return self.optimal_reward_b * self.rng.binomial(
                1, self.optimal_reward_prob
            )

        # Give anti-shaped reward
        if x is not None and next_x is not None:
            if (x[0] == 0 or x[0] == 1) and next_x[0] == 2:
                # positive reward for failing
                return self.anti_shaping_reward * self.rng.binomial(1, 0.5)
            elif next_x[0] == 0 or next_x[0] == 1:
                # negative reward for suceeding
                return -self.anti_shaping_reward2 / (self.horizon - 1)

        return 0

    def get_optimal_value(self):
        return (
            max(self.optimal_reward_a, self.optimal_reward_b) * self.optimal_reward_prob
            - self.anti_shaping_reward2
        )

    @staticmethod
    def calc_obs_dim(config):
        noise_type = RLAcidWrapper.get_noise(config["noise_type"])
        horizon = config["horizon"]
        exo_dim = horizon if config["exo_dim"] == -1 else config["exo_dim"]

        if noise_type == RLAcidWrapper.GAUSSIAN:
            # We encode the state type and time separately. The type is one of the 2 and the time could be any value
            # in 1 to horizon + 1.
            return horizon + 4 + exo_dim

        elif noise_type == RLAcidWrapper.BERNOULLI:
            # We encode the state type and time separately. The type is one of the 2 and the time could be any value
            # in 1 to horizon + 1. We further add noise of size horizon.
            return horizon + 4 + exo_dim + horizon  # Add noise of length horizon

        elif noise_type == RLAcidWrapper.HADAMHARD:
            # We encode the state type and time separately. The type is one of the 2 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = horizon + 4 + exo_dim
            return generated_hadamhard_matrix(lower_bound).shape[0]

        elif noise_type == RLAcidWrapper.HADAMHARDG:
            # We encode the state type and time separately. The type is one of the 2 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = horizon + 4 + exo_dim
            return generated_hadamhard_matrix(lower_bound).shape[0]

        else:
            raise AssertionError("Unhandled noise type %r" % noise_type)

    def adapt_config(self, config):
        assert (
            config["obs_dim"] == -1
        ), "obs_dim key in config is automatically set. Please set it to -1"

        config["obs_dim"] = TemporalDiabCombinationLock.calc_obs_dim(config)

    def get_endogenous_state(self, state):
        return state[0:2]

    @staticmethod
    def validate_config(config):
        calc_dim = TemporalDiabCombinationLock.calc_obs_dim(config)
        assert config["obs_dim"] == calc_dim, (
            "Observation dimension in dictionary %d does not match what is expected %d"
            % (config["obs_dim"], calc_dim)
        )

    def generate_homing_policy_validation_fn(self):
        return (
            lambda dist, step: str((0, step)) in dist
            and str((1, step)) in dist
            and str((2, step)) in dist
            and dist[str((0, step))] > 33 - self.tolerance
            and dist[str((1, step))] > 33 - self.tolerance
            and dist[str((2, step))] > 33 - self.tolerance
        )

    def act_to_str(self, action):
        return "%d" % action

    def get_perfect_homing_policy(self, step):
        opt_c = []
        for i in range(0, step):
            actions_copy = list(self.actions)
            actions_copy.remove(self.opt_a[i])
            actions_copy.remove(self.opt_b[i])
            opt_c.append(actions_copy[0])

        policy_a = {
            i: StationaryConstantPolicy(self.opt_a[i - 1]) for i in range(1, step + 1)
        }
        policy_b = {
            i: StationaryConstantPolicy(self.opt_b[i - 1]) for i in range(1, step + 1)
        }
        policy_c = {
            i: StationaryConstantPolicy(opt_c[i - 1]) for i in range(1, step + 1)
        }

        return [policy_a, policy_b, policy_c]
