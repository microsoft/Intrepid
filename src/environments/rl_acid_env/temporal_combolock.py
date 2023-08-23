import numpy as np
from environments.rl_acid_env.noise_gen import generated_hadamhard_matrix
from environments.rl_acid_env.rl_acid_wrapper import RLAcidWrapper


class TemporalCombinationLock(RLAcidWrapper):
    env_name = "temporal_combolock"

    def __init__(self, config):
        """
        :param config: Configuration of the environment
        """
        RLAcidWrapper.__init__(self, config)

        self.noise_type = self.get_noise(config["noise_type"])
        self.adapt_config(config)

        self.horizon = config["horizon"]
        self.num_actions = config["num_actions"]
        self.optimal_reward = config["optimal_reward"]
        self.optimal_reward_prob = 1.0
        self.rng = np.random.RandomState(10 * config["seed"])
        self.anti_shaping_reward = config["anti_shaping_reward"]
        self.anti_shaping_reward2 = config["anti_shaping_reward2"]
        self.tolerance = 0.1

        self.exo_flip_prob = config["exo_flip_prob"]
        self.exo_dim = (
            self.horizon if config["exo_dim"] == -1 else config["exo_dim"]
        )  # Dimension of exogenous noise

        assert (
            self.anti_shaping_reward < self.optimal_reward * self.optimal_reward_prob
        ), "Anti shaping reward shouldn't exceed optimal reward which is %r" % (
            self.optimal_reward * self.optimal_reward_prob
        )

        assert self.num_actions >= 2, "Atleast two actions are needed"
        self.actions = list(range(0, self.num_actions))

        self.opt = self.rng.randint(low=0, high=self.num_actions, size=self.horizon)

        if self.noise_type == RLAcidWrapper.GAUSSIAN:
            # We encode the state type and time separately. The type is one of the 2 and the time could be any value
            # in 1 to horizon + 1.
            self.dim = self.horizon + 3 + self.exo_dim

        elif self.noise_type == RLAcidWrapper.BERNOULLI:
            # We encode the state type and time separately. The type is one of the 2 and the time could be any value
            # in 1 to horizon + 1. We further add noise of size horizon.
            self.dim = (
                self.horizon + 3 + self.exo_dim + self.horizon
            )  # Add noise of length horizon

        elif self.noise_type == RLAcidWrapper.HADAMHARD:
            # We encode the state type and time separately. The type is one of the 2 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = self.horizon + 3 + self.exo_dim
            self.hadamhard_matrix = generated_hadamhard_matrix(lower_bound)
            self.dim = self.hadamhard_matrix.shape[0]

        elif self.noise_type == RLAcidWrapper.HADAMHARDG:
            # We encode the state type and time separately. The type is one of the 2 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = self.horizon + 3 + self.exo_dim
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

        if x[0] == 0 and a == self.opt[x[1]]:
            # If in "good" state and took right action then go to "good" state
            return 0, x[1] + 1, new_exo
        else:
            # Else go to "bad" state
            return 1, x[1] + 1, new_exo

    def calc_step(self, state, action):
        if state[0] == 0 and action == self.opt[state[1]]:
            # If in "good" state and took right action then go to "good" state
            return 0, state[1] + 1
        else:
            # Else go to "bad" state
            return 1, state[1] + 1

    def make_obs(self, x):
        if self.noise_type == RLAcidWrapper.BERNOULLI:
            v = np.zeros(self.dim, dtype=float)
            v[x[0]] = 1.0
            v[2 + x[1]] = 1.0
            v[self.horizon + 3 : self.horizon + 3 + self.exo_dim] = x[2]
            v[self.horizon + 3 + self.exo_dim :] = self.rng.binomial(
                1, 0.5, self.horizon
            )

        elif self.noise_type == RLAcidWrapper.GAUSSIAN:
            v = np.zeros(self.dim, dtype=float)
            v[x[0]] = 1.0
            v[2 + x[1]] = 1.0
            v[self.horizon + 3 :] = x[2]
            v = v + self.rng.normal(loc=0.0, scale=0.1, size=v.shape)

        elif self.noise_type == RLAcidWrapper.HADAMHARD:
            v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
            v[x[0]] = 1.0
            v[2 + x[1]] = 1.0
            v[self.horizon + 3 :] = x[2]
            v = np.matmul(self.hadamhard_matrix, v)

        elif self.noise_type == RLAcidWrapper.HADAMHARDG:
            v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
            v[x[0]] = 1.0
            v[2 + x[1]] = 1.0
            v[self.horizon + 3 :] = x[2]
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
        # If the agent reaches the final live states then give it the optimal reward.
        if x[0] == 0 and x[1] == self.horizon - 1 and a == self.opt[x[1]]:
            return self.optimal_reward * self.rng.binomial(1, self.optimal_reward_prob)

        # Give anti-shaped reward
        if x is not None and next_x is not None:
            if x[0] == 0 and next_x[0] == 1:
                return self.anti_shaping_reward * self.rng.binomial(1, 0.5)
            elif x[0] == 0 and next_x[0] == 0:
                return -self.anti_shaping_reward2 / (self.horizon - 1)

        return 0

    def get_optimal_value(self):
        return (
            self.optimal_reward * self.optimal_reward_prob - self.anti_shaping_reward2
        )

    @staticmethod
    def calc_obs_dim(config):
        noise_type = RLAcidWrapper.get_noise(config["noise_type"])
        horizon = config["horizon"]
        exo_dim = horizon if config["exo_dim"] == -1 else config["exo_dim"]

        if noise_type == RLAcidWrapper.GAUSSIAN:
            # We encode the state type and time separately. The type is one of the 2 and the time could be any value
            # in 1 to horizon + 1.
            return horizon + 3 + exo_dim

        elif noise_type == RLAcidWrapper.BERNOULLI:
            # We encode the state type and time separately. The type is one of the 2 and the time could be any value
            # in 1 to horizon + 1. We further add noise of size horizon.
            return horizon + 3 + exo_dim + horizon  # Add noise of length horizon

        elif noise_type == RLAcidWrapper.HADAMHARD:
            # We encode the state type and time separately. The type is one of the 2 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = horizon + 3 + exo_dim
            return generated_hadamhard_matrix(lower_bound).shape[0]

        elif noise_type == RLAcidWrapper.HADAMHARDG:
            # We encode the state type and time separately. The type is one of the 2 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = horizon + 3 + exo_dim
            return generated_hadamhard_matrix(lower_bound).shape[0]

        else:
            raise AssertionError("Unhandled noise type %r" % noise_type)

    def adapt_config(self, config):
        assert (
            config["obs_dim"] == -1
        ), "obs_dim key in config is automatically set. Please set it to -1"

        config["obs_dim"] = TemporalCombinationLock.calc_obs_dim(config)

    def get_endogenous_state(self, state):
        return state[0:2]

    @staticmethod
    def validate_config(config):
        calc_dim = TemporalCombinationLock.calc_obs_dim(config)
        assert config["obs_dim"] == calc_dim, (
            "Observation dimension in dictionary %d does not match what is expected %d"
            % (config["obs_dim"], calc_dim)
        )

    def generate_homing_policy_validation_fn(self):
        return (
            lambda dist, step: str((0, step)) in dist
            and str((1, step)) in dist
            and dist[str((0, step))] > 50 - self.tolerance
            and dist[str((1, step))] > 50 - self.tolerance
        )

    def act_to_str(self, action):
        return "%d" % action
