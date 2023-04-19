from environments.rl_acid_env.noise_gen import *
from environments.rl_acid_env.rl_acid_wrapper import RLAcidWrapper


class DiabolicalCombinationLock(RLAcidWrapper):

    env_name = "diabcombolock"

    def __init__(self, config):
        """
        :param config: Configuration of the environment
        """

        RLAcidWrapper.__init__(self, config)

        self.noise_type = self.get_noise(config["noise_type"])
        self.adapt_config(config)

        self.horizon = config["horizon"]
        self.swap = config["swap_prob"]
        self.num_actions = config["num_actions"]
        self.optimal_reward = config["optimal_reward"]
        self.optimal_reward_prob = 1.0
        self.rng = np.random.RandomState(config["env_seed"])
        self.anti_shaping_reward = config["anti_shaping_reward"]
        self.anti_shaping_reward2 = config["anti_shaping_reward2"]

        assert self.anti_shaping_reward * 0.5 < self.optimal_reward * self.optimal_reward_prob, \
            "Anti shaping reward shouldn't exceed optimal reward which is %r" % \
            (self.optimal_reward * self.optimal_reward_prob)

        assert self.num_actions >= 2, "Atleast two actions are needed"
        self.actions = list(range(0, self.num_actions))
        self.spawn_prob = config["spawn_prob"]

        self.opt_a = self.rng.randint(low=0, high=self.num_actions, size=self.horizon)
        self.opt_b = self.rng.randint(low=0, high=self.num_actions, size=self.horizon)

        if self.noise_type == RLAcidWrapper.GAUSSIAN:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            self.dim = self.horizon + 4

        elif self.noise_type == RLAcidWrapper.BERNOULLI:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1. We further add noise of size horizon.
            self.dim = self.horizon + 4 + self.horizon  # Add noise of length horizon

        elif self.noise_type == RLAcidWrapper.HADAMHARD:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = self.horizon + 4
            self.hadamhard_matrix = generated_hadamhard_matrix(lower_bound)
            self.dim = self.hadamhard_matrix.shape[0]

        elif self.noise_type == RLAcidWrapper.HADAMHARDG:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = self.horizon + 4
            self.hadamhard_matrix = generated_hadamhard_matrix(lower_bound)
            self.dim = self.hadamhard_matrix.shape[0]

        else:
            raise AssertionError("Unhandled noise type %r" % self.noise_type)

    def get_env_name(self):
        return self.env_name

    def get_actions(self):
        return self.actions

    def get_num_actions(self):
        return self.num_actions

    def get_horizon(self):
        return self.horizon

    def transition(self, x, a):

        b = self.rng.binomial(1, self.swap)

        if x[0] == 0 and a == self.opt_a[x[1]]:
            if b == 0:
                return 0, x[1] + 1
            else:
                return 1, x[1] + 1
        if x[0] == 1 and a == self.opt_b[x[1]]:
            if b == 0:
                return 1, x[1] + 1
            else:
                return 0, x[1] + 1
        else:
            return 2, x[1] + 1

    def make_obs(self, x):

        if self.noise_type == RLAcidWrapper.BERNOULLI:

            v = np.zeros(self.dim, dtype=float)
            v[x[0]] = 1.0
            v[3 + x[1]] = 1.0
            v[self.horizon + 4:] = self.rng.binomial(1, 0.5, self.horizon)

        elif self.noise_type == RLAcidWrapper.GAUSSIAN:

            v = np.zeros(self.dim, dtype=float)
            v[x[0]] = 1.0
            v[3 + x[1]] = 1.0
            v = v + self.rng.normal(loc=0.0, scale=0.1, size=v.shape)

        elif self.noise_type == RLAcidWrapper.HADAMHARD:

            v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
            v[x[0]] = 1.0
            v[3 + x[1]] = 1.0
            v = np.matmul(self.hadamhard_matrix, v)

        elif self.noise_type == RLAcidWrapper.HADAMHARDG:

            v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
            v[x[0]] = 1.0
            v[3 + x[1]] = 1.0
            v = v + self.rng.normal(loc=0.0, scale=0.1, size=v.shape)
            v = np.matmul(self.hadamhard_matrix, v)

        else:
            raise AssertionError("Unhandled noise type %r" % self.noise_type)

        return v

    def start(self):
        # Start stochastically in one of the two live states
        toss_value = self.rng.binomial(1, self.spawn_prob)

        if toss_value == 0:
            return 0, 0
        elif toss_value == 1:
            return 1, 0
        else:
            raise AssertionError("Toss value can only be 1 or 0. Found %r" % toss_value)

    def reward(self, x, a, next_x):

        # If the agent reaches the final live states then give it the optimal reward.
        if (x == (0, self.horizon - 1) and a == self.opt_a[x[1]]) or \
                (x == (1, self.horizon - 1) and a == self.opt_b[x[1]]):
            return self.optimal_reward * self.rng.binomial(1, self.optimal_reward_prob)

        # If reaching the dead state for the first time then give it a small anti-shaping reward.
        # This anti-shaping reward is anti-correlated with the optimal reward.
        if x is not None and next_x is not None:
            if x[0] != 2 and next_x[0] == 2:
                return self.anti_shaping_reward * self.rng.binomial(1, 0.5)
            elif x[0] != 2 and next_x[0] != 2:
                return - self.anti_shaping_reward2 / (self.horizon - 1)

        return 0

    def get_optimal_value(self):
        # TODO: HARDCODING FOR MIKAEL's ENVIRONMENT. REMOVE IN FUTURE
        return 4.0
        # return self.optimal_reward * self.optimal_reward_prob

    def adapt_config(self, config):

        assert config["obs_dim"] == -1, "obs_dim key in config is automatically set. Please set it to -1"

        if self.noise_type == RLAcidWrapper.BERNOULLI:
            config["obs_dim"] = 2 * config["horizon"] + 4

        elif self.noise_type == RLAcidWrapper.GAUSSIAN:
            config["obs_dim"] = config["horizon"] + 4

        elif self.noise_type == RLAcidWrapper.HADAMHARD:
            config["obs_dim"] = get_sylvester_hadamhard_matrix_dim(config["horizon"] + 4)

        elif self.noise_type == RLAcidWrapper.HADAMHARDG:
            config["obs_dim"] = get_sylvester_hadamhard_matrix_dim(config["horizon"] + 4)

        else:
            raise AssertionError("Unhandled noise type %r" % config["noise_type"])

    def get_homing_policy_validation_fn(self, tolerance):

        return lambda dist, step: \
                str((0, step)) in dist and str((1, step)) in dist and str((2, step)) in dist and \
                dist[str((0, step))] + dist[str((1, step))] > 50 - tolerance and \
                dist[str((2, step))] > 50 - tolerance
