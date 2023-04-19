import pdb
from environments.rl_acid_env.noise_gen import *
from environments.rl_acid_env.rl_acid_wrapper import RLAcidWrapper


class SafetyWorld(RLAcidWrapper):

    env_name = "safetyworld"

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
        self.rng = np.random.RandomState(config["env_seed"])
        self.anti_shaping_reward = config["anti_shaping_reward"]
        self.anti_shaping_reward2 = config["anti_shaping_reward2"]

        assert self.anti_shaping_reward < self.optimal_reward * self.optimal_reward_prob, \
            "Anti shaping reward shouldn't exceed optimal reward which is %r" % \
            (self.optimal_reward * self.optimal_reward_prob)

        assert self.num_actions >= 4, "Atleast 4 actions are needed"
        self.actions = list(range(0, self.num_actions))
        self.stop_action = config["stop_action"]

        self.safety_block_size = 1  # 5
        self.safety_dim = self.horizon * self.safety_block_size

        # Safety Metrics
        self.num_unsafe_actions = 0
        self.num_oracle_calls = 0

        # Log the episode ID at which each unsafe action metric was received.
        # Can be at most number of episodes x actions x horizon
        self.unsafe_actions_metric = []

        # Log the episode ID at which each unsafe action metric was received.
        # Can be at most number of episodes x actions x horizon
        self.num_oracle_calls_metric = []

        self.gold_w = np.random.randint(low=0, high=1, size=(self.safety_dim,)).astype(np.float32) - 0.5
        self.gold_w = self.gold_w + np.random.randn(1,)[0] * 0.025
        self.gold_w /= (np.linalg.norm(self.gold_w) + 1e-6)
        self.gold_b = 0.5

        self.gold_w_norm = np.linalg.norm(self.gold_w)
        self.gold_w_norm_sq = self.gold_w_norm * self.gold_w_norm

        self.constant_safe_feature, self.constant_unsafe_feature = self.generate_safe_unsafe_ftrs()

        self.switch_actions = []
        self.unsafe_actions = []

        for i in range(self.horizon):
            actions = list(range(0, self.num_actions))

            actions.remove(self.stop_action)
            switch_action = self.rng.choice(actions)
            self.switch_actions.append(switch_action)

            actions.remove(switch_action)
            unsafe_action = self.rng.choice(actions)
            self.unsafe_actions.append(unsafe_action)

        self.switch_actions = np.array(self.switch_actions).astype(np.int32)
        self.unsafe_actions = np.array(self.unsafe_actions).astype(np.int32)

        # Check dimensions below
        if self.noise_type == RLAcidWrapper.GAUSSIAN:

            # We encode the state type and time separately. The type is one of the 4 and the time could be any value
            # in 0 to horizon .
            self.dim = self.horizon + 5

        elif self.noise_type == RLAcidWrapper.BERNOULLI:

            # We encode the state type and time separately. The type is one of the 4 and the time could be any value
            # in 0 to horizon. We further add noise of size horizon.
            self.dim = self.horizon + 5 + self.horizon  # Add noise of length horizon

        elif self.noise_type == RLAcidWrapper.HADAMHARD:

            # We encode the state type and time separately. The type is one of the 4 and the time could be any value
            # in 0 to horizon.
            lower_bound = self.horizon + 5
            self.hadamhard_matrix = generated_hadamhard_matrix(lower_bound)
            self.dim = self.hadamhard_matrix.shape[0]

        elif self.noise_type == RLAcidWrapper.HADAMHARDG:

            # We encode the state type and time separately. The type is one of the 4 and the time could be any value
            # in 0 to horizon.
            lower_bound = self.horizon + 5
            self.hadamhard_matrix = generated_hadamhard_matrix(lower_bound)
            self.dim = self.hadamhard_matrix.shape[0]

        else:
            raise AssertionError("Unhandled noise type %r" % self.noise_type)

    def generate_safe_unsafe_ftrs(self, block_ix=-1):

        vec = np.random.randint(low=-1, high=2, size=(self.safety_dim,)).astype(np.float32)
        vec += np.random.randn(self.safety_dim,) * 0.01
        # vec /= (np.linalg.norm(vec) + 1e-6)

        if block_ix >= 0:
            vec[:(block_ix * self.safety_block_size)] = 0.0
            vec[((block_ix + 1) * self.safety_block_size):] = 0.0

        val = np.dot(self.gold_w, vec) + self.gold_b

        if val >= 0:
            safe_ftr = vec

            # If <w, v> + b >= 0 then <w,  - v - 2b w> + b = - <w, v> - 2b + b = - <w, v> - b <= 0
            unsafe_ftr = - vec - 2 * self.gold_b * self.gold_w / self.gold_w_norm_sq
        else:
            safe_ftr = - vec - 2 * self.gold_b * self.gold_w / self.gold_w_norm_sq

            # If <w, v> + b >= 0 then <w,  - v - 2b w> + b = - <w, v> - 2b + b = - <w, v> - b <= 0
            unsafe_ftr = vec

        pos_val = np.dot(self.gold_w, safe_ftr) + self.gold_b
        neg_val = np.dot(self.gold_w, unsafe_ftr) + self.gold_b
        assert pos_val >= 0, "Found val %f which was supposed to be positive" % pos_val
        assert neg_val <= 0, "Found val %f which was supposed to be negative" % neg_val

        return safe_ftr, unsafe_ftr

    def get_env_name(self):
        return self.env_name

    def get_actions(self):
        return self.actions

    def get_num_actions(self):
        return self.num_actions

    def get_horizon(self):
        return self.horizon

    def transition(self, x, a):

        # The world transitions deterministically as follows: There are 4 actions and 4 paths
        #
        # 1. There is a bad path (index 0) that gives a reward of -1/H at each step. The agent enters this path on
        #    taking an unsafe action in any state. Along this path, every action is unsafe except the stop action and
        #    safety features don't vary and so this path gives little information for learning safety.
        #
        # 2. There is a bad path (index 1) that yields negative reward of -1 at each time step but along which safety
        #    features are revealed quickly, i.e., after step 4 of this path, the safety features vary in all directions.
        #
        # 3. There is a good path (index 2) that yields positive reward of 1/H at each time step
        #    but along which safety features are revealed slowly, i.e., at step d of this path,
        #    safety features only vary along dimension d. This path is safe and also the optimal path.
        #    Agent starts on this path.
        #
        # 4. There is an absorbing (index 3) safe path that gives 0 reward and where the agent transitions after
        #    taking the stop action. All actions in this path are safe and safety features dont vary.
        #
        #  Every state on path index 1 and index 2 has one unsafe action which leads to path 0, a safe action that leads
        #  to path 1, a safe action that leads to path 2, and stop action that leads to path 4.

        ix, h = x

        if ix != 3 and a == self.unsafe_actions[h]:
            self.num_unsafe_actions += 1
            self.unsafe_actions_metric.append(self.num_eps)

        if ix == 0:
            if a == self.stop_action:           # stop/safe action
                return 3, h + 1
            else:
                return 0, h + 1                 # continue

        elif ix == 1:
            if a == self.unsafe_actions[h]:     # continue action
                return 0, h + 1
            elif a == self.switch_actions[h]:   # switch action
                return 2, h + 1
            elif a == self.stop_action:         # stop/safe action
                return 3, h + 1
            else:                               # continue
                return 1, h + 1

        elif ix == 2:

            if a == self.unsafe_actions[h]:     # unsafe action
                return 0, h + 1
            elif a == self.switch_actions[h]:   # switch action
                return 1, h + 1
            elif a == self.stop_action:         # stop/safe action
                return 3, h + 1
            else:
                return 2, h + 1                 # continue

        elif ix == 3:
            return 3, h + 1                     # continue

        else:
            raise AssertionError("Path index can only take values in {0, 1, 2, 3}")

    def get_safety_ftrs(self, x):

        return np.vstack([self.get_safety_ftr(x, a) for a in range(self.num_actions)])      # num_action x dim

    def get_safety_ftr(self, x, a):

        ix, h = x

        if ix == 0:     # Safety features are constant
            if a == self.stop_action:
                return self.constant_safe_feature
            else:
                return self.constant_unsafe_feature

        elif ix == 1:   # Fast revelation of safety features

            safe_ftr, unsafe_ftr = self.generate_safe_unsafe_ftrs()

            if a == self.unsafe_actions[h]:
                return unsafe_ftr
            else:
                return safe_ftr

        elif ix == 2:  # Slow revelation of safety features

            safe_ftr, unsafe_ftr = self.generate_safe_unsafe_ftrs(h)

            if a == self.unsafe_actions[h]:
                return unsafe_ftr
            else:
                return safe_ftr

        elif ix == 3:   # Safety features are constant. Everything here is safe.
            return self.constant_safe_feature

        else:
            raise AssertionError("Path index can only take values in {0, 1, 2, 3}")

    def safety_query(self, safety_ftr, save=True):

        if save:
            self.num_oracle_calls += 1
            self.num_oracle_calls_metric.append(self.num_eps)

        val = np.dot(self.gold_w, safety_ftr) + self.gold_b

        if val >= 0:
            return True
        else:
            return False

    def make_obs(self, x):

        if self.noise_type == RLAcidWrapper.BERNOULLI:

            v = np.zeros(self.dim, dtype=float)
            v[x[0]] = 1.0
            v[4 + x[1]] = 1.0
            v[self.horizon + 5:] = self.rng.binomial(1, 0.5, self.horizon)

        elif self.noise_type == RLAcidWrapper.GAUSSIAN:

            v = np.zeros(self.dim, dtype=float)
            v[x[0]] = 1.0
            v[4 + x[1]] = 1.0
            v = v + self.rng.normal(loc=0.0, scale=0.1, size=v.shape)

        elif self.noise_type == RLAcidWrapper.HADAMHARD:

            v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
            v[x[0]] = 1.0
            v[4 + x[1]] = 1.0
            v = np.matmul(self.hadamhard_matrix, v)

        elif self.noise_type == RLAcidWrapper.HADAMHARDG:

            v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
            v[x[0]] = 1.0
            v[4 + x[1]] = 1.0
            v = v + self.rng.normal(loc=0.0, scale=0.1, size=v.shape)
            v = np.matmul(self.hadamhard_matrix, v)

        else:
            raise AssertionError("Unhandled noise type %r" % self.noise_type)

        return v

    def start(self):
        # Starts in path with index 2 where safety features are revealed slowly
        return 2, 0

    def reward(self, x, a, next_x):

        ix, _ = x
        next_ix, h = next_x

        if next_ix == 2 and h == self.horizon:
            return 2.0

        if next_ix == 0:
            return -1.0

        elif next_ix == 1:
            return -1.0

        elif next_ix == 2:
            return 1.0 / float(self.horizon)

        elif next_ix == 3:
            return 0.0

        raise AssertionError("Shouldn't reach here")

    def get_optimal_value(self):
        # return 1.0
        raise NotImplementedError()

    def adapt_config(self, config):

        assert config["obs_dim"] == -1, "obs_dim key in config is automatically set. Please set it to -1"

        if self.noise_type == RLAcidWrapper.BERNOULLI:
            config["obs_dim"] = 2 * config["horizon"] + 5

        elif self.noise_type == RLAcidWrapper.GAUSSIAN:
            config["obs_dim"] = config["horizon"] + 5

        elif self.noise_type == RLAcidWrapper.HADAMHARD:
            config["obs_dim"] = get_sylvester_hadamhard_matrix_dim(config["horizon"] + 5)

        elif self.noise_type == RLAcidWrapper.HADAMHARDG:
            config["obs_dim"] = get_sylvester_hadamhard_matrix_dim(config["horizon"] + 5)

        else:
            raise AssertionError("Unhandled noise type %r" % config["noise_type"])

    def get_homing_policy_validation_fn(self, tolerance):
        raise NotImplementedError()

        # return lambda dist, step: \
        #         str((0, step)) in dist and str((1, step)) in dist and \
        #         dist[str((0, step))] > 50 - tolerance and dist[str((1, step))] > 50 - tolerance
