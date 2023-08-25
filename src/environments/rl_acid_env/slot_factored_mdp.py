import math
import random
import numpy as np


class SlotFactoredMDP:
    GAUSSIAN, GAUSSIAN2 = range(2)

    env_name = "slotfactoredmdp"

    def __init__(self, config):
        # Dimension of the world
        self.horizon = config["horizon"]
        self.num_factors = config["state_dim"]
        self.figures = [None] * self.num_factors
        self.actions = config["actions"]
        self.num_action = len(self.actions)
        self.num_atoms = 2 * self.num_factors

        self.fixed_actions = dict()
        self.atom_shuffle = dict()
        self.atom_factor_set = dict()

        self.noise_type = dict()

        for step in range(0, self.horizon + 1):
            # We allow the flexibility of different transition function at different time step

            if step < self.horizon:
                action_set = list(range(0, self.num_action))
                random.shuffle(action_set)
                self.fixed_actions[step] = action_set

            atom_set = list(range(0, self.num_atoms))
            random.shuffle(atom_set)
            self.atom_shuffle[step] = atom_set

            self.atom_factor_set[step] = []
            self.noise_type[step] = []

            for i in range(0, self.num_factors):
                # Each factor emits two atoms, e.g., factor i emits atoms 2i and 2i+1, and these are, however,
                # shuffled and put in different buckets
                self.atom_factor_set[step].append([atom_set[2 * i], atom_set[2 * i + 1]])

                # y = np.random.randint(0, 2)
                y = 0  # TODO enable other noise
                if y == 0:
                    self.noise_type[step].append(SlotFactoredMDP.GAUSSIAN)
                else:
                    self.noise_type[step].append(SlotFactoredMDP.GAUSSIAN2)

        assert self.num_factors == self.num_action, "Number of actions should be equal to number of factors"

        # Quantities below will change as agent takes action
        self.curr_state = None
        self.timestep = 0

    def reset(self):
        curr_state_ = [0] * self.num_factors
        self.curr_state = np.array(curr_state_)

        self.timestep = 0
        curr_obs = self.make_obs(self.curr_state)

        info = {"state": self.curr_state, "timestep": self.timestep}

        return curr_obs, info

    def get_reward(self, state, action):
        return 0.0

    def make_obs(self, state):
        """Each state currently gets two atoms"""

        data = []
        for ix in range(0, self.num_factors):
            # vec = np.zeros(2).astype(np.float32)
            # vec[state[ix]] = 1.0
            # vec += np.random.normal(loc=0.0, scale=0.1, size=(2,))      # Add noise

            if self.noise_type[self.timestep][ix] == SlotFactoredMDP.GAUSSIAN:
                if state[ix] == 0:
                    vec = np.array([1.0, 0.0]).astype(np.float32)
                elif state[ix] == 1:
                    vec = np.array([0.0, 1.0]).astype(np.float32)
                else:
                    raise AssertionError("State can only take value in {0, 1}. Found value %r" % state[ix])

                vec += np.random.normal(loc=0.0, scale=0.1, size=(1,))  # Add noise

            elif self.noise_type[self.timestep][ix] == SlotFactoredMDP.GAUSSIAN2:
                val = -1.0 if random.randint(0, 1) == 0 else 1.0
                if state[ix] == 0:
                    vec = np.array([0.5, val]).astype(np.float32)
                elif state[ix] == 1:
                    vec = np.array([val, 0.5]).astype(np.float32)
                else:
                    raise AssertionError("State can only take value in {0, 1}. Found value %r" % state[ix])

                vec += np.random.normal(loc=0.0, scale=0.1, size=(1,))  # Add noise

            else:
                raise AssertionError("Unhandled noise type %r" % self.noise_type[self.timestep][ix])

            data.append(vec)

        unshuffled_obs = np.concatenate(data)

        shuffled_obs = np.zeros((self.num_atoms,), dtype=np.float32)

        for i, k in enumerate(self.atom_shuffle[self.timestep]):
            shuffled_obs[k] = unshuffled_obs[i]

        return shuffled_obs

    def get_children_factors(self, step):
        return self.atom_factor_set[step]

    def step(self, action):
        """Given an action which is the acceleration output. Returns
        obs, reward, done, info
        """

        # self.visualize(self.curr_state, action)

        if self.timestep >= self.horizon:
            # Cannot take any more actions
            raise AssertionError("Cannot take any more actions this episode. Horizon completed")

        else:
            new_state = self.curr_state.copy()

            for i in range(self.num_factors):
                if action == self.fixed_actions[self.timestep][i]:
                    new_state[i] = 1.0 - new_state[i]

            reward = self.get_reward(self.curr_state, action)

            self.curr_state = new_state
            self.timestep += 1

            curr_obs = self.make_obs(new_state)

            done = self.timestep == self.horizon

            info = {"state": self.curr_state, "timestep": self.timestep}

            return curr_obs, reward, done, info


class SlotFactoredMDP1:
    def __init__(self, config):
        # Dimension of the world
        self.grid_x = config["grid_x"]
        self.grid_y = config["grid_y"]
        self.horizon = config["horizon"]
        self.num_states = math.factorial(self.grid_x) * self.grid_y

        self.figures = [None] * self.grid_x
        self.state = []

        # There is an action for every element of the grid.
        # action i * self.grid_y + j will swap element Grid[i, j] and Grid[i - 1, j]
        self.num_action = self.grid_x * self.grid_y

        self.curr_state = None
        self.timestep = 0

    def reset(self):
        curr_state_ = []

        for _ in range(self.grid_y):
            indices = list(range(0, self.grid_x))
            random.shuffle(indices)
            curr_state_.append(indices)

        self.curr_state = np.array(curr_state_).transpose()  # self.grid_x x self.grid_y

        self.timestep = 0
        curr_obs = self.make_obs(self.curr_state)

        info = {"state": self.curr_state, "timestep": self.timestep}

        return curr_obs, info

    def get_reward(self, state, action):
        return 0.0

    def make_obs(self, state):
        data = []
        for row in range(0, self.grid_x):
            for col in range(0, self.grid_y):
                vec = [0.0] * self.grid_x
                vec[state[row][col]] = 1.0

                data.extend(vec)

        return np.array(data).astype(np.float32)

    def step(self, action):
        """Given an action which is the acceleration output. Returns
        obs, reward, done, info
        """

        # self.visualize(self.curr_state, action)

        if self.timestep >= self.horizon:
            # Cannot take any more actions
            raise AssertionError("Cannot take any more actions this episode. Horizon completed")

        else:
            action_x, action_y = action // self.grid_y, action % self.grid_y
            new_state = self.curr_state.copy()
            prev_x = (action_x - 1) % self.grid_x
            new_state[prev_x, action_y] = self.curr_state[action_x, action_y]
            new_state[action_x, action_y] = self.curr_state[prev_x, action_y]

            reward = self.get_reward(self.curr_state, action)

            self.curr_state = new_state
            curr_obs = self.make_obs(new_state)

            self.timestep += 1

            done = self.timestep == self.horizon

            info = {"state": self.curr_state, "timestep": self.timestep}

            return curr_obs, reward, done, info
