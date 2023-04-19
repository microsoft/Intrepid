import os
import time
import random
import pickle
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from environments.rl_acid_env.rl_acid_wrapper import RLAcidWrapper


class VisualComboLock(RLAcidWrapper):
    """ Combination lock problem with Visual Features """

    env_name = "visualcombolock"

    IMAGE, FEATURE = range(2)

    def __init__(self, horizon, swap, num_actions, anti_shaping_reward, obs_dim, vary_instance=True):
        """
        :param horizon: Horizon of the MDP
        :param swap: Probability for stochastic edges
        :param vary_instance: Vary the instance of the object class when creating images
        """

        self.horizon = horizon
        self.swap = swap
        self.num_actions = num_actions
        self.optimal_reward = 1.0
        self.optimal_reward_prob = 1.0
        self.vary_instance = vary_instance

        assert anti_shaping_reward < self.optimal_reward * self.optimal_reward_prob, \
            "Anti shaping reward shouldn't exceed optimal reward which is %r" % \
            (self.optimal_reward * self.optimal_reward_prob)
        self.anti_shaping_reward = anti_shaping_reward

        assert num_actions >= 2, "Atleast two actions are needed"
        self.actions = list(range(0, num_actions))

        self.opt_a = np.random.choice(self.actions, size=self.horizon)
        self.opt_b = np.random.choice(self.actions, size=self.horizon)
        self.opt_c = np.random.choice(self.actions, size=self.horizon)
        self.blocked_states = self.create_random_setup(horizon)
        self.agent_pos = None

        if type(obs_dim) == list:

            # Image at a given time is a 4x4 grid with 20% of the top and bottom pixels
            # spent on background and 80% is spent on the grid.

            self.feature_type = VisualComboLock.IMAGE

            frame, image_height, image_width, channel = obs_dim
            assert frame == 1 and channel == 3, "Unsupported configuration"

            self.pixel_per_grid = (int(image_height/4.0), int(image_width/4.0))
            self.border_space = int((image_height - 3 * self.pixel_per_grid[0]) / 2)

            assert self.pixel_per_grid[0] > 0 and self.pixel_per_grid[1] > 0, \
                "Must have space for grids. Make image larger or reduce grid size."

            # Create the background
            self.background = Image.open("./data/visualcombolock/granite1.jpg")
            self.background = self.background.resize((image_height, image_width))
            self.current_background = self.background

            # Read objects
            self.object_images, self.agent_image, self.water_images = self._read__objects()

            # Select an object for each blocked state
            self.blocked_state_object_map = {}
            object_classes = list(self.object_images.keys())

            for i in range(0, self.horizon + 1):
                self.blocked_state_object_map[i] = random.choice(object_classes)

        elif type(obs_dim) == int:

            self.feature_type = VisualComboLock.FEATURE
            assert obs_dim >= 3 * self.horizon + 3
            self.obs_dim = obs_dim

        else:
            raise AssertionError("Unhandled type %r" % type(obs_dim))

    @staticmethod
    def create_random_setup(horizon):

        blocked_states = []
        for i in range(0, horizon + 1):
            blocked_state = random.randint(0, 2)        # There are three states at each point
            blocked_states.append(blocked_state)

        return blocked_states

    def _read__objects(self):

        # Object class contains a unique name and a set of object instances that are treated as equivalent.
        # Between transitions the object can change from one instance to another without affecting the state.
        object_classes = {
                        # "circle": ["circle1", "circle2"],
                        "lava": ["lava1", "lava2", "lava3", "lava4"],
                        # "star": ["star1", "star2"],
                        # "square": ["square1"],
                        # "cross": ["cross1", "cross2"],
                        # "triangle": ["triangle1", "triangle2"]
        }

        object_images = {}

        for object_class in object_classes:

            object_images[object_class] = []
            for name in object_classes[object_class]:

                img = Image.open("./data/visualcombolock/grid_objects/%s.png" % name)
                img = img.convert("RGBA")
                img = img.resize(self.pixel_per_grid)
                pixdata = img.load()

                for i in range(0, self.pixel_per_grid[0]):
                    for j in range(0, self.pixel_per_grid[1]):
                        if pixdata[i, j] == (255, 255, 255, 255):
                            pixdata[i, j] = (255, 255, 255, 0)

                object_images[object_class].append(img)

        agent_image = Image.open("./data/visualcombolock/grid_objects/agent.png")
        agent_image = agent_image.convert("RGBA")
        agent_image = agent_image.resize(self.pixel_per_grid)
        pixdata = agent_image.load()

        for i in range(0, self.pixel_per_grid[0]):
            for j in range(0, self.pixel_per_grid[1]):
                if pixdata[i, j] == (255, 255, 255, 255):
                    pixdata[i, j] = (255, 255, 255, 0)

        water_images = []
        for k in range(1, 3):
            water_image = Image.open("./data/visualcombolock/grid_objects/water%d.png" % k)
            water_image = water_image.convert("RGBA")
            water_image = water_image.resize((4 * self.pixel_per_grid[1], self.border_space))
            pixdata = water_image.load()

            for j in range(0, self.border_space):
                for i in range(0, 4 * self.pixel_per_grid[1]):
                    if pixdata[i, j] == (255, 255, 255, 255):
                        pixdata[i, j] = (255, 255, 255, 0)

            water_images.append(water_image)

        return object_images, agent_image, water_images

    def get_reachable_states(self):
        raise NotImplementedError()

    def get_num_states(self):
        return 3 * self.horizon + 3

    def reset(self):

        all_choices = [0, 1, 2]
        all_choices.remove(self.blocked_states[0])

        if np.random.binomial(1, 0.5) == 1:
            self.agent_pos = (all_choices[0], 0)
        else:
            self.agent_pos = (all_choices[1], 0)

        image = self.make_obs()

        return image, {"state": self.agent_pos}

    def make_obs(self):

        if self.feature_type == VisualComboLock.IMAGE:
            return self.make_image()
        elif self.feature_type == VisualComboLock.FEATURE:
            return self.make_feature()
        else:
            raise AssertionError("Unhandled feature type %r " % self.feature_type)

    def make_image(self):

        self.current_background = self.background.copy()

        water_id_1, water_id_2 = random.randint(0, 1), random.randint(0, 1)
        self.current_background.paste(self.water_images[water_id_1], (0, 0), self.water_images[water_id_1])
        self.current_background.paste(self.water_images[water_id_2],
                                      (0, self.border_space + 3 * self.pixel_per_grid[0]),
                                      self.water_images[water_id_2])

        for i in range(0, 3):   # Three Rows

            for j in range(0, 4):   # Four Columns

                time_step = self.agent_pos[1] + j

                if time_step <= self.horizon and self.blocked_states[time_step] == i:

                    # Show the blocked object at this place
                    offset = (j * self.pixel_per_grid[1], self.border_space + i * self.pixel_per_grid[0])

                    # Randomly pick an instance of selected class
                    object_class = self.blocked_state_object_map[time_step]
                    if self.vary_instance:
                        object_ix = random.randint(0, len(self.object_images[object_class]) - 1)
                    else:
                        object_ix = 0
                    chosen_object = self.object_images[object_class][object_ix]

                    self.current_background.paste(chosen_object, offset, chosen_object)

        # Paste the agent
        agent_offset = (0 * self.pixel_per_grid[1], self.border_space + self.agent_pos[0] * self.pixel_per_grid[0])
        self.current_background.paste(self.agent_image, agent_offset, self.agent_image)

        return np.asarray(self.current_background, dtype=np.float32)

    def make_feature(self):

        row, col = self.agent_pos
        ix = row + col * 3

        feat = np.zeros(self.obs_dim, dtype=np.float32)
        feat[ix] = 1.0  # Set the index
        feat += np.random.normal(loc=0.0, scale=0.1)

        return feat

    def get_optimal_value(self):
        return self.optimal_reward * self.optimal_reward_prob

    def step(self, act):

        old_pos = self.agent_pos
        self.agent_pos = self._dynamics(self.agent_pos, act)
        image = self.make_obs()

        reward = self.get_reward(old_pos, act, self.agent_pos)

        # Compute if the task is done
        if self.agent_pos[1] == self.horizon:
            done = True
        else:
            done = False

        return image, reward, done, {"state": self.agent_pos}

    def get_reward(self, old_pos, act, new_pos):

        # If the agent was blocked already then no reward
        if old_pos[0] == self.blocked_states[old_pos[1]]:
            return 0.0

        # If the agent reaches the final live states then give it the optimal reward.
        if (old_pos == (0, self.horizon - 1) and act == self.opt_a[self.horizon - 1]) or (
                old_pos == (1, self.horizon - 1) and act == self.opt_b[self.horizon - 1]) or (
                old_pos == (2, self.horizon - 1) and act == self.opt_c[self.horizon - 1]):
            return self.optimal_reward * np.random.binomial(1, self.optimal_reward_prob)

        # If reaching the dead state for the first time then give it a small anti-shaping reward.
        # This anti-shaping reward is anti-correlated with the optimal reward.
        if old_pos is not None and new_pos is not None:
            # Moving from a blocked state to non-block state gives an anti-shaped reward
            if old_pos[0] != self.blocked_states[old_pos[1]] and new_pos[0] == self.blocked_states[new_pos[1]]:
                return self.anti_shaping_reward * np.random.binomial(1, 0.5)

        return 0.0

    def _dynamics(self, old_pos, act):

        if old_pos[1] == self.horizon:  # Maximum actions achieved
            return None

        # Check if the agent is stuck; if the agent is stuck then it stays stuck
        if old_pos[0] == self.blocked_states[old_pos[1]]:
            return self.blocked_states[old_pos[1] + 1], old_pos[1] + 1

        b = np.random.binomial(1, self.swap)

        options = [0, 1, 2]
        options.remove(self.blocked_states[old_pos[1] + 1])

        if old_pos[0] == 0 and act == self.opt_a[old_pos[1]]:
            if b == 0:
                return options[0], old_pos[1] + 1
            else:
                return options[1], old_pos[1] + 1

        if old_pos[0] == 1 and act == self.opt_b[old_pos[1]]:
            if b == 0:
                return options[0], old_pos[1] + 1
            else:
                return options[1], old_pos[1] + 1

        if old_pos[0] == 2 and act == self.opt_c[old_pos[1]]:
            if b == 0:
                return options[0], old_pos[1] + 1
            else:
                return options[1], old_pos[1] + 1
        else:
            return self.blocked_states[old_pos[1] + 1], old_pos[1] + 1

    def render(self, wait_time=1):
        """ Renders the image """

        plt.ion()
        plt.imshow(self.current_background)
        plt.show()
        plt.pause(wait_time)

    def save(self, folder_name):
        """ Save the environment given the folder name """

        timestamp = time.time()

        if not os.path.exists(folder_name + "/env_%d" % timestamp):
            os.makedirs(folder_name + "/env_%d" % timestamp, exist_ok=True)

        with open(folder_name + "/env_%d/diabcombolock" % timestamp, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(env_folder_name):
        """ Load the environment from the environment folder name """

        with open(env_folder_name + "/diabcombolock", "rb") as f:
            env = pickle.load(f)

        return env


if __name__ == "__main__":

    horizon = 10
    env = VisualComboLock(horizon=horizon,
                          swap=0.5,
                          num_actions=10,
                          anti_shaping_reward=0.1,
                          obs_dim=[1, 484, 484, 3],
                          vary_instance=True)

    img, meta = env.reset()
    print("Start with Meta: %r" % meta)

    for step_ in range(horizon):
        agent_pos = meta["state"]

        if agent_pos[0] == 0:
            action = env.opt_a[agent_pos[1]]
        elif agent_pos[0] == 1:
            action = env.opt_b[agent_pos[1]]
        elif agent_pos[0] == 2:
            action = env.opt_c[agent_pos[1]]
        else:
            raise AssertionError("should be in {0, 1, 2}. Found %r" % agent_pos[0])
        img, reward, done, meta = env.step(action)
        print("Step %d, Action: %d, Reward: %r, Done: %r, Meta: %r" % (step_ + 1, action, reward, done, meta))
        # imageio.imwrite("./img%state_dim.png" % (step_ + 1), img)
        env.render()
