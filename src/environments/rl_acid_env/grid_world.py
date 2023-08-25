import random
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from environments.rl_acid_env.rl_acid_wrapper import RLAcidWrapper


class Setup:
    def __init__(self, grid_size, horizon, start_pos, blocked_states, goal_pos):
        self.grid_size = grid_size
        self.horizon = horizon
        self.start_pos = start_pos
        self.blocked_states = blocked_states
        self.goal_pos = goal_pos

    def get_grid_size(self):
        return self.grid_size

    def get_horizon(self):
        return self.horizon

    def get_start_pos(self):
        return self.start_pos

    def get_blocked_states(self):
        return self.blocked_states

    def get_goal_pos(self):
        return self.goal_pos


class GridWorld(RLAcidWrapper):
    """Find the hidden key in a grid world"""

    IMAGE, FEATURE = range(2)

    def __init__(self, num_grid_row, num_grid_col, horizon, obs_dim, vary_instance=True):
        """
        :param num_grid_row: Number of rows in the grid
        :param num_grid_col: Number of columns in the grid
        :param horizon: Horizon of the problem
        :param obs_dim: The dimension of the image (list of size 4) or feature (one natural number)
        :param vary_instance: Vary the instance of the object class when creating images
        """

        self.num_grid_row = num_grid_row
        self.num_grid_col = num_grid_col
        self.setup = self._get_setup_4_by_4(horizon)
        self.agent_pos = self.setup.get_start_pos()
        self.blocked_states = self.setup.get_blocked_states()
        self.time_step = 0
        self.reachable_states = None
        self.vary_instance = vary_instance

        if isinstance(obs_dim, list):
            self.feature_type = GridWorld.IMAGE

            frame, image_height, image_width, channel = obs_dim
            assert frame == 1 and channel == 3, "Unsupported configuration"

            self.pixel_per_grid = (
                int(image_height / num_grid_row),
                int(image_width / num_grid_col),
            )

            assert (
                self.pixel_per_grid[0] > 0 and self.pixel_per_grid[1] > 0
            ), "Must have space for grids. Make image larger or reduce grid size."

            # Create a green background
            image_r = np.zeros((image_height, image_width, 1), dtype=np.float32)
            image_g = np.ones((image_height, image_width, 1), dtype=np.float32)
            image_b = np.zeros((image_height, image_width, 1), dtype=np.float32)
            self.background = np.concatenate([image_r, image_g, image_b], axis=2)
            self.background = Image.fromarray(self.background.astype(np.uint8))
            self.current_background = self.background

            # Read objects
            self.object_images, self.agent_image = self._read__objects()

            # Select an object for each blocked state
            self.blocked_state_object_map = {}
            object_classes = list(self.object_images.keys())

            for i in range(0, self.num_grid_row):
                for j in range(0, self.num_grid_col):
                    if self.blocked_states[i][j]:
                        self.blocked_state_object_map[(i, j)] = random.choice(object_classes)

        elif isinstance(obs_dim, int):
            self.feature_type = GridWorld.FEATURE
            self.noise_feature = obs_dim - num_grid_row * num_grid_col
            self.obs_dim = obs_dim

        else:
            raise AssertionError("Unhandled type %r" % type(obs_dim))

    def _read__objects(self):
        # Object class contains a unique name and a set of object instances that are treated as equivalent.
        # Between transitions the object can change from one instance to another without affecting the state.
        object_classes = {
            "circle": ["circle1", "circle2"],
            "lava": ["lava1", "lava2", "lava3", "lava4"],
            "star": ["star1", "star2"],
            "square": ["square1"],
            "cross": ["cross1", "cross2"],
            "triangle": ["triangle1", "triangle2"],
        }

        object_images = {}

        for object_class in object_classes:
            object_images[object_class] = []
            for name in object_classes[object_class]:
                img = Image.open("./data/gridworld/grid_objects/%s.png" % name)
                img = img.convert("RGBA")
                img = img.resize(self.pixel_per_grid)
                pixdata = img.load()

                for i in range(0, self.pixel_per_grid[0]):
                    for j in range(0, self.pixel_per_grid[1]):
                        if pixdata[i, j] == (255, 255, 255, 255):
                            pixdata[i, j] = (255, 255, 255, 0)

                object_images[object_class].append(img)

        agent_image = Image.open("./data/gridworld/grid_objects/agent.png")
        agent_image = agent_image.convert("RGBA")
        agent_image = agent_image.resize(self.pixel_per_grid)
        pixdata = agent_image.load()

        for i in range(0, self.pixel_per_grid[0]):
            for j in range(0, self.pixel_per_grid[1]):
                if pixdata[i, j] == (255, 255, 255, 255):
                    pixdata[i, j] = (255, 255, 255, 0)

        return object_images, agent_image

    @staticmethod
    def _get_setup_4_by_4(horizon):
        blocked_states = np.zeros((4, 4), dtype=np.bool)

        blocked_states[1][1] = True
        blocked_states[1][2] = True
        blocked_states[2][1] = True
        blocked_states[3][1] = True

        goal_pos = (3, 2)
        # reward_func = lambda pos: 1.0 if pos == (3, 2) else 0.0

        return Setup(
            grid_size=(4, 4),
            horizon=horizon,
            start_pos=(0, 0),
            blocked_states=blocked_states,
            goal_pos=goal_pos,
        )

    def get_reachable_states(self):
        if self.reachable_states is None:
            reachable_states = dict()
            reachable_states[0] = set()
            reachable_states[0].add((0, 0))
            horizon = self.setup.get_horizon()

            for time in range(1, horizon + 1):
                prev_states = reachable_states[time - 1]
                reachable_states[time] = set()
                for state in prev_states:
                    for action in [0, 1, 2, 3]:
                        new_state = self._dynamics(state, action)
                        reachable_states[time].add(tuple(new_state))

            self.reachable_states = reachable_states

        return self.reachable_states

    def get_num_states(self):
        return self.num_grid_row * self.num_grid_col

    def reset(self):
        self.agent_pos = self.setup.get_start_pos()
        self.time_step = 0
        image = self.make_obs()

        return image, {"state": self.agent_pos}

    def make_obs(self):
        if self.feature_type == GridWorld.IMAGE:
            return self.make_image()
        elif self.feature_type == GridWorld.FEATURE:
            return self.make_feature()
        else:
            raise AssertionError("Unhandled feature type %r " % self.feature_type)

    def make_image(self):
        self.current_background = self.background.copy()

        for i in range(0, self.num_grid_row):
            for j in range(0, self.num_grid_col):
                if self.blocked_states[i][j]:
                    # Define the offset:  Note the second value represents the space from top of the image
                    # and first value represents the space from the left of the image.
                    offset = (j * self.pixel_per_grid[1], i * self.pixel_per_grid[0])

                    # Randomly pick an instance of selected class
                    object_class = self.blocked_state_object_map[(i, j)]
                    if self.vary_instance:
                        object_ix = random.randint(0, len(self.object_images[object_class]) - 1)
                    else:
                        object_ix = 0
                    chosen_object = self.object_images[object_class][object_ix]

                    self.current_background.paste(chosen_object, offset, chosen_object)

        # Paste the agent
        agent_offset = (
            self.agent_pos[1] * self.pixel_per_grid[1],
            self.agent_pos[0] * self.pixel_per_grid[0],
        )
        self.current_background.paste(self.agent_image, agent_offset, self.agent_image)

        self.current_background = np.asarray(self.current_background, dtype=np.float32)

        return self.current_background

    def make_feature(self):
        row, col = self.agent_pos
        ix = row * self.num_grid_col + col
        num_squares = self.num_grid_row * self.num_grid_col

        feat = np.zeros(self.obs_dim, dtype=np.float32)
        feat[ix] = 1.0  # Set the index
        feat[num_squares:] = np.random.binomial(1, 0.5, self.obs_dim - num_squares)

        return feat

    def step(self, act):
        self.agent_pos = self._dynamics(self.agent_pos, act)
        image = self.make_obs()

        self.time_step = self.time_step + 1

        # Compute the reward
        goal_pos = self.setup.get_goal_pos()
        # reward = reward_func(self.agent_pos)
        reward = (self.agent_pos[0] == goal_pos[0]) and (self.agent_pos[1] == goal_pos[1])

        # Compute if the task is done
        if self.time_step >= self.setup.get_horizon():
            done = True
        else:
            done = False

        return image, reward, done, {"state": self.agent_pos}

    def _dynamics(self, old_pos, act):
        grid_rows, grid_cols = self.setup.get_grid_size()
        blocked_states = self.setup.get_blocked_states()
        row, col = old_pos

        if act == 0:  # Top
            new_row, new_col = max(row - 1, 0), col
        elif act == 1:  # Right
            new_row, new_col = row, min(col + 1, grid_cols - 1)
        elif act == 2:  # Bottom
            new_row, new_col = min(row + 1, grid_rows - 1), col
        elif act == 3:  # Left
            new_row, new_col = row, max(col - 1, 0)
        else:
            raise AssertionError("Action must be in {0, 1, 2, 3}")

        if blocked_states[new_row][new_col]:  # Cannot move if blocked
            new_row, new_col = row, col

        return new_row, new_col

    def render(self):
        """Renders the image"""

        plt.imshow(self.current_background, interpolation="nearest")
        plt.show()


if __name__ == "__main__":
    import imageio

    env = GridWorld(num_grid_row=4, num_grid_col=4, horizon=7, obs_dim=[1, 84, 84, 3])

    actions = [1, 1, 1, 2, 2, 3, 2]

    img, meta = env.reset()
    print(meta)
    imageio.imwrite("./img0.png", img)

    for step_, action in enumerate(actions):
        img, reward, done, meta = env.step(action)
        print("Step %d, Action: %d, Reward: %r, Done: %r, Meta: %r" % (step_ + 1, action, reward, done, meta))
        imageio.imwrite("./img%d.png" % (step_ + 1), img)

    env.render()
