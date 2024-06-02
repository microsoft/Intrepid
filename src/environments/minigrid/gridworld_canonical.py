import numpy as np
import random
from enum import IntEnum
from gym import spaces
from gym_minigrid.minigrid import Goal, Grid, Lava, MiniGridEnv, Wall


class GridWorldCanonical(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    env_name = "gridworld-canonical"

    class Actions(IntEnum):
        # Move left, right, up and down
        left = 0
        right = 1
        up = 2
        down = 3

    def __init__(self, config):
        width = config["width"]
        height = config["height"]
        horizon = config["horizon"]
        seed = config["env_seed"]
        agent_view_size = config["agent_view_size"]
        self.det_start = config["det_start"]

        self.obstacle_type = Lava
        self.wall_type = Wall
        self.last_done = False

        super().__init__(
            width=width,
            height=height,
            max_steps=horizon,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=seed,
            agent_view_size=agent_view_size,
        )

        self.min_dist_to_goal = 8

        # Actions are discrete integer values
        self.actions = GridWorldCanonical.Actions
        self.action_space = spaces.Discrete(len(self.actions))

        self.reward_decay_ratio = 0.1  # config["reward_decay_ratio"]

    def _gen_grid(self, width, height):
        assert width == 7 and height == 7

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # # Place the agent in the center
        # self.agent_pos = (3, 3)
        # self.agent_dir = 0

        # Chose the below stochastically
        if self.det_start:
            self.agent_pos = (3, 3)
            self.agent_dir = 0
        else:
            self.agent_pos = (random.randint(1, 5), random.randint(1, 5))
            self.agent_dir = random.randint(0, 3)

        # Place a goal square in the mid-right
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def reset(self):
        self.last_done = False
        return super().reset()

    def step(self, action):
        if self.last_done:
            # If done then the agent gets stuck
            obs = None
            # obs = self.gen_obs()
            return obs, 0.0, True, {}

        self.step_count += 1

        reward = self._noop_reward()
        done = False

        if action == self.actions.left:
            self.agent_dir = 2
        elif action == self.actions.right:
            self.agent_dir = 0
        elif action == self.actions.up:
            self.agent_dir = 3
        elif action == self.actions.down:
            self.agent_dir = 1
        else:
            raise AssertionError("Unhandled action value.")

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move forward if possible

        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = fwd_pos

        if fwd_cell is not None and fwd_cell.type == "goal":
            done = True
            self.agent_pos = fwd_pos
            reward = self._goal_reward()

        if fwd_cell is not None and fwd_cell.type == "lava":
            done = True
            self.agent_pos = fwd_pos
            reward = self._lava_reward()

        if self.step_count >= self.max_steps:
            done = True

        obs = None
        # obs = self.gen_obs()

        return obs, reward, done, {}

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        raise NotImplementedError

    def _noop_reward(self):
        return -0.01

    def _lava_reward(self):
        return -1

    def _goal_reward(self):
        return 1
