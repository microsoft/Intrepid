import numpy as np
from enum import IntEnum
from gym import spaces
from gym_minigrid.minigrid import Goal, Grid, Lava, MiniGridEnv, Wall


class GridWorld1(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    env_name = "gridworld1"

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        left_forward = 3
        right_forward = 4

    def __init__(self, config):
        self.obstacle_type = Lava
        self.wall_type = Wall
        self.last_done = False

        width = config["width"]
        height = config["height"]
        horizon = config["horizon"]
        config["env_seed"]
        agent_view_size = config["agent_view_size"]

        super().__init__(
            width=width,
            height=height,
            max_steps=horizon,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None,
            agent_view_size=agent_view_size,
        )

        self.min_dist_to_goal = 17

        # Actions are discrete integer values
        self.actions = GridWorld1.Actions
        self.action_space = spaces.Discrete(len(self.actions))

        self.reward_decay_ratio = 0.1  # config["reward_decay_ratio"]

    def _gen_grid(self, width, height):
        assert width == 10 and height == 10

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the mid-left
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Place a goal square in the mid-right
        self.goal_pos = np.array((width - 2, height // 2))
        self.put_obj(Goal(), *self.goal_pos)

        self.grid.vert_wall(2, 3, 1, self.obstacle_type)
        self.grid.vert_wall(2, 4, 1, self.wall_type)
        self.grid.vert_wall(2, 5, 1, self.wall_type)
        self.grid.vert_wall(2, 6, 1, self.wall_type)
        self.grid.vert_wall(2, 7, 1, self.obstacle_type)

        self.grid.vert_wall(4, 3, 1, self.obstacle_type)
        self.grid.vert_wall(4, 4, 1, self.wall_type)
        self.grid.vert_wall(4, 5, 1, self.wall_type)
        self.grid.vert_wall(4, 6, 1, self.wall_type)
        self.grid.vert_wall(4, 7, 1, self.obstacle_type)

        self.grid.vert_wall(5, 7, 1, self.obstacle_type)

        self.grid.vert_wall(6, 3, 1, self.obstacle_type)
        self.grid.vert_wall(6, 4, 1, self.wall_type)
        self.grid.vert_wall(6, 5, 1, self.wall_type)
        self.grid.vert_wall(6, 6, 1, self.wall_type)
        self.grid.vert_wall(6, 7, 1, self.obstacle_type)

        self.grid.vert_wall(7, 3, 1, self.obstacle_type)
        self.grid.vert_wall(8, 3, 1, self.wall_type)

        self.grid.vert_wall(6, 2, 1, self.obstacle_type)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def step(self, action):
        if self.last_done:
            # If done then the agent gets stuck
            obs = self.gen_obs()
            return obs, 0.0, True, {}

        self.step_count += 1

        reward = self._noop_reward()
        done = False

        # Rotate left
        if action == self.actions.left or action == self.actions.left_forward:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right or action == self.actions.right_forward:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move forward
        if action == self.actions.left or action == self.actions.right:
            pass
        elif (
            action == self.actions.forward
            or action == self.actions.left_forward
            or action == self.actions.right_forward
        ):
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
        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()
        self.last_done = done

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
