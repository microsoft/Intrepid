import random
from gym_minigrid.minigrid import *


class GridWorldRandomized(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    env_name = "gridworld-randomized"

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        # TODO left_forward and right_forward dont work properly in this code.
        left_forward = 3
        right_forward = 4

    def __init__(self, config):

        width = config["width"]
        height = config["height"]
        horizon = config["horizon"]
        seed = config["env_seed"]
        agent_view_size = config["agent_view_size"]

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
            agent_view_size=agent_view_size
        )

        self.min_dist_to_goal = 8

        # Actions are discrete integer values
        self.actions = GridWorldRandomized.Actions
        self.action_space = spaces.Discrete(len(self.actions))

        self.reward_decay_ratio = 0.1  # config["reward_decay_ratio"]

    def _gen_grid(self, width, height):

        assert width == 15 and height == 15

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.wall_squares = {
            (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11),
            (2, 3),
            (4, 5), (5, 5),
            (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8), (10, 8), (11, 8),
            (10, 4), (10, 5), (10, 6), (10, 7),
            (8, 2), (8, 3), (8, 4), (9, 4),
            (12, 2), (12, 3), (12, 4), (11, 4),
            (10, 8), (10, 9), (10, 10), (10, 11),
            (6, 11), (7, 11), (8, 11), (9, 11),
            (6, 10)
        }

        self.agent_pos, self.goal_pos = self._sample_agent_goal(width, height)

        # Place the agent in the mid-left
        # self.agent_pos = (1, 1)
        self.agent_dir = random.randint(0, 3)

        # Place a goal square in the mid-right
        # self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        for grid_sq in self.wall_squares:
            self.grid.vert_wall(grid_sq[0], grid_sq[1], 1, self.wall_type)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def _sample_agent_goal(self, width, height):

        available_squares = []
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if (i, j) not in self.wall_squares:
                    available_squares.append((i, j))

        agent_pos = random.choice(available_squares)
        available_squares.remove(agent_pos)

        # Find a goal at least 5 distance way
        available_pos = [potential_goal  for potential_goal in available_squares
                         if 10 >= self._l1dist(potential_goal, agent_pos) >= 5]

        if len(available_pos) > 0:
            goal = random.choice(available_pos)
        else:
            goal = random.choice(available_squares)

        return agent_pos, goal

    @staticmethod
    def _l1dist(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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
        elif action == self.actions.forward \
                or action == self.actions.left_forward or action == self.actions.right_forward:

            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos

            if fwd_cell is not None and fwd_cell.type == 'goal':
                done = True
                self.agent_pos = fwd_pos
                reward = self._goal_reward()

            if fwd_cell is not None and fwd_cell.type == 'lava':
                done = True
                self.agent_pos = fwd_pos
                reward = self._lava_reward()
        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = None
        # obs = self.gen_obs()

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
