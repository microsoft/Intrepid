import random

from gym_minigrid.minigrid import *


class GridWorldEmpty(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    env_name = "gridworld-empty"

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        left_forward = 3
        right_forward = 4

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
        self.actions = GridWorldEmpty.Actions
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

        # Place a ball square in corners to easily identify them
        self.ball_pos1 = np.array((1, 1))
        self.ball_pos2 = np.array((1, height - 2))
        self.ball_pos3 = np.array((width - 2, 1))
        self.put_obj(Ball("blue"), *self.ball_pos1)
        self.put_obj(Ball("grey"), *self.ball_pos2)
        self.put_obj(Ball("yellow"), *self.ball_pos3)

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

        obs = None
        # obs = self.gen_obs()

        return obs, reward, done, {}

    def calc_step(self, state, action):
        agent_pos, agent_dir = (state[0], state[1]), state[2]

        # Rotate left
        if action == self.actions.left or action == self.actions.left_forward:
            agent_dir -= 1
            if agent_dir < 0:
                agent_dir += 4

        # Rotate right
        elif action == self.actions.right or action == self.actions.right_forward:
            agent_dir = (agent_dir + 1) % 4

        # Get the position in front of the agent
        if agent_dir == 0:
            fwd_pos = agent_pos[0] + 1, agent_pos[1]
        elif agent_dir == 1:
            fwd_pos = agent_pos[0], agent_pos[1] + 1
        elif agent_dir == 2:
            fwd_pos = agent_pos[0] - 1, agent_pos[1]
        elif agent_dir == 3:
            fwd_pos = agent_pos[0], agent_pos[1] - 1
        else:
            raise AssertionError("Action dir has to be in {0, 1, 2, 3}")

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
                agent_pos = fwd_pos

            if fwd_cell is not None and fwd_cell.type == "goal":
                agent_pos = fwd_pos

            if fwd_cell is not None and fwd_cell.type == "lava":
                agent_pos = fwd_pos
        else:
            assert False, "unknown action"

        return agent_pos[0], agent_pos[1], agent_dir

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
