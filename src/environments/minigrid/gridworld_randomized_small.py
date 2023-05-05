import random
from collections import deque

from gym_minigrid.minigrid import *


class GridWorldRandSmall(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    env_name = "gridworld-randomized-small"

    WALL_SQUARES = {
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 5),
        (3, 6),
        (3, 7),
        (3, 8),
        (3, 9),
        (2, 3),
        (4, 5),
        (5, 5),
        (2, 8),
        (3, 8),
        (4, 8),
        (5, 8),
        (6, 8),
        (7, 8),
        (8, 8),
        (9, 8),
        (10, 8),
        (11, 8),
        (9, 4),
        (9, 5),
        (9, 6),
        (9, 7),
        (8, 2),
        (8, 3),
        (8, 4),
        (9, 4),
        (6, 11),
        (7, 11),
        (8, 11),
        (9, 11),
        (6, 10),
    }

    MIN_GOAL_DIST = 5
    MAX_GOAL_DIST = 10

    BFS_Map = dict()
    BFS_MAP_KEYS = []

    # Dictionary over state which gives dictionary of goals with value of action path that takes them to the goal
    BFS_PATH = dict()

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
        self.goal_reached = False

        self._create_bfs_map(height, width)

        super().__init__(
            width=width,
            height=height,
            max_steps=horizon,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=seed,
            agent_view_size=agent_view_size,
        )

        self.min_dist_to_goal = None

        # Actions are discrete integer values
        self.actions = GridWorldRandSmall.Actions
        self.action_space = spaces.Discrete(len(self.actions))

        self.reward_decay_ratio = 0.1  # config["reward_decay_ratio"]

    def _create_bfs_map(self, height, width):
        for w in range(1, width - 1):
            for h in range(1, height - 1):
                if (w, h) in GridWorldRandSmall.WALL_SQUARES:
                    continue

                for direction in range(0, 4):
                    start_state = (w, h, direction)

                    path_map = self._get_path_map(start_state, width, height)

                    GridWorldRandSmall.BFS_PATH[start_state] = path_map

                    selected_goals = [
                        goal
                        for goal, path in path_map.items()
                        if GridWorldRandSmall.MIN_GOAL_DIST
                        <= len(path)
                        <= GridWorldRandSmall.MAX_GOAL_DIST
                    ]

                    if len(selected_goals) > 0:
                        GridWorldRandSmall.BFS_Map[start_state] = selected_goals

        assert len(GridWorldRandSmall.BFS_Map) != 0, "Cannot find any valid nodes"

        GridWorldRandSmall.BFS_MAP_KEYS = list(GridWorldRandSmall.BFS_Map.keys())

    def _get_path_map(self, start_state, width, height):
        path_map = dict()  # Goals along with action sequence that takes the agent there

        queue = deque([start_state])
        path_map[start_state] = []

        while len(queue) > 0:
            state = queue.popleft()
            children = self._get_children(state, width, height)

            for action, child in children.items():
                if child in path_map:
                    continue

                # This is the first time we have seen this state
                parent_path = list(path_map[state])
                parent_path.append(action)
                path_map[child] = parent_path

                queue.append(child)

        # Make the path directionless, i.e., we coalesce all goal states that have the same position
        # but different direction
        coalesced_path_map = dict()

        for goal, path in path_map.items():
            goal_pos = goal[0], goal[1]

            if goal_pos in coalesced_path_map:
                if len(path) < len(coalesced_path_map[goal_pos]):
                    coalesced_path_map[goal_pos] = path
            else:
                coalesced_path_map[goal_pos] = path

        return coalesced_path_map

    @staticmethod
    def _get_children(state, width, height):
        children = dict()
        children_set = set()

        w, h, direction = state

        for action in range(0, 3):
            if action == GridWorldRandSmall.Actions.left:
                new_state = (w, h, (direction - 1) % 4)

            elif action == GridWorldRandSmall.Actions.right:
                new_state = (w, h, (direction + 1) % 4)

            elif action == GridWorldRandSmall.Actions.forward:
                if direction == 0:  # right
                    new_state = (w + 1, h, direction)
                elif direction == 1:  # down
                    new_state = (w, h + 1, direction)
                elif direction == 2:  # left
                    new_state = (w - 1, h, direction)
                elif direction == 3:  # top
                    new_state = (w, h - 1, direction)
                else:
                    raise AssertionError("Direction can only be in {0, 1, 2, 3}")

                if (
                    new_state[0] == 0
                    or new_state[0] == width - 1
                    or new_state[1] == 0
                    or new_state[1] == height - 1
                    or (new_state[0], new_state[1]) in GridWorldRandSmall.WALL_SQUARES
                ):
                    continue

            else:
                raise NotImplementedError()

            if new_state not in children_set:
                children_set.add(new_state)
                children[action] = new_state

        return children

    def _gen_grid(self, width, height):
        assert width == 12 and height == 12

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.wall_squares = GridWorldRandSmall.WALL_SQUARES

        # Sample agent's start position and the goal position. Note that goal is always chosen above and below a certain
        # range from the agent's position.
        self.agent_pos, self.agent_dir, self.goal_pos = self._sample_agent_goal()

        agent_state = self.agent_pos[0], self.agent_pos[1], self.agent_dir

        self.min_dist_to_goal = len(
            GridWorldRandSmall.BFS_PATH[agent_state][self.goal_pos]
        )

        self.put_obj(Key("yellow"), *self.goal_pos)

        for grid_sq in self.wall_squares:
            self.grid.vert_wall(grid_sq[0], grid_sq[1], 1, self.wall_type)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    @staticmethod
    def _sample_agent_goal():
        agent_pos = random.choice(list(GridWorldRandSmall.BFS_MAP_KEYS))
        goal = random.choice(GridWorldRandSmall.BFS_Map[agent_pos])

        return agent_pos[:2], agent_pos[2], goal

    def get_current_goal(self):
        return self.goal_pos

    def sample_goal(self):
        agent_state = self.agent_pos[0], self.agent_pos[1], self.agent_dir
        return random.choice(GridWorldRandSmall.BFS_Map[agent_state])

    @staticmethod
    def _l1dist(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reset(self):
        self.goal_reached = False
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

            if fwd_cell is not None and (
                fwd_cell.type == "goal" or fwd_cell.type == "key"
            ):
                done = True
                self.goal_reached = True
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

        self.last_done = done

        return obs, reward, done, {}

    def get_optimal_action(self):
        return self.get_goal_pos_action(self.goal_pos)

    def get_goal_pos_action(self, goal_pos):
        agent_state = self.agent_pos[0], self.agent_pos[1], self.agent_dir
        paths = self.BFS_PATH[agent_state][goal_pos]

        if len(paths) == 0:
            # Already at goal, then simply rotate
            return 0
        else:
            # Return the first action to the path
            return paths[0]

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
