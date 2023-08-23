import math
import numpy as np
import torch
import torch.nn.functional as F

from collections import deque
from environments.minigrid.exogenous_noise_util import get_exo_util
from environments.cerebral_env_meta.environment_keys import EnvKeys
from environments.cerebral_env_meta.cerebral_env_interface import CerebralEnvInterface
from model.policy.stationary_constant_policy import StationaryConstantPolicy


class GridWorldWrapper(CerebralEnvInterface):
    def __init__(self, env, config, logger):
        self.timestep = -1  # Current time step
        self.horizon = config["horizon"]
        self.actions = config["actions"]
        self.env = env
        self.ego_centric = config["ego_centric"] > 0

        # self.env = RGBImgPartialObsWrapper(self.env)
        # self.env = RGBImgObsWrapper(self.env, tile_size=8)
        # self.env = ViewSizeWrapper(self.env, agent_view_size=3)

        self.grid_height = config["height"]
        self.grid_width = config["width"]
        self.tile_width = config["tile_size"]
        self.tile_height = config["tile_size"]
        self.agent_view_size = config["agent_view_size"]

        if self.ego_centric:
            self.height, self.width = (
                config["agent_view_size"] * config["tile_size"],
                config["agent_view_size"] * config["tile_size"],
            )
        else:
            self.height, self.width = (
                config["height"] * config["tile_size"],
                config["width"] * config["tile_size"],
            )

        self.enable_exo = config["enable_exo"] > 0
        self.exo_type = config["exo_type"]
        self.use_exo_reward = config["exo_reward"] > 0
        self.color_map = config["color_map"] > 0

        if (
            self.enable_exo or self.use_exo_reward
        ):  # works with both bool type and integers/floats
            self.exo_util = get_exo_util(self.exo_type, config)
            logger.log("Created exogenous noise of type %r" % self.exo_type)
        else:
            self.exo_util = None
            logger.log("No exogenous noise enabled")

        if self.use_exo_reward:
            logger.log("Use exo reward")
        else:
            logger.log("Will not use exo reward")

        if self.color_map:
            logger.log("Using color map")

        with open("%s/progress.csv" % config["save_path"], "w") as f:
            f.write("Episode,     Moving Avg.,     Mean Return\n")

        self.moving_avg = deque([], maxlen=10)
        self.sum_return = 0
        self.num_eps = 0
        self._eps_return = 0.0
        self.save_path = config["save_path"]
        self.feature_type = config["feature_type"]
        self.tile_size = config["tile_size"]

        self._homing_policies = None

        self.get_perfect_homing_policy(self.horizon)

        # self.color_map = np.load('color_map.npy')

        self.gen = torch.Generator()
        self.gen.manual_seed(999)

        if self.color_map:
            self.color_map_filter = torch.rand(
                size=(1, 3, self.grid_width, self.grid_height), generator=self.gen
            )
            self.color_map_filter = F.interpolate(
                self.color_map_filter,
                size=(
                    config["height"] * config["tile_size"],
                    config["width"] * config["tile_size"],
                ),
            ).squeeze(0)
            self.color_map_filter = self.color_map_filter.permute(1, 2, 0).numpy()
            self.color_map_filter = self.color_map_filter * 0.9 + 0.1

    def act_to_str(self, action):
        if action == 0:
            return "left"
        elif action == 1:
            return "right"
        elif action == 2:
            return "forward"
        elif action == 3:
            return "left+forward"
        elif action == 4:
            return "right+forward"
        else:
            raise AssertionError("Action must be in {0, 1, 2, 3, 4}")

    def start(self):
        raise NotImplementedError()

    def transition(self, state, action):
        raise NotImplementedError()

    def reward(self, state, action, new_state):
        raise NotImplementedError()

    def get_env_name(self):
        raise NotImplementedError()

    def get_actions(self):
        raise NotImplementedError()

    def get_num_actions(self):
        raise NotImplementedError()

    def get_horizon(self):
        raise NotImplementedError()

    def get_endogenous_state(self, state):
        return state

    def reset(self, generate_obs=True):
        """
        :return:
            obs:        Agent observation. No assumption made on the structure of observation.
            info:       Dictionary containing relevant information such as latent state, etc.
        """

        self.env.reset()

        self.timestep = 0
        state = (
            self.env.agent_pos[0],
            self.env.agent_pos[1],
            self.env.agent_dir,
            self.timestep,
        )
        info = {
            "state": state,
            EnvKeys.ENDO_STATE: state,
            EnvKeys.TIME_STEP: self.timestep,
        }

        if self.num_eps > 0:
            self.moving_avg.append(self._eps_return)
            self.sum_return += self._eps_return

            if self.num_eps % 100 == 0:
                mov_avg = sum(self.moving_avg) / float(len(self.moving_avg))
                mean_result = self.sum_return / float(self.num_eps)

                with open("%s/progress.csv" % self.save_path, "a") as f:
                    f.write(
                        "%d,     %f,    %f\n" % (self.num_eps, mov_avg, mean_result)
                    )

        self._eps_return = 0.0
        self.num_eps += 1  # Index of current episode starting from 0

        # Reset exogenous information
        if self.exo_util is not None:
            self.exo_util.reset()

        # Generate observation
        if generate_obs:
            img = self.env.render(
                "rgb_array", tile_size=self.tile_size, highlight=False
            )
            img, gen_info_dict = self.generate_image(img, state)

            for key, val in gen_info_dict.items():
                info[key] = val

        else:
            img = None

        return img, info

    def to_egocentric(self, img, agent_state):
        # image is of size (height x tile_width) * (width x tile_width) * 3
        # agent_y is vertical position and agent_x is horizontal position
        agent_x, agent_y, agent_dir, _ = agent_state

        new_image = np.zeros(
            (
                self.agent_view_size * self.tile_height,
                self.agent_view_size * self.tile_width,
                3,
            ),
            dtype=np.uint8,
        )

        if agent_dir == 0:  # Right
            start_x = min(agent_x * self.tile_width, self.grid_width * self.tile_width)
            end_x = min(
                (agent_x + self.agent_view_size) * self.tile_width,
                self.grid_width * self.tile_width,
            )

            val = (agent_y - math.floor(self.agent_view_size / 2.0)) * self.tile_height
            start_y = min(max(val, 0), self.grid_height * self.tile_height)
            end_y = min(
                (agent_y + math.ceil(self.agent_view_size / 2.0)) * self.tile_height,
                self.grid_height * self.tile_height,
            )

            # agent_view_size * tile_height and agent_view_size * tile_width
            pad_up = 0  # - val if val < 0 else 0

            # Cropping
            img2 = img[
                start_y:end_y, start_x:end_x, :
            ]  # Height is first, and height is y

            new_image[pad_up : pad_up + img2.shape[0], 0 : img2.shape[1], :] = img2

        elif agent_dir == 1:  # Down
            start_y = min(
                agent_y * self.tile_height, self.grid_height * self.tile_height
            )
            end_y = min(
                max((agent_y + self.agent_view_size) * self.tile_height, 0),
                self.grid_height * self.tile_height,
            )

            val = (agent_x - math.floor(self.agent_view_size / 2.0)) * self.tile_width
            start_x = min(max(val, 0), self.grid_height * self.tile_width)
            end_x = min(
                (agent_x + math.ceil(self.agent_view_size / 2.0)) * self.tile_width,
                self.grid_width * self.tile_width,
            )

            # agent_view_size * tile_height and agent_view_size * tile_width
            pad_up = 0  # - val if val < 0 else 0

            # Cropping
            img2 = img[
                start_y:end_y, start_x:end_x, :
            ]  # Height is first, and height is y

            img2 = np.rot90(img2, 1, (0, 1))
            new_image[pad_up : pad_up + img2.shape[0], 0 : img2.shape[1], :] = img2

        elif agent_dir == 2:  # Left
            start_x = min(
                max((agent_x - self.agent_view_size + 1) * self.tile_width, 0),
                self.grid_width * self.tile_width,
            )
            end_x = min(
                (agent_x + 1) * self.tile_width, self.grid_width * self.tile_width
            )

            val = (agent_y - math.floor(self.agent_view_size / 2.0)) * self.tile_height
            start_y = min(max(val, 0), self.grid_height * self.tile_height)
            end_y = min(
                ((agent_y + math.ceil(self.agent_view_size / 2.0)) * self.tile_height),
                self.grid_height * self.tile_height,
            )

            # agent_view_size * tile_height and agent_view_size * tile_width
            pad_up = 0  # - val if val < 0 else 0

            # Cropping
            img2 = img[
                start_y:end_y, start_x:end_x, :
            ]  # Height is first, and height is y

            img2 = np.rot90(img2, 2, (0, 1))
            new_image[pad_up : pad_up + img2.shape[0], 0 : img2.shape[1], :] = img2

        elif agent_dir == 3:  # Top
            start_y = min(
                max((agent_y - self.agent_view_size + 1) * self.tile_height, 0),
                self.grid_height * self.tile_height,
            )
            end_y = min(
                (agent_y + 1) * self.tile_height, self.grid_height * self.tile_height
            )

            val = (agent_x - math.floor(self.agent_view_size / 2.0)) * self.tile_width
            start_x = min(max(val, 0), self.grid_height * self.tile_width)
            end_x = min(
                (agent_x + math.ceil(self.agent_view_size / 2.0)) * self.tile_width,
                self.grid_width * self.tile_width,
            )

            # agent_view_size * tile_height and agent_view_size * tile_width
            pad_up = 0  # - val if val < 0 else 0

            # Cropping
            img2 = img[
                start_y:end_y, start_x:end_x, :
            ]  # Height is first, and height is y

            img2 = np.rot90(img2, 3, (0, 1))
            # pdb.set_trace()
            new_image[pad_up : pad_up + img2.shape[0], 0 : img2.shape[1], :] = img2

        else:
            raise AssertionError(
                "Agent direction must be in {0, 1, 2, 3}. Found %r" % agent_dir
            )

        new_image = torch.nn.functional.interpolate(
            torch.Tensor(new_image).unsqueeze(0).permute(0, 3, 1, 2), size=(56, 56)
        )
        new_image = new_image.squeeze(0).permute(1, 2, 0)
        new_image = new_image.numpy()

        return new_image

    def generate_image(self, img, state):
        # Original fully-observed image
        orig = img

        # multiply by color_map
        if self.color_map:
            # color_map = torch.rand(size=(96,96,3), generator=self.gen).numpy()
            after_color_map = self.color_map_filter * img
            img = after_color_map
        else:
            after_color_map = None

        if self.ego_centric:
            img = self.to_egocentric(img, state)

        if self.enable_exo:
            img, exo_noise = self.exo_util.generate_image(img)
        else:
            exo_noise = None

        # Scale down image(s)
        img = img / 255.0
        orig = orig / 255.0

        if exo_noise is not None:
            exo_noise = exo_noise / 255.0

        # Possible convert the image to grayscale if needed
        # img = color.rgb2gray(img / 255.0)

        gen_info_dict = {"full": orig}

        if exo_noise is not None:
            gen_info_dict["exo_noise"] = exo_noise

        # Comment out the block below if needed during debugging
        # import matplotlib.pyplot as plt
        #
        # plt.clf()
        # f, axarr = plt.subplots(1, 4)
        #
        # axarr[0].imshow(orig)
        # axarr[0].title.set_text("Original image")
        #
        # if after_color_map is not None:
        #     axarr[1].imshow(after_color_map)
        #     axarr[1].title.set_text("After Color Map")
        # else:
        #     axarr[1].imshow(orig)
        #     axarr[1].title.set_text("No Color Map")
        #
        # if exo_noise is not None:
        #     axarr[2].imshow(exo_noise)
        #     axarr[2].title.set_text("Exogenous Noise")
        # else:
        #     # axarr[2].imshow(exo_noise)
        #     axarr[2].title.set_text("No Exogenous Noise")
        #
        # axarr[3].imshow(img)
        # axarr[3].title.set_text("Final Image")
        #
        # plt.show()
        # # block ends

        return img, gen_info_dict

    def step(self, action, generate_obs=True):
        """
        :param action:
        :param generate_obs: If True then observation is generated, otherwise, None is returned
        :return:
            obs:        Agent observation. No assumption made on the structure of observation.
            reward:     Reward received by the agent. No Markov assumption is made.
            done:       True if the episode has terminated and False otherwise.
            info:       Dictionary containing relevant information such as latent state, etc.
        """

        if self.timestep > self.horizon:
            raise AssertionError(
                "Cannot take more actions than horizon %d" % self.horizon
            )

        obs, reward, done, info = self.env.step(action)
        self.timestep += 1

        done = done or self.timestep == self.horizon

        state = (
            self.env.agent_pos[0],
            self.env.agent_pos[1],
            self.env.agent_dir,
            self.timestep,
        )
        info = {
            EnvKeys.STATE: state,
            EnvKeys.ENDO_STATE: state,
            EnvKeys.TIME_STEP: self.timestep,
        }

        if self.exo_util is not None:
            self.exo_util.update()

        if generate_obs:
            img = self.env.render(
                "rgb_array", tile_size=self.tile_size, highlight=False
            )

            img, gen_info_dict = self.generate_image(img, state)

            for key, val in gen_info_dict.items():
                info[key] = val

        else:
            img = None

        if self.use_exo_reward:
            exo_reward = self.exo_util.get_reward()
        else:
            exo_reward = 0.0

        # We only log the true reward but give the exogenous reward to the agent
        self._eps_return += reward

        ######
        if (
            img is not None
            and self.env.env_name == "gridworld-randomized-small"
            and self.env.goal_reached
        ):
            img = img * 0.0
        ######

        return img, (reward + exo_reward), done, info

    def save(self, save_path, fname=None):
        """
        Save the environment
        :param save_path:   Save directory
        :param fname:       Additionally, a file name can be provided. If save is a single file, then this will be
                            used else it can be ignored.
        :return: None
        """
        pass

    def load(self, load_path, fname=None):
        """
        Save the environment
        :param load_path:   Load directory
        :param fname:       Additionally, a file name can be provided. If load is a single file, then only file
                            with the given fname will be used.
        :return: Environment
        """
        raise NotImplementedError()

    def is_episodic(self):
        """
        :return:                Return True or False, True if the environment is episodic and False otherwise.
        """
        return True

    def generate_homing_policy_validation_fn(self):
        """
        :return:                Returns a validation function to test for exploration success
        """
        return None

    @staticmethod
    def adapt_config(config):
        """
            Adapt configuration file based on the environment
        :return:
        """
        raise NotImplementedError()

    def num_completed_episode(self):
        """
        :return:    Number of completed episode
        """

        return max(0, self.num_eps - 1)

    def get_mean_return(self):
        """
        :return:    Get mean return of the agent
        """
        return self.sum_return / float(max(1, self.num_completed_episode()))

    def get_optimal_value(self):
        """
            Return V* value
        :return:
        """

        # 1 for reaching goal in the fasted way and paying -0.01 for every step except the last
        # Note that once we reach the goal, the agent gets stuck and only gets reward of 0
        return 1.0 + (self.env.min_dist_to_goal - 1) * -0.01

    def _follow_path(self, policy, action):
        _ = self.reset(generate_obs=False)

        for _action in policy:
            _ = self.step(_action, generate_obs=False)

        _, _, _, info = self.step(action, generate_obs=False)

        return info[EnvKeys.ENDO_STATE]

    def get_perfect_homing_policy(self, step):
        """
        Return a list of homing policies to explore the domain
        Code currently only works for deterministic problems
        # TODO: return warning if the code is made stochastic
        :param step: A time step in {1, 2, ..., Horizon}
        :return:
        """

        if self._homing_policies is not None:
            return self._homing_policies[step]

        self._homing_policies = dict()

        assert 1 <= step <= self.horizon

        path_cover = {0: [[]]}

        for i in range(1, self.horizon + 1):
            # Homing policy for step i
            policies = set()
            path_cover[i] = []

            for policy in path_cover[i - 1]:
                for action in self.actions:
                    state = self._follow_path(policy, action)

                    if state not in policies:
                        policies.add(state)
                        actions_copy = list(policy)
                        actions_copy.append(action)
                        path_cover[i].append(actions_copy)

            self._homing_policies[i] = []

            for path in path_cover[i]:
                policy = {
                    i + 1: StationaryConstantPolicy(action)
                    for i, action in enumerate(path)
                }
                self._homing_policies[i].append(policy)

        return self._homing_policies[step]

    def calc_step(self, state, action):
        agent_pos, agent_dir = (state[0], state[1]), state[2]

        # Rotate left
        if action == self.env.actions.left or action == self.env.actions.left_forward:
            agent_dir -= 1
            if agent_dir < 0:
                agent_dir += 4

        # Rotate right
        elif (
            action == self.env.actions.right or action == self.env.actions.right_forward
        ):
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
        fwd_cell = self.env.grid.get(*fwd_pos)

        # Move forward
        if action == self.env.actions.left or action == self.env.actions.right:
            pass
        elif (
            action == self.env.actions.forward
            or action == self.env.actions.left_forward
            or action == self.env.actions.right_forward
        ):
            if fwd_cell is None or fwd_cell.can_overlap():
                agent_pos = fwd_pos

            if fwd_cell is not None and fwd_cell.type == "goal":
                agent_pos = fwd_pos

            if fwd_cell is not None and fwd_cell.type == "lava":
                agent_pos = fwd_pos

            if fwd_cell is not None and (fwd_cell.type not in ["lava", "goal", "wall"]):
                raise NotImplementedError("Unhandled cell type %s" % fwd_cell.type)
        else:
            assert False, "unknown action"

        return agent_pos[0], agent_pos[1], agent_dir

    def get_optimal_action(self):
        return self.env.get_optimal_action()

    def get_action_set(self, state1, state2, ignore_timestep=False):
        """
        Given a pair of states, return the set of all actions that can take from state1 to state2
        state1: First state
        state2: Second state to which actions from first state are to be computed
        ignore_timestep: If ignore_timestep is set to True, then ignore timestep information in the state, i.e.,
                         we remove the timestep information that is stored in the state and then compute the
                         actions. If ignore_timestep is set to False and state2's time step is not 1 more than
                          the time step of state1 then the returned action set will be empty.
        """

        action_set = set()

        for action in self.actions:
            next_state = self.calc_step(state1, action)
            next_state_with_timestep = (
                next_state[0],
                next_state[1],
                next_state[2],
                state1[3] + 1,
            )

            if ignore_timestep:
                if (
                    next_state == state2[:3]
                ):  # [:3] is done to remove timestep information
                    action_set.add(action)
            elif next_state_with_timestep == state2:
                action_set.add(action)

        return action_set
