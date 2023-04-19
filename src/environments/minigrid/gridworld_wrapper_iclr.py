import random
import imageio
from skimage.transform import resize

from collections import deque
from PIL import Image, ImageDraw
from gym_minigrid.minigrid import *
from environments.minigrid.exogenous_noise_util import Circle, Duck
from environments.cerebral_env_meta.environment_keys import EnvKeys
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from environments.cerebral_env_meta.cerebral_env_interface import CerebralEnvInterface
from model.policy.stationary_constant_policy import StationaryConstantPolicy


class GridWorldWrapperICLR(CerebralEnvInterface):

    def __init__(self, env, config):

        self.timestep = -1  # Current time step
        self.horizon = config["horizon"]
        self.actions = config["actions"]
        self.tile_size = config["tile_size"]
        self.env = env

        # self.env = RGBImgPartialObsWrapper(self.env)
        # self.env = ImgObsWrapper(self.env)
        self.circles = None

        self.height = (config["height"] - 1) * config["tile_size"]
        self.width = (config["width"] - 1 ) * config["tile_size"]

        self.ducks = [Duck(coord=(2, 2), dir=0),
                      Duck(coord=(4, 2), dir=0)]

        self.background = imageio.imread("./data/gridworld_iclr/background.png")
        self.background = self.background[:, :, :3]
        self.background = resize(self.background, (self.height, self.width, 3)) * 255
        self.background = self.background.astype(np.uint8)

        self.duck_img = Image.open("./data/gridworld_iclr/duck.png").resize((self.tile_size, self.tile_size))

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

        ####
        self.img_ctr = 0
        ####

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
        state = (self.env.agent_pos[0], self.env.agent_pos[1], self.env.agent_dir, self.timestep)
        info = {
            "state": state,
            EnvKeys.ENDO_STATE: state,
            EnvKeys.TIME_STEP: self.timestep
        }

        if self.num_eps > 0:

            self.moving_avg.append(self._eps_return)
            self.sum_return += self._eps_return

            if self.num_eps % 100 == 0:
                mov_avg = sum(self.moving_avg) / float(len(self.moving_avg))
                mean_result = self.sum_return / float(self.num_eps)

                with open("%s/progress.csv" % self.save_path, "a") as f:
                    f.write("%d,     %f,    %f\n" % (self.num_eps, mov_avg, mean_result))

        self._eps_return = 0.0
        self.num_eps += 1  # Index of current episode starting from 0

        if generate_obs:
            # img = self.env.render('rgb_array', tile_size=self.tile_size, highlight=False)
            img = self.generate_image(state)
        else:
            img = None

        return img, info

    def generate_image(self, state):

        image = Image.fromarray(self.background)
        draw = ImageDraw.Draw(image)

        # local center of agent
        agent_x, agent_y, agent_dir, _ = state
        center_x, center_y = (agent_x - 1 + 0.5) * self.tile_size, (agent_y - 1 + 0.5) * self.tile_size

        if agent_dir == 0:
            # face left
            draw.polygon([(center_x - 0.25 * self.tile_size, center_y + 0.25 * self.tile_size),
                          (center_x - 0.25 * self.tile_size, center_y - 0.25 * self.tile_size),
                          (center_x + 0.5 * self.tile_size, center_y)],
                         fill="red")
        elif agent_dir == 1:
            # face down
            pass
        elif agent_dir == 2:
            # face right
            pass
        elif agent_dir == 3:
            # face up
            pass
        else:
            raise AssertionError("Agent direction can only take values in {0, 1, 2, 3}")

        # Paste ducks
        for duck in self.ducks:
            coord = int((duck.coord[0] - 1 + 0.5) * self.tile_size), int((duck.coord[1] - 1 + 0.5) * self.tile_size)
            image.paste(self.duck_img, coord, self.duck_img)

        img = np.array(image).astype(np.uint8)
        print("Image shape ", img.shape)

        img = img / 255.0

        return img

    def perturb(self):
        self.ducks = [self.perturb_duck(duck, self.height, self.width) for duck in self.ducks]

    def perturb_duck(self, duck, height, width):

        raise NotImplementedError()

        # Duck moves
        r = [random.choice([-1, 1]) for _ in range(4)]
        coord = circle.coord[0] + r[0] * int(self.circle_motion * width), \
                circle.coord[1] + r[1] * int(self.circle_motion * height), \
                circle.coord[2] + r[2] * int(self.circle_motion * width), \
                circle.coord[3] + r[3] * int(self.circle_motion * height)

        return Circle(coord=coord, color=circle.color, width=circle.width)

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
            raise AssertionError("Cannot take more actions than horizon %d" % self.horizon)

        obs, reward, done, info = self.env.step(action)
        self.timestep += 1

        self._eps_return += reward

        done = done or self.timestep == self.horizon

        state = (self.env.agent_pos[0], self.env.agent_pos[1], self.env.agent_dir, self.timestep)
        info = {
            "state": state,
            EnvKeys.ENDO_STATE: state,
            EnvKeys.TIME_STEP: self.timestep
        }

        # self.perturb()

        if generate_obs:
            img = imageio.imread("./data/gridworld_iclr/background_iclr.png")
            # img = self.env.render('rgb_array', tile_size=self.tile_size, highlight=False)
            img = self.generate_image(img)
        else:
            img = None

        return img, reward, done, info

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
                policy = {i + 1: StationaryConstantPolicy(action) for i, action in enumerate(path)}
                self._homing_policies[i].append(policy)

        return self._homing_policies[step]
