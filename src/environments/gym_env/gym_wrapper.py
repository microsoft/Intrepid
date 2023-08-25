import gym
import numpy as np

from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from environments.intrepid_env_meta.environment_keys import EnvKeys
from environments.intrepid_env_meta.intrepid_env_interface import IntrepidEnvInterface


class GymWrapper(IntrepidEnvInterface):

    TARGET_SHAPE = (8, 11)
    MAX_PIX_VALUE = 8

    def __init__(self, name, config):
        self.name = name
        self.base_env = gym.make(name)
        self.config = config
        self.horizon = config["horizon"]
        self.height = self.config["obs_dim"][0]
        self.width = self.config["obs_dim"][1]
        self.channels = self.config["obs_dim"][2]
        self.num_repeat_action = self.config["num_repeat_action"]

        self.timestep = 0

        with open("%s/progress.csv" % config["save_path"], "w") as f:
            f.write("Episode,     Moving Avg.,     Mean Return\n")
        self.moving_avg = deque([], maxlen=10)
        self.sum_return = 0
        self.num_eps = 0
        self._eps_return = 0.0
        self.save_path = config["save_path"]

    def reset(self, generate_obs=True):
        self.timestep = 0
        obs = self.base_env.reset()
        processed_obs = self._process_obs(obs)
        state = self._get_state()

        info = {
            EnvKeys.STATE: state,
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
                    f.write("%d,     %f,    %f\n" % (self.num_eps, mov_avg, mean_result))

        self._eps_return = 0.0
        self.num_eps += 1  # Index of current episode starting from 0

        return processed_obs, info

    def step(self, action, generate_obs=True):
        if self.timestep > self.horizon:
            raise AssertionError("Cannot take more actions than horizon %d" % self.horizon)

        self.timestep += 1

        obs, reward, done, info = None, None, None, None
        for _ in range(self.num_repeat_action):
            obs, reward, done, info = self.base_env.step(action)

        processed_obs = self._process_obs(obs)
        state = self._get_state()

        info.update(
            {
                EnvKeys.STATE: state,
                EnvKeys.ENDO_STATE: state,
                EnvKeys.TIME_STEP: self.timestep,
            }
        )

        self._eps_return += reward

        done = done or self.timestep == self.horizon

        return processed_obs, reward, done, info

    def _get_state(self):
        return self._ram_state()

    @staticmethod
    def convert_state(state):
        import cv2

        return (
            (
                cv2.resize(
                    cv2.cvtColor(state, cv2.COLOR_RGB2GRAY),
                    GymWrapper.TARGET_SHAPE,
                    interpolation=cv2.INTER_AREA,
                )
                / 255.0
            )
            * GymWrapper.MAX_PIX_VALUE
        ).astype(np.uint8)

    def _ram_state(self):
        """Create State for OpenAI gym_env using RAM. This is useful for debugging."""

        ram = self.base_env.env._get_ram()

        if self.name == "MontezumaRevengeDeterministic-v4":
            # x, y position and orientation of agent, x-position of the skull and position of items like key
            state = "(%d, %d, %d, %d, %d)" % (
                ram[42],
                ram[43],
                ram[52],
                ram[47],
                ram[67],
            )
            return state
        else:
            return ram

    def _process_obs(self, obs):
        obs = obs / 256.0

        if self.name == "MontezumaRevengeDeterministic-v4":
            obs = obs[34 : 34 + 160, :160]  # 160 x 160 x 3

            if self.channels == 3:
                obs = resize(obs, (self.height, self.width, 3), mode="constant")
                obs = np.expand_dims(obs, 3)
                return obs

            elif self.channels == 1:
                obs = resize(rgb2gray(obs), (self.height, self.width), mode="constant")
                obs = np.expand_dims(obs, 2)
                return obs

            else:
                raise AssertionError("Image can be either black and white or RGB")
        else:
            return obs

    def get_action_type(self):
        """
        :return:
            action_type:     Return type of action space the agent is using
        """
        raise NotImplementedError()

    def save(self, save_path, fname=None):
        """
        Save the environment
        :param save_path:   Save directory
        :param fname:       Additionally, a file name can be provided. If save is a single file, then this will be
                            used else it can be ignored.
        :return: None
        """
        raise NotImplementedError()

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
        Should return V* but we return None since we dont know this value apriori for openai gym environments
        :returns: V*.
        """
        return None
