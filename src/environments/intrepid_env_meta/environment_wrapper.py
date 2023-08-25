import gym
import numpy as np
import os
import pickle
import time
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from environments.rl_acid_env.combolock import CombinationLock
from environments.rl_acid_env.grid_world import GridWorld
from environments.rl_acid_env.diabolical_combolock import DiabolicalCombinationLock
from environments.control_env.newtonian_motion import NewtonianMotion
from environments.control_env.simple_lqr import SimpleLQR
from environments.rl_acid_env.slot_factored_mdp import SlotFactoredMDP
from environments.rl_acid_env.visual_combolock import VisualComboLock
from environments.rl_acid_env.noise_gen import get_sylvester_hadamhard_matrix_dim


class GenerateEnvironmentWrapper:
    """ " Wrapper class for generating environments using names and config"""

    OpenAIGym, RL_ACID, GRIDWORLD, VISUALCOMBOLOCK, MATTERPORT, SLOTFACTOREDMDP = range(6)

    def __init__(self, env_name, config, bootstrap_env=None):
        """
        :param env_name: Name of the environment to create
        :param config:  Configuration to use
        :param bootstrap_env: Environment used for defining
        """

        self.tolerance = 0.5
        self.env_type = None
        self.env_name = env_name
        self.config = config

        # A boolean flag indicating if we should save traces or not
        # A trace is a sequence of {(obs, state, action, reward, obs, state, ...., obs, state)}
        self.save_trace = config["save_trace"]
        self.trace_sample_rate = config["trace_sample_rate"]  # How many often should we save the traces
        self.trace_folder = config["save_path"] + "/traces"  # Folder for saving traces
        self.trace_data = []  # Set of currently unsaved traces
        self.current_trace = None  # Current trace
        self.num_eps = 0  # Number of episodes passed
        self.sum_total_reward = 0.0  # Total reward received by the agent
        self._sum_this_episode = 0.0  # Total reward received in current episode
        self.moving_average_reward = deque(maxlen=10)  # For moving average calculation

        # self.observation_space = gym_env.spaces.Box(low=0.0, high=1.0, shape=(config["obs_dim"],), dtype=np.float)
        # self.action_space = gym_env.spaces.Discrete(config["num_actions"])

        # Create a folder for saving traces
        if not os.path.exists(self.trace_folder):
            os.makedirs(self.trace_folder)

        if env_name == "combolock":
            # Deterministic Combination Lock

            self.env_type = GenerateEnvironmentWrapper.RL_ACID
            self.thread_safe = True
            self.reward_range = (0.0, config["optimal_reward"])
            self.metadata = None

            self.env = bootstrap_env if bootstrap_env is not None else CombinationLock(config=config)
            self.homing_policy_validation_fn = self.env.get_homing_policy_validation_fn(tolerance=self.tolerance)

        elif env_name == "diabcombolock":
            # Diabolical Stochastic Combination Lock

            self.env_type = GenerateEnvironmentWrapper.RL_ACID
            self.thread_safe = True
            self.reward_range = (0.0, config["optimal_reward"])
            self.metadata = None

            self.env = bootstrap_env if bootstrap_env is not None else DiabolicalCombinationLock(config=config)
            self.homing_policy_validation_fn = self.env.get_homing_policy_validation_fn(tolerance=self.tolerance)

        elif env_name == "montezuma":
            # Montezuma Revenge

            self.env_type = GenerateEnvironmentWrapper.OpenAIGym
            self.thread_safe = True
            self.num_repeat_action = 4  # Repeat each action these many times.

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = gym.make(
                    "MontezumaRevengeDeterministic-v4",
                    obs_type=config.get("obs_type", "image"),
                )

            # Since we don't have access to underline state in this problem, we cannot define a validation function
            self.homing_policy_validation_fn = None

        elif env_name == "gridworld" or env_name == "gridworld-feat":
            # Grid World

            self.env_type = GenerateEnvironmentWrapper.GRIDWORLD
            self.thread_safe = True

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = GridWorld(
                    num_grid_row=4,
                    num_grid_col=4,
                    horizon=config["horizon"],
                    obs_dim=config["obs_dim"],
                )

            reachable_states = self.env.get_reachable_states()
            num_states = self.env.get_num_states()

            self.homing_policy_validation_fn = lambda dist, step: all(
                [
                    str(state) in dist and dist[str(state)] >= 1.0 / float(max(1, num_states)) - self.tolerance
                    for state in reachable_states[step]
                ]
            )

        elif env_name == "visualcombolock":
            # Visual Combo Lock

            self.env_type = GenerateEnvironmentWrapper.VISUALCOMBOLOCK
            self.thread_safe = True

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = VisualComboLock(
                    horizon=config["horizon"],
                    swap=0.5,
                    num_actions=10,
                    anti_shaping_reward=0.1,
                    obs_dim=config["obs_dim"],
                    vary_instance=True,
                )

            # TODO make this validation function stricter like for other combination lock environments
            self.homing_policy_validation_fn = (
                lambda dist, step: str((0, step)) in dist
                and str((1, step)) in dist
                and str((2, step)) in dist
                and dist[str((0, step))] > 20 - self.tolerance
                and dist[str((1, step))] > 20 - self.tolerance
                and dist[str((2, step))] > 20 - self.tolerance
            )

        elif env_name == "newtonianmotion":
            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = NewtonianMotion(config)

            self.homing_policy_validation_fn = None

        elif env_name == "simplelqr":
            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = SimpleLQR(config)

            self.homing_policy_validation_fn = None

        elif env_name == "matterport":
            # MatterPort
            import MatterSim

            self.env_type = GenerateEnvironmentWrapper.MATTERPORT

            # We will use a specific house and room for now
            # TODO: use general house and room in future
            self.house_id = "17DRP5sb8fy"
            self.room_id = "0f37bd0737e349de9d536263a4bdd60d"
            self.thread_safe = True
            self.num_repeat_action = 1  # Repeat each action these many times.

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = MatterSim.Simulator()
                self.env.setCameraResolution(640, 480)
                self.env.setPreloadingEnabled(False)
                self.env.setDepthEnabled(False)
                self.env.setBatchSize(1)
                self.env.setCacheSize(2)

                self.env.setDatasetPath("/mnt/data/matterport/v1/scans")
                self.env.setNavGraphPath("/mnt/data/matterport/v1/connectivity/")

                self.env.initialize()

            # Since we don't have access to underline state in this problem, we cannot define a validation function
            self.homing_policy_validation_fn = None

        elif env_name == "slotfactoredmdp":
            self.env_type = GenerateEnvironmentWrapper.SLOTFACTOREDMDP

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = SlotFactoredMDP(config)

            self.homing_policy_validation_fn = None

        else:
            raise AssertionError("Environment name %r not in RL Acid Environments " % env_name)

    def generate_homing_policy_validation_fn(self):
        if self.homing_policy_validation_fn is not None:
            return self.homing_policy_validation_fn

    def step(self, action):
        if self.env_type == GenerateEnvironmentWrapper.RL_ACID:
            observation, reward, info = self.env.act(action)
            done = self.env.h == self.config["horizon"]

            if self.save_trace:
                self.current_trace.extend([action, reward, observation, info["state"]])
            self._sum_this_episode += reward

            return observation, float(reward), done, info

        elif self.env_type == GenerateEnvironmentWrapper.OpenAIGym:
            # Repeat the action K steps
            for _ in range(self.num_repeat_action):
                image, reward, done, info = self.env.step(action)
            image = self.openai_gym_process_image(image)

            assert "state" not in info
            info["state"] = self.openai_ram_for_state()

            return image, float(reward), done, info

        elif self.env_type == GenerateEnvironmentWrapper.GRIDWORLD:
            image, reward, done, info = self.env.step(action)
            return image, float(reward), done, info

        elif self.env_type == GenerateEnvironmentWrapper.VISUALCOMBOLOCK:
            image, reward, done, info = self.env.step(action)
            return image, float(reward), done, info

        elif self.env_type == GenerateEnvironmentWrapper.MATTERPORT:
            # Simulator can be queried in batch
            state = self.env.getState()[0]

            if action == 0:
                # Stay in place
                pass
            elif action == 1:
                # Go straight if possible, otherwise stay in place
                if len(state.navigableLocations) > 1:
                    self.env.makeAction([1], [0], [0])
            elif action == 2:
                # Look 30 degrees left
                self.env.makeAction([0], [-0.523599], [0])
            elif action == 3:
                # Look 30 degrees right
                self.env.makeAction([0], [0.523599], [0])
            elif action == 4:
                # Look 30 degrees down
                self.env.makeAction([0], [0], [-0.523599])
            elif action == 5:
                # Look 30 degrees up
                self.env.makeAction([0], [0], [0.523599])

            state = self.env.getState()[0]
            img = np.array(state.rgb, copy=False)

            height, width, channel = img.shape
            assert height == 480 and width == 640 and channel == 3, "Wrong shape. Found %r. Expected 480 x 640 x 3" % img.shape
            img = resize(img, (height // 5, width // 5, channel)).swapaxes(2, 1).swapaxes(1, 0)
            img = np.ascontiguousarray(img)

            # return img, 0, False, {"state": img, "location": state.location.viewpointId}
            return img, 0, False, {"state": state.location.viewpointId}

        else:
            return self.env.step(action)

    def reset(self):
        if self.env_type == GenerateEnvironmentWrapper.RL_ACID:
            self.sum_total_reward += self._sum_this_episode
            if self.num_eps > 0:
                self.moving_average_reward.append(self._sum_this_episode)
            self._sum_this_episode = 0.0

            if self.num_eps % 1000 == 0:
                # TODO these calculations only make sense when reward-free planner doesn't use samples
                # TODO or multi-processing is disabled. This should be added as a condition.
                mean_result = self.sum_total_reward / float(max(1, self.num_eps))
                mean_moving_average = sum(self.moving_average_reward) / float(max(1, len(self.moving_average_reward)))
                with open(self.trace_folder + "/progress.csv", "a") as g:
                    if self.num_eps == 0:
                        g.write("Episodes Completed,   Mean Total Reward,      Mean Moving Average\n")
                    else:
                        g.write("%d \t %f \t %f \n" % (self.num_eps, mean_result, mean_moving_average))

            self.num_eps += 1  # Current episode ID

            obs, info = self.env.start_episode()

            if self.save_trace:
                # Add current trace to list of traces at a certain rate
                if self.current_trace is not None and self.num_eps % self.trace_sample_rate == 0:
                    self.trace_data.append(self.current_trace)

                # Save data if needed
                if len(self.trace_data) == 5:
                    with open(
                        self.trace_folder + "/%s_%d" % (self.env_name, time.time()),
                        "wb",
                    ) as f:  # TODO
                        pickle.dump(self.trace_data, f)
                    self.trace_data = []

                self.current_trace = [obs, info["state"]]

            if self.config["return_state"]:
                return obs, info
            else:
                return obs

        elif self.env_type == GenerateEnvironmentWrapper.OpenAIGym:
            image = self.env.reset()
            image = self.openai_gym_process_image(image)
            return image, {"state": self.openai_ram_for_state()}

        elif self.env_type == GenerateEnvironmentWrapper.GRIDWORLD:
            return self.env.reset()

        elif self.env_type == GenerateEnvironmentWrapper.VISUALCOMBOLOCK:
            return self.env.reset()

        elif self.env_type == GenerateEnvironmentWrapper.MATTERPORT:
            self.env.newEpisode([self.house_id], [self.room_id], [0], [0])

            # self.env.newRandomEpisode([self.house_id])
            state = self.env.getState()[0]
            img = np.array(state.rgb, copy=False)
            height, width, channel = img.shape
            assert height == 480 and width == 640 and channel == 3, "Wrong shape. Found %r. Expected 480 x 640 x 3" % img.shape
            img = resize(img, (height // 5, width // 5, channel)).swapaxes(2, 1).swapaxes(1, 0)
            img = np.ascontiguousarray(img)

            # return img, {"state": img, "location": state.location.viewpointId}
            return img, {"state": state.location.viewpointId}

        else:
            return self.env.reset()

    def openai_gym_process_image(self, image):
        if self.env_name == "montezuma":
            image = image[34 : 34 + 160, :160]  # 160 x 160 x 3
            image = image / 256.0

            if self.config["obs_dim"] == [1, 160, 160, 3]:
                return image
            elif self.config["obs_dim"] == [1, 84, 84, 1]:
                image = resize(rgb2gray(image), (84, 84), mode="constant")
                image = np.expand_dims(image, 2)  # 84 x 84 x 1
                return image
            else:
                raise AssertionError("Unhandled configuration %r" % self.config["obs_dim"])
        else:
            raise AssertionError("Unhandled OpenAI Gym environment %r" % self.env_name)

    def openai_ram_for_state(self):
        """Create State for OpenAI gym_env using RAM. This is useful for debugging."""

        ram = self.env.env._get_ram()

        if self.env_name == "montezuma":
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
            raise NotImplementedError()

    def get_optimal_value(self):
        if (
            self.env_name == "combolock"
            or self.env_name == "stochcombolock"
            or self.env_name == "diabcombolock"
            or self.env_name == "visualcombolock"
        ):
            return self.env.get_optimal_value()
        else:
            return None

    def is_thread_safe(self):
        return self.thread_safe

    @staticmethod
    def adapt_config_to_domain(env_name, config):
        """This function adapts the config to the environment."""

        if config["obs_dim"] == -1:
            if env_name == "combolock":
                config["obs_dim"] = 3 * config["horizon"] + 2

            elif env_name == "stochcombolock" or env_name == "diabcombolock":
                if config["noise"] == "bernoulli":
                    config["obs_dim"] = 2 * config["horizon"] + 4
                elif config["noise"] == "gaussian":
                    config["obs_dim"] = config["horizon"] + 4
                elif config["noise"] == "hadamhard":
                    config["obs_dim"] = get_sylvester_hadamhard_matrix_dim(config["horizon"] + 4)
                elif config["noise"] == "hadamhardg":
                    config["obs_dim"] = get_sylvester_hadamhard_matrix_dim(config["horizon"] + 4)
                else:
                    raise AssertionError("Unhandled noise type %r" % config["noise"])

            elif env_name == "slotfactoredmdp":
                config["obs_dim"] = 2 * config["state_dim"]  # config["grid_x"] * config["grid_y"] * config["grid_x"]

            else:
                raise AssertionError("Cannot adapt to unhandled environment %s" % env_name)

    def get_bootstrap_env(self):
        """Environments which are thread safe can be bootstrapped. There are two ways to do so:
        1. Environment with internal state which can be replicated directly.
            In this case we return the internal environment.
        2. Environments without internal state which can be created exactly from their name.
            In this case we return None"""

        assert self.thread_safe, "To bootstrap it must be thread safe"
        if (
            self.env_name == "stochcombolock"
            or self.env_name == "combolock"
            or self.env_name == "diabcombolock"
            or self.env_name == "visualcombolock"
        ):
            return self.env
        else:
            return None

    def save_environment(self, folder_name, trial_name):
        if self.env_type == GenerateEnvironmentWrapper.RL_ACID or self.env_type == GenerateEnvironmentWrapper.VISUALCOMBOLOCK:
            return self.env.save(folder_name + "/trial_%r_env" % trial_name)
        else:
            pass  # Nothing to save

    def load_environment_from_folder(self, env_folder_name):
        if self.env_type == GenerateEnvironmentWrapper.RL_ACID or self.env_type == GenerateEnvironmentWrapper.VISUALCOMBOLOCK:
            self.env = self.env.load(env_folder_name)
        else:
            raise AssertionError("Cannot load environment for Non RL Acid settings")

    def is_deterministic(self):
        raise NotImplementedError()

    @staticmethod
    def make_env(env_name, config, bootstrap_env):
        return GenerateEnvironmentWrapper(env_name, config, bootstrap_env)
