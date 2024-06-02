import os
import pickle

from collections import deque
from environments.intrepid_env_meta.action_type import ActionType
from environments.intrepid_env_meta.intrepid_env_interface import IntrepidEnvInterface


class RLAcidWrapper(IntrepidEnvInterface):
    """Any environment using Cerebral Env Interface must support the following API"""

    BERNOULLI, GAUSSIAN, HADAMHARD, HADAMHARDG = range(4)

    def __init__(self, config):
        self.curr_state = None  # Current state
        self.timestep = -1  # Current time step

        self.curr_eps = []
        self.traces = []
        self.save_trace = True
        self.save_trace_freq = 1

        ########
        with open("%s/progress.csv" % config["save_path"], "w") as f:
            f.write("Episode,     Moving Avg.,     Mean Return\n")
        self.moving_avg = deque([], maxlen=10)
        self.sum_return = 0
        self.num_eps = 0
        self._eps_return = 0.0
        self.save_path = config["save_path"]
        ########

    def start(self):
        raise NotImplementedError()

    def make_obs(self, state):
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

        self.curr_state = self.start()
        self.timestep = 0
        obs = self.make_obs(self.curr_state)

        info = {
            "state": self.curr_state,
            "time_step": self.timestep,
            "endogenous_state": self.get_endogenous_state(self.curr_state),
        }

        if self.num_eps > 0:
            self.moving_avg.append(self._eps_return)
            self.sum_return += self._eps_return

            if self.num_eps % 100 == 0:
                mov_avg = sum(self.moving_avg) / float(len(self.moving_avg))
                mean_result = self.sum_return / float(self.num_eps)

                with open("%s/progress.csv" % self.save_path, "a") as f:
                    f.write("%d,     %f,    %f\n" % (self.num_eps, mov_avg, mean_result))

            if self.save_trace and self.num_eps % self.save_trace_freq == 0:
                self.traces.append(list(self.curr_eps))

        self._eps_return = 0.0
        self.num_eps += 1  # Index of current episode starting from 0

        self.curr_eps = []
        self.curr_eps.append(self.curr_state)

        return obs, info

    def step(self, action, generate_obs=True):
        """
        :param action:
        :return:
            obs:        Agent observation. No assumption made on the structure of observation.
            reward:     Reward received by the agent. No Markov assumption is made.
            done:       True if the episode has terminated and False otherwise.
            info:       Dictionary containing relevant information such as latent state, etc.
        """

        horizon = self.get_horizon()

        if self.curr_state is None or self.timestep < 0:
            raise AssertionError("Environment not reset")

        if self.timestep > horizon:
            raise AssertionError("Cannot take more actions than horizon %d" % horizon)

        new_state = self.transition(self.curr_state, action)
        recv_reward = self.reward(self.curr_state, action, new_state)
        obs = self.make_obs(new_state)

        self.curr_state = new_state
        self.timestep += 1

        self._eps_return += recv_reward

        self.curr_eps.append(action)
        self.curr_eps.append(recv_reward)
        self.curr_eps.append(new_state)

        done = self.timestep == horizon

        info = {
            "state": self.get_endogenous_state(new_state),  # new_state,
            "time_step": self.timestep,
            "endogenous_state": self.get_endogenous_state(new_state),
        }

        return obs, recv_reward, done, info

    def get_action_type(self):
        """
        :return:
            action_type:     Return type of action space the agent is using
        """
        return ActionType.Discrete

    @staticmethod
    def get_noise(noise_type_str):
        if noise_type_str == "bernoulli":
            return RLAcidWrapper.BERNOULLI

        elif noise_type_str == "gaussian":
            return RLAcidWrapper.GAUSSIAN

        elif noise_type_str == "hadamhard":
            return RLAcidWrapper.HADAMHARD

        elif noise_type_str == "hadamhardg":
            return RLAcidWrapper.HADAMHARDG

        else:
            raise AssertionError("Unhandled noise type %r" % noise_type_str)

    def get_traces(self):
        return self.traces

    def save(self, save_path, fname=None):
        """
        Save the environment
        :param save_path:   Save directory
        :param fname:       Additionally, a file name can be provided. If save is a single file, then this will be
                            used else it can be ignored.
        :return: None
        """

        fname = fname if fname is not None else self.get_env_name()

        if not os.path.exists("%s" % save_path):
            os.makedirs(save_path)

        with open("%s/%s" % (save_path, fname), "wb") as f:
            pickle.dump(self, f)

    def load(self, load_path, fname=None):
        """
        Save the environment
        :param load_path:   Load directory
        :param fname:       Additionally, a file name can be provided. If load is a single file, then only file
                            with the given fname will be used.
        :return: Environment
        """

        fname = fname if fname is not None else self.get_env_name()

        with open("%s/%s" % (load_path, fname), "rb") as f:
            env = pickle.load(f)

        return env

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
        raise NotImplementedError()
