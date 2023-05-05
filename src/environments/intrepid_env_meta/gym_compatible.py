import gym
from gym import spaces


class GymCompatible(gym.Env):
    def __init__(self, cerebral_env):
        self.cerebral_env = cerebral_env
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = None  # TODO spaces.Discrete(N_DISCRETE_ACTIONS)

        # Example for using image as input:
        self.observation_space = None
        # self.observation_space = spaces.Box(low=0, high=255, shape=
        # (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)      # TODO

    def step(self, action):
        # Execute one time step within the environment
        return self.cerebral_env.step(action)

    def reset(self):
        # Reset the state of the environment to an initial state
        obs, info = self.cerebral_env.reset()
        return obs

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        raise NotImplementedError()
