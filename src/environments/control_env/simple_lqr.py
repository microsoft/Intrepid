import numpy as np

from model.misc.lqr_model import LQRModel


class SimpleLQR:

    def __init__(self, config):

        # Dimension of the world
        self.state_dim = config["state_dim"]
        self.act_dim = config["act_dim"]
        self.obs_dim = config["obs_dim"]
        self.horizon = config["horizon"]

        assert self.state_dim == 2, "Unsupported configuration"
        assert self.act_dim == 1, "Unsupported configuration"
        assert self.obs_dim >= self.state_dim, "Cannot handle observation dimension smaller than world dimension"

        self.A = np.diag([0.9, 0.3])                    # Matrix of size 2 x 2
        self.B = np.array([[1.0], [1.0]])               # Matrix of size 2 x 1

        self.Q = np.eye(self.state_dim, dtype=np.float32)
        self.R = np.eye(self.act_dim, dtype=np.float32)

        self.latent_lqr = LQRModel(
            A=self.A,
            B=self.B,
            Q=self.Q,
            R=self.R,
            Sigma_w=np.eye(self.d),
            Sigma_0=np.eye(self.d) * 10.0
        )

        self.curr_state = None
        self.timestep = 0

    def reset(self):

        self.curr_state = 10 * np.random.randn(self.state_dim)
        self.timestep = 0
        curr_obs = self.make_obs(self.curr_state)

        info = {"state": self.curr_state, "timestep": self.timestep}

        return curr_obs, info

    def get_reward(self, state, action):

        q_cost = np.dot(state, np.matmul(self.Q, state))
        r_cost = np.dot(action, np.matmul(self.R, action))

        # Reward is negative of cost
        return - q_cost - r_cost

    def get_latent_lqr(self):
        return self.latent_lqr

    def make_obs(self, state):

        # Pad with Gaussian noise
        if self.obs_dim == self.state_dim:

            return state
        elif self.obs_dim > self.state_dim:

            noise = np.random.randn(self.obs_dim - self.state_dim)
            return np.concatenate([state, noise], axis=0)
        else:
            raise AssertionError("Cannot handle observation dimension smaller than world dimension")

    def step(self, action):
        """ Given an action which is the acceleration output. Returns
            obs, reward, done, info
        """

        # self.visualize(self.curr_state, action)

        if self.timestep >= self.horizon:
            # Cannot take any more actions
            raise AssertionError("Cannot take any more actions this episode. Horizon completed")

        else:

            new_state = np.matmul(self.A, self.curr_state) + np.matmul(self.B, action)
            noise = np.random.normal(0, 1, size=self.curr_state.shape)
            new_state = new_state + noise

            reward = self.get_reward(self.curr_state, action)

            self.curr_state = new_state
            curr_obs = self.make_obs(new_state)

            self.timestep += 1

            done = self.timestep == self.horizon

            info = {"state": self.curr_state, "timestep": self.timestep}

            return curr_obs, reward, done, info

    def get_model(self):

        return self.A, self.B, self.Q, self.R
