import numpy as np
import matplotlib.pyplot as plt

from model.misc.lqr_model import LQRModel


class NewtonianMotion:

    def __init__(self, config):

        # Dimension of the world
        self.world_dim = config["world_dim"]
        self.obs_dim = config["obs_dim"]
        self.obs_type = config["feature_type"]
        self.noise = config["noise"]

        if self.obs_type == "feature":

            assert self.obs_dim >= self.world_dim, "Cannot handle observation dimension smaller than world dimension"
        elif self.obs_type == "image":

            assert self.world_dim == 2, "Can only generate images for 2D LQR problem"
            assert self.obs_dim[2] == 6, "Can only generate RGB images"
        else:
            raise AssertionError("Unhandled obs_type %r" % (self.obs_type,))

        self.d = 2 * self.world_dim                 # State is concatenation of position and velocity

        # self.position = 10 * np.random.randn(self.world_dim)
        self.position = np.random.randn(self.world_dim)
        self.velocity = np.random.randn(self.world_dim)

        self.curr_state = np.concatenate([self.position, self.velocity], axis=0)
        self.horizon = config["horizon"]
        self.timestep = 0

        # Block = numpy.block([[A11, A12], [A21, A22]])

        identity = np.eye(self.world_dim, dtype=np.float32)
        zero = np.zeros((self.world_dim, self.world_dim), dtype=np.float32)

        # s_{t + 1} = s_t + v_t * 1 sec + 0.5 a_t * 1 sec
        # v_{t + 1} = v_t + a_t * 1 sec
        self.A = np.block([[0.9 * identity, identity], [zero, 0.3 * identity]])     # Matrix of size self.state_dim x self.state_dim
        self.B = np.block([[0.5 * identity], [identity]])                           # Matrix of size self.state_dim x self.world_dim

        self.Q = np.eye(self.d, dtype=np.float32)
        self.R = config["acc_penalty"] * identity

        self.latent_lqr = LQRModel(
            A=self.A,
            B=self.B,
            Q=self.Q,
            R=self.R,
            Sigma_w=np.eye(self.d) * 10.0,
            Sigma_0=np.eye(self.d)
        )

        # self.fig, self.ax = plt.subplots()
        # plt.ion()

    def reset(self):

        self.position = 10 * np.random.randn(self.world_dim)
        # self.position = np.random.randn(self.world_dim)
        # self.velocity = np.random.randn(self.world_dim)
        # self.position = np.zeros(self.world_dim)
        # self.velocity = np.zeros(self.world_dim)
        self.timestep = 0

        self.curr_state = np.concatenate([self.position, self.velocity], axis=0)
        curr_obs = self.make_obs(self.curr_state)

        info = {"state": self.curr_state, "timestep": self.timestep}

        return curr_obs, info

    def get_reward(self, state, action):

        q_cost = np.dot(state, np.matmul(self.Q, state))
        r_cost = np.dot(action, np.matmul(self.R, action))

        # Reward is negative of cost
        return - q_cost - r_cost

    def make_obs(self, state):

        if self.obs_type == "feature":

            # Pad with Gaussian noise
            if self.obs_dim == self.d:

                return state
            elif self.obs_dim > self.d:

                noise = np.random.randn(self.obs_dim - self.d)
                return np.concatenate([state, noise], axis=0)
            else:
                raise AssertionError("Cannot handle observation dimension=%d smaller than state dimension=%d" %
                                     (self.obs_dim, self.d))

        elif self.obs_type == "image":

            height, width, _ = self.obs_dim
            half_grid_size = 20

            grid_height = height // (2 * half_grid_size)
            grid_width = width // (2 * half_grid_size)

            pos_image = np.zeros((3, height, width), dtype=np.float)
            vel_image = np.zeros((3, height, width), dtype=np.float)

            pos_x, pos_y, vel_x, vel_y = self.curr_state

            pos_x = min(half_grid_size, max(-half_grid_size, pos_x))
            pos_y = min(half_grid_size, max(-half_grid_size, pos_y))
            vel_x = min(half_grid_size, max(-half_grid_size, vel_x))
            vel_y = min(half_grid_size, max(-half_grid_size, vel_y))

            pt_pos_x = min(height - 1, int((pos_x + half_grid_size) * grid_height))
            pt_pos_y = min(width - 1, int((pos_y + half_grid_size) * grid_width))
            pt_vel_x = min(height - 1, int((vel_x + half_grid_size) * grid_height))
            pt_vel_y = min(width - 1, int((vel_y + half_grid_size) * grid_width))

            pos_noise = np.random.random((height, width)) * self.noise
            pos_image[0, :, :] = pos_noise
            pos_image[1, :, :] = pos_noise
            pos_image[2, :, :] = pos_noise
            pos_image[0, pt_pos_x, pt_pos_y] = 0.0
            pos_image[1, pt_pos_x, pt_pos_y] = 1.0

            vel_noise = np.random.random((height, width)) * self.noise
            vel_image[0, :, :] = vel_noise
            vel_image[1, :, :] = vel_noise
            vel_image[2, :, :] = vel_noise
            vel_image[0, pt_vel_x, pt_vel_y] = 0.0
            vel_image[1, pt_vel_x, pt_vel_y] = 1.0

            obs = np.concatenate([pos_image, vel_image], axis=0)

            # if self.timestep >= 10:
            #     exit(0)
            # else:
            #     import imageio, skimage
            #
            #     pos_image = pos_image.swapaxes(0, 1).swapaxes(1, 2)
            #     vel_image = vel_image.swapaxes(0, 1).swapaxes(1, 2)
            #     pos_image_rescaled = skimage.transform.resize(pos_image, (600, 600, 3))
            #     vel_image_rescaled = skimage.transform.resize(vel_image, (600, 600, 3))
            #
            #     imageio.imwrite("./temp/pos_%d.png" % self.timestep, pos_image_rescaled)
            #     imageio.imwrite("./temp/vel_%d.png" % self.timestep, vel_image_rescaled)

            return obs

        else:
            raise AssertionError("Unhandled type")

    def step(self, action):
        """ Given an action which is the acceleration output. Returns
            obs, reward, done, info
        """

        # self.visualize(self.curr_state, action)

        if self.timestep > 100 * self.horizon:
            # Cannot take any more actions
            raise AssertionError("Cannot take any more actions this episode. Horizon completed")

        else:

            new_state = np.matmul(self.A, self.curr_state) + np.matmul(self.B, action)
            noise = np.random.normal(0, 1.0, size=self.curr_state.shape)
            new_state = new_state + noise

            reward = self.get_reward(self.curr_state, action)

            self.curr_state = new_state
            curr_obs = self.make_obs(new_state)

            self.timestep += 1

            done = self.timestep == self.horizon

            info = {"state": self.curr_state, "timestep": self.timestep}

            return curr_obs, reward, done, info

    def get_latent_lqr(self):
        return self.latent_lqr

    def visualize(self, state, action):

        print("World num_factors ", self.world_dim)
        if self.world_dim != 2:
            return

        pixel = 0.01 * (state[0] + 25.0), 0.01 * (state[1] + 25.0)
        print("%r -> %r" % ((state[0],state[1]), pixel))

        circle = plt.Circle((pixel[0], pixel[1]), 0.02, color='r')
        # self.ax.add_patch(patches.Rectangle((0.1, 0.1), 0.5, 0.5, fill=False))
        self.ax.add_artist(circle)
        self.ax.arrow(pixel[0], pixel[1], 0.1 * action[0], 0.1 * action[1],
                      length_includes_head=True, head_width=0.05, head_length=0.05)
        # self.ax.arrow(0, 0, pixel[0], pixel[1], length_includes_head=True, head_width=0.1, head_length=0.2)
        plt.pause(0.2)
        plt.show()
        self.ax.cla()

    def get_model(self):

        return self.A, self.B, self.Q, self.R
