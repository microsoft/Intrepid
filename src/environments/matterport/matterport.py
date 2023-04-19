import math
import glob
import random
import imageio
import numpy as np

from skimage.transform import resize

from environments.cerebral_env_meta.action_type import ActionType
from environments.cerebral_env_meta.cerebral_env_interface import CerebralEnvInterface


class Matterport(CerebralEnvInterface):

    env_name = "matterport"

    def __init__(self, config):

        import MatterSim

        self.sim = MatterSim.Simulator()
        self.sim.setCameraResolution(config["width"], config["height"])
        self.sim.setCameraVFOV(math.radians(config["vfov"]))
        self.sim.setDepthEnabled(False)  # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
        self.sim.setPreloadingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setBatchSize(1)
        self.sim.setCacheSize(2)

        # Paths
        self.sim.setDatasetPath(config["dataset"])
        self.sim.setNavGraphPath(config["connectivity"])

        # house and room information
        self.obs_dim = tuple(config["obs_dim"])
        self.house_id = '17DRP5sb8fy'
        self.room_id = '0f37bd0737e349de9d536263a4bdd60d'

        self.horizon = config["horizon"]
        self.timestep = 0
        self.num_eps = 0
        self.sum_return = 0

        # Distractors
        self.use_exo = config["use_exo"]

        if self.use_exo:
            self.distractors = self._read_distractors()
            self.distractor_hor_range = [0.0, int(0.75 * self.obs_dim[1])]
            self.distractor_ver_range = [0.0, int(0.75 * self.obs_dim[0])]

        self.distractor_id = None
        self.distractor_hor = None
        self.distractor_ver = None

        self.sim.initialize()

    def _read_distractors(self):

        fnames = glob.glob("./data/matterport/icon_figs/*png")
        distractors = []

        assert self.obs_dim[0] >= 5 and self.obs_dim[1] >= 5, "Image has to be be at least 5x5 (hxw) large"

        for fname in fnames:
            distractor_img = imageio.imread(fname)

            assert len(distractor_img.shape) == 3 and (3 <= distractor_img.shape[2] <= 4), \
                "Can only read RGB and RGBA images"

            if distractor_img.shape[2] == 4:
                distractor_img = distractor_img[:, :, :3]

            # Resize based on original image so that width of the obstacle is 10% of the width and
            # height is at most 40% of the height
            distractor_img = resize(distractor_img, (min(distractor_img.shape[0], int(0.2 * self.obs_dim[0])),
                                                     min(distractor_img.shape[1], int(0.2 * self.obs_dim[1])),
                                                     3))
            # distractor_img = (distractor_img * 255).astype(np.uint8)
            distractors.append(distractor_img)

        return distractors

    def distractor_move(self):

        # Add random motion to one of the neigbhoring position
        hor_s = [self.distractor_hor]
        left = self.distractor_hor - int(0.1 * self.obs_dim[1])
        if self.distractor_hor_range[0] <= left <= self.distractor_hor_range[1]:
            hor_s.append(left)

        right = self.distractor_hor + int(0.1 * self.obs_dim[1])
        if self.distractor_hor_range[0] <= right <= self.distractor_hor_range[1]:
            hor_s.append(right)

        ver_s = [self.distractor_ver]
        top = self.distractor_ver - int(0.1 * self.obs_dim[0])
        if self.distractor_ver_range[0] <= top <= self.distractor_ver_range[1]:
            ver_s.append(top)

        bottom = self.distractor_ver + int(0.1 * self.obs_dim[0])
        if self.distractor_ver_range[0] <= bottom <= self.distractor_ver_range[1]:
            ver_s.append(bottom)

        assert self.distractor_hor_range[1] - self.distractor_hor_range[0] >= 2
        assert self.distractor_ver_range[1] - self.distractor_ver_range[0] >= 2

        self.distractor_hor = random.choice(hor_s)
        self.distractor_ver = random.choice(ver_s)

    def _process_image(self, img):

        height, width, channel = img.shape
        assert height == 480 and width == 640 and channel == 3, \
            "Wrong shape. Found %r. Expected 480 x 640 x 3" % img.shape
        img = resize(img, self.obs_dim)
        img = np.ascontiguousarray(img)

        if self.use_exo:
            # Add distractor
            distractor_img = self.distractors[self.distractor_id]
            distractor_shape = distractor_img.shape

            img_slice = img[self.distractor_ver: self.distractor_ver + distractor_shape[0],
                            self.distractor_hor: self.distractor_hor + distractor_shape[1], :]

            distractor_img = distractor_img.reshape((-1, 3))
            img_slice = img_slice.reshape((-1, 3))
            distractor_img_min = distractor_img.min(1)
            blue_pixel_ix = np.argwhere(
                distractor_img_min < 220 / 255.0)  # flattened (x, y) position where pixels are blue in color
            values = np.squeeze(distractor_img[blue_pixel_ix])
            np.put_along_axis(img_slice, blue_pixel_ix, values, axis=0)

            img_slice = img_slice.reshape(distractor_shape)     # distractor and img_slice have the same shape

            img[self.distractor_ver: self.distractor_ver + distractor_shape[0],
                self.distractor_hor: self.distractor_hor + distractor_shape[1], :] = img_slice

        return img

    def reset(self, generate_obs=True):

        self.sim.newEpisode([self.house_id], [self.room_id], [0], [0])
        self.num_eps += 1

        # Sample a distractor
        if self.use_exo:
            self.distractor_id = random.randint(0, len(self.distractors) - 1)
            # Horizon pixel location
            self.distractor_hor = random.randint(self.distractor_hor_range[0], self.distractor_hor_range[1] - 1)
            # Horizon pixel location
            self.distractor_ver = random.randint(self.distractor_ver_range[0], self.distractor_ver_range[1] - 1)

        state = self.sim.getState()[0]
        self.timestep = 0

        if generate_obs:
            img = np.array(state.rgb, copy=False)
            img = self._process_image(img)
        else:
            img = None

        # return img, 0, False, {"state": img, "location": state.location.viewpointId}
        return img, {
            "state": (state.location.viewpointId, int(math.degrees(state.heading)), self.timestep),
            "endogenous_state": (state.location.viewpointId, int(math.degrees(state.heading)), self.timestep),
            "timestep": self.timestep
        }

    def step(self, action, generate_obs=True):
        # self.sim.makeAction([location], [heading], [elevation])

        if self.timestep >= self.horizon:
            raise AssertionError("Cannot take more actions than %d" % self.horizon)

        # Simulator can be queried in batch
        state = self.sim.getState()[0]

        if action == 0:
            # Stay in place
            pass

        elif action == 1:
            # Go straight if possible, otherwise stay in place
            if len(state.navigableLocations) > 1:
                self.sim.makeAction([1], [0], [0])

        elif action == 2:
            # Look 30 degrees left
            self.sim.makeAction([0], [-0.523599], [0])

        elif action == 3:
            # Look 30 degrees right
            self.sim.makeAction([0], [0.523599], [0])

        elif action == 4:
            # Look 30 degrees down
            self.sim.makeAction([0], [0], [-0.523599])

        elif action == 5:
            # Look 30 degrees up
            self.sim.makeAction([0], [0], [0.523599])

        # Move distractor
        if self.use_exo:
            self.distractor_move()

        state = self.sim.getState()[0]
        if generate_obs:
            img = np.array(state.rgb, copy=False)
            img = self._process_image(img)
        else:
            img = None

        info = {
            "state": (state.location.viewpointId, int(math.degrees(state.heading)), self.timestep),
            "endogenous_state": (state.location.viewpointId, int(math.degrees(state.heading)), self.timestep),
            "timestep": self.timestep
        }
        self.timestep += 1

        reward = 0
        self.sum_return += reward

        return img, reward, False, info

    def is_episodic(self):
        return True

    def act_to_str(self, act):

        if act == 0:
            return "no-op"
        elif act == 1:
            return "forward"
        elif act == 2:
            return "turn-left"
        elif act == 3:
            return "turn-right"
        elif act == 4:
            return "look-down"
        elif act == 5:
            return "look-up"
        else:
            raise AssertionError("Action has to be in {0, 1, 2, 3, 4, 5}. Found %r" % act)

    def get_action_type(self):
        """
            :return:
                action_type:     Return type of action space the agent is using
        """
        return ActionType.Discrete

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

        return 0.0
