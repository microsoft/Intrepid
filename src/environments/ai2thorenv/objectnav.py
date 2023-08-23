import random
import numpy as np

from environments.cerebral_env_meta.environment_keys import EnvKeys
from environments.cerebral_env_meta.cerebral_env_interface import CerebralEnvInterface


class ObjectNav(CerebralEnvInterface):
    MOVE_AHEAD = "MoveAhead"
    MOVE_BACK = "MoveBack"
    MOVE_LEFT = "MoveLeft"
    MOVE_RIGHT = "MoveRight"
    ROTATE_LEFT = "RotateLeft"
    ROTATE_RIGHT = "RotateRight"
    STOP = "Done"

    env_name = "objectnav"

    def __init__(self, config):
        from ai2thor.controller import Controller

        self.scene_name = config["scene_name"]
        self.headless = config["headless"] > 0
        self.horizon = config["horizon"]
        self.curr_event = None

        self.height, self.width, self.num_channels = config["obs_dim"]

        assert self.num_channels == 3, "Can only do RGB right now"

        self.enable_exo = config["enable_exo"] > 0

        # Brightness
        self.exo_brightness = 1.0
        self.exo_brightness_high = 3.0  # 1.5
        self.exo_brightness_low = 0.5

        self.exo_hue = 0.5
        self.exo_hue_high = 1.0
        self.exo_hue_low = 0.0

        self.exo_sat = 0.75
        self.exo_sat_high = 1.0
        self.exo_sat_low = 0.0

        if self.headless:
            from ai2thor.platform import CloudRendering

            self.controller = Controller(
                platform=CloudRendering,
                scene=self.scene_name,
                agentMode="default",
                visibilityDistance=1.5,
                # step sizes
                gridSize=0.25,
                # snapToGrid=True,          # Default setting from AI2Thor
                snapToGrid=False,  # Made false so that we can rotate by 45 degree
                rotateStepDegrees=45,  # Default angle is 90
                # image modalities
                renderDepthImage=False,
                renderInstanceSegmentation=False,
                # camera properties
                width=self.width,
                height=self.height,
                fieldOfView=90,
            )

        else:
            self.controller = Controller(
                scene=self.scene_name,
                agentMode="default",
                visibilityDistance=1.5,
                # step sizes
                gridSize=0.25,
                # snapToGrid=True,          # Default setting from AI2Thor
                snapToGrid=False,  # Made false so that we can rotate by 45 degree
                rotateStepDegrees=45,  # Default angle is 90
                # image modalities
                renderDepthImage=False,
                renderInstanceSegmentation=False,
                # camera properties
                width=self.width,
                height=self.height,
                fieldOfView=90,
            )

        self.act_to_name = {
            0: ObjectNav.MOVE_AHEAD,
            1: ObjectNav.ROTATE_LEFT,
            2: ObjectNav.ROTATE_RIGHT,
            3: ObjectNav.STOP,
        }

        self.num_actions = len(self.act_to_name)
        assert self.num_actions == config["num_actions"], (
            "Number of actions in config do not match the number of"
            "supported actions in AI2Thor Nav Bot."
        )

        self.timestep = 0
        self.goal_obj = "Book_4549024d"

    def reset(self):
        if self.enable_exo:
            self.exo_brightness = (
                random.random() * (self.exo_brightness_high - self.exo_brightness_low)
                + self.exo_brightness_low
            )
            self.exo_hue = (
                random.random() * (self.exo_hue_high - self.exo_hue_low)
                + self.exo_hue_low
            )
            self.exo_sat = (
                random.random() * (self.exo_sat_high - self.exo_sat_low)
                + self.exo_sat_low
            )

        self.timestep = 0
        new_event = self.controller.reset(scene=self.scene_name)

        if self.enable_exo:
            new_event = self.exo_update(old_event=new_event, respawn=True)

        self.curr_event = new_event
        obs = new_event.frame  # Height x Width x {RGB channel} as uint8
        obs = self._process_image(obs)

        info = self._get_info_from_event(new_event)

        return obs, info

    def step(self, action):
        if self.timestep >= self.horizon:
            raise AssertionError(
                "Cannot take more actions as H many actions have been already taken in this episode"
                "where H is the horizon. Resetting the episode, or increasing the horizon may solve"
                "this problem."
            )

        if action in self.act_to_name:
            new_event = self.controller.step(self.act_to_name[action])

        else:
            raise AssertionError(
                "Action can take values only in {0, ..., %d}" % (self.num_actions - 1)
            )

        if self.enable_exo:
            new_event = self.exo_update(new_event)

        obs = new_event.frame  # Height x Width x {RGB channel} as uint8
        obs = self._process_image(obs)

        self.timestep += 1
        done = self.timestep == self.horizon
        reward = 0  # TODO: define reward somehow
        self.curr_event = new_event
        info = self._get_info_from_event(new_event)

        return obs, reward, done, info

    def _process_image(self, obs):
        return (obs / 255.0).astype(np.float32)

    def get_objects(self, event):
        obj_pos = dict()
        for obj in event.metadata["objects"]:
            obj_pos[obj["name"]] = np.array(
                [obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]]
            )
        return obj_pos

    def exo_update(self, old_event, respawn=False):
        if respawn:
            new_event = self.controller.step(
                action="InitialRandomSpawn",
                randomSeed=random.randint(0, int(1e6)),  # random.randint(0, int(1e9))
                forceVisible=False,
                numPlacementAttempts=5,
                placeStationary=True,
                numDuplicatesOfType=[],
                excludedReceptacles=[],
                excludedObjectIds=[],
            )

        # Randomize Materials (Not Exogenous)
        if random.random() < 0.2:
            new_event = self.controller.step(
                action="RandomizeMaterials",
                useTrainMaterials=None,
                useValMaterials=None,
                useTestMaterials=None,
                inRoomTypes=None,
            )

        self.exo_brightness = self.update_val(
            self.exo_brightness, 0.25, self.exo_brightness_high, self.exo_brightness_low
        )
        self.exo_hue = self.update_val(
            self.exo_hue, 0.15, self.exo_hue_high, self.exo_hue_low
        )
        self.exo_sat = self.update_val(
            self.exo_sat, 0.15, self.exo_sat_high, self.exo_sat_low
        )

        # Randomize Lighting
        new_event = self.controller.step(
            action="RandomizeLighting",
            brightness=(self.exo_brightness, self.exo_brightness),
            randomizeColor=True,
            hue=(self.exo_hue, self.exo_hue),
            saturation=(self.exo_sat, self.exo_sat),
            synchronized=False,
        )

        # Randomize Colors (Not Exogenous)
        if random.random() < 0.2:
            new_event = self.controller.step(action="RandomizeColors")

        return new_event

    @staticmethod
    def update_val(current, step, high_val, low_val):
        if random.random() < 0.5:
            val = current - step
        else:
            val = current + step

        val = min(max(val, low_val), high_val)

        return val

    @staticmethod
    def _get_info_from_event(event):
        agent_state = (
            event.metadata["agent"]["position"]["x"],  # x position of the agent
            event.metadata["agent"]["position"]["y"],  # y position of the agent
            event.metadata["agent"]["position"]["z"],  # z position of the agent
            event.metadata["agent"]["rotation"]["x"],  # x rotation of the agent
            event.metadata["agent"]["rotation"]["y"],  # y rotation of the agent
            event.metadata["agent"]["rotation"]["z"],  # z rotation of the agent
            event.metadata["agent"][
                "cameraHorizon"
            ],  # The angle in degrees that the camera's pitch is rotated
        )

        info = {EnvKeys.STATE: agent_state, EnvKeys.ENDO_STATE: agent_state}

        return info

    def is_episodic(self):
        return True

    def act_to_str(self, action):
        return self.act_to_name[action]
