import numpy as np

from environments.ai2thorenv.ai2thor_exo_util import AI2ThorExoUtil
from environments.cerebral_env_meta.cerebral_env_interface import CerebralEnvInterface
from environments.cerebral_env_meta.environment_keys import EnvKeys


class NavAI2Thor(CerebralEnvInterface):
    MOVE_AHEAD = "MoveAhead"
    MOVE_BACK = "MoveBack"
    MOVE_LEFT = "MoveLeft"
    MOVE_RIGHT = "MoveRight"
    ROTATE_LEFT = "RotateLeft"
    ROTATE_RIGHT = "RotateRight"
    STOP = "Done"

    env_name = "ai2thornav"

    def __init__(self, config):
        from ai2thor.controller import Controller

        self.scene_name = config["scene_name"]
        self.headless = config["headless"] > 0
        self.horizon = config["horizon"]
        self.curr_event = None

        self.enable_exo = config["enable_exo"] > 0

        if self.enable_exo:
            self.exo_util = AI2ThorExoUtil(config)

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
                width=56,  # 120
                height=56,  # 120
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
                width=56,  # 120
                height=56,  # 120
                fieldOfView=90,
            )

        self.act_to_name = {
            0: NavAI2Thor.MOVE_AHEAD,
            1: NavAI2Thor.ROTATE_LEFT,
            2: NavAI2Thor.ROTATE_RIGHT,
            3: NavAI2Thor.STOP,
        }

        self.num_actions = len(self.act_to_name)
        assert self.num_actions == config["num_actions"], (
            "Number of actions in config do not match the number of"
            "supported actions in AI2Thor Nav Bot."
        )

        self.timestep = 0

    def reset(self):
        if self.enable_exo:
            self.exo_util.reset()

        self.timestep = 0
        new_event = self.controller.reset(scene=self.scene_name)
        self.curr_event = new_event

        obs = new_event.frame  # Height x Width x {RGB channel} as uint8
        obs = self._process_image(obs)

        info = self._get_info_from_event(new_event)

        return obs, info

    def step(self, action):
        if self.enable_exo:
            self.exo_util.update()

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

        obs = new_event.frame  # Height x Width x {RGB channel} as uint8
        obs = self._process_image(obs)

        self.timestep += 1
        done = self.timestep == self.horizon
        reward = 0  # TODO: define reward somehow
        self.curr_event = new_event
        info = self._get_info_from_event(new_event)

        return obs, reward, done, info

    def _process_image(self, obs):
        if self.enable_exo:
            obs = self.exo_util.generate(obs)
            return obs.astype(np.float32)
        else:
            return (obs / 255.0).astype(np.float32)

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
