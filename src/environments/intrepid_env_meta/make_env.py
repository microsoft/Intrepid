import gym
import glob
import inspect
import importlib

from environments.gym_env.gym_wrapper import GymWrapper
from environments.minigrid.gridworld_wrapper import GridWorldWrapper
from environments.minigrid.gridworld_wrapper_iclr import GridWorldWrapperICLR
from environments.rl_acid_env.rl_acid_wrapper import RLAcidWrapper


class MakeEnvironment:
    """ " Wrapper class for generating environments using names and config"""

    def __init__(self):
        # Read name of environments and their mapping
        self.rl_acid_names = self._get_rl_acid_envs()

        self.openai_names = self._get_gym_envs()

        self.control_names = self._get_control_envs()

        self.minigrid_names = self._get_mingrid_envs()

        self.matterport_names = self._get_matterport_envs()

        self.ai2thor_names = self._get_ai2thor_envs()

        # Ensure that these environment names are all unique
        self.all_names = (
            list(self.rl_acid_names.keys())
            + self.openai_names
            + list(self.control_names.keys())
            + list(self.minigrid_names.keys())
            + list(self.matterport_names.keys())
        )

        uniq = set()
        counts = dict()

        for name in self.all_names:
            if name not in uniq:
                counts[name] = 0
            counts[name] += 1

        duplicates = {k for k, v in counts.items() if v > 1}

        if len(duplicates) > 0:
            raise AssertionError(
                "Environemt names %s, are duplicated across different runs. This confuses the code"
                "as to which environment is being asked to run. One way to fix this error can be"
                "to explicitly supply the type of environment that is being asked to run e.g.,"
                "minigrid/gridworld1 as opposes to gridworld1, which can be the name of an environment"
                "in another set of environments."
            )

    @staticmethod
    def _get_rl_acid_envs():
        env_names = dict()
        fnames = glob.glob("./src/environments/rl_acid_env/*py")

        for fname in fnames:
            fname = fname[len("./src/") : -len(".py")].replace("/", ".")
            module = importlib.import_module(fname)

            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and str(obj).startswith("<class '%s" % fname)
                    and issubclass(obj, RLAcidWrapper)
                ):
                    if hasattr(obj, "env_name"):
                        env_names[obj.env_name] = obj

        return env_names

    @staticmethod
    def _get_matterport_envs():
        env_names = dict()
        fnames = glob.glob("./src/environments/matterport/*py")

        for fname in fnames:
            fname = fname[len("./src/") : -len(".py")].replace("/", ".")
            module = importlib.import_module(fname)

            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and str(obj).startswith(
                    "<class '%s" % fname
                ):  # TODO add MatterportWrapper
                    if hasattr(obj, "env_name"):
                        env_names[obj.env_name] = obj

        return env_names

    @staticmethod
    def _get_gym_envs():
        all_envs = gym.envs.registry.all()
        return [env_spec.id for env_spec in all_envs]

    @staticmethod
    def _get_control_envs():
        env_names = dict()
        fnames = glob.glob("./src/environments/control_env/*py")

        for fname in fnames:
            fname = fname[len("./src/") : -len(".py")].replace("/", ".")
            module = importlib.import_module(fname)

            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and str(obj).startswith(
                    "<class '%s" % fname
                ):  # TODO add control wrapper
                    if hasattr(obj, "env_name"):
                        env_names[obj.env_name] = obj

        return env_names

    @staticmethod
    def _get_mingrid_envs():
        env_names = dict()
        fnames = glob.glob("./src/environments/minigrid/*py")

        for fname in fnames:
            fname = fname[len("./src/") : -len(".py")].replace("/", ".")
            module = importlib.import_module(fname)

            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and str(obj).startswith(
                    "<class '%s" % fname
                ):  # TODO add Minigrid wrapper
                    if hasattr(obj, "env_name"):
                        env_names[obj.env_name] = obj

        return env_names

    @staticmethod
    def _get_ai2thor_envs():
        env_names = dict()
        fnames = glob.glob("./src/environments/ai2thorenv/*py")

        for fname in fnames:
            fname = fname[len("./src/") : -len(".py")].replace("/", ".")
            module = importlib.import_module(fname)

            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and str(obj).startswith(
                    "<class '%s" % fname
                ):  # TODO add Minigrid wrapper
                    if hasattr(obj, "env_name"):
                        env_names[obj.env_name] = obj

        return env_names

    def make(self, exp_setup):
        env_name = exp_setup.env_name
        print(env_name)
        # raise Exception
        base_env_name = exp_setup.base_env_name

        if (
            env_name.startswith("rlacid/") and base_env_name in self.rl_acid_names
        ) or env_name in self.rl_acid_names:
            return self.rl_acid_names[base_env_name](exp_setup.config)

        elif (
            env_name.startswith("openai/") and base_env_name in self.openai_names
        ) or env_name in self.openai_names:
            return GymWrapper(base_env_name, exp_setup.config)

        elif (
            env_name.startswith("control/") and base_env_name in self.control_names
        ) or env_name in self.control_names:
            return self.control_names[base_env_name](exp_setup.config)

        elif (
            env_name.startswith("minigrid/") and base_env_name in self.minigrid_names
        ) or env_name in self.minigrid_names:
            base_env = self.minigrid_names[base_env_name](exp_setup.config)
            if base_env_name == "gridworld_iclr":
                return GridWorldWrapperICLR(base_env, exp_setup.config)
            else:
                return GridWorldWrapper(base_env, exp_setup.config, exp_setup.logger)

        elif (
            env_name.startswith("matterport/")
            and base_env_name in self.matterport_names
        ) or env_name in self.matterport_names:
            return self.matterport_names[base_env_name](exp_setup.config)

        elif (
            env_name.startswith("ai2thor/") and base_env_name in self.ai2thor_names
        ) or env_name in self.ai2thor_names:
            return self.ai2thor_names[base_env_name](exp_setup.config)

        else:
            raise AssertionError("Unhandled environment name %r" % env_name)
