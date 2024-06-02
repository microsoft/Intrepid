import time
import numpy as np

from environments.cerebral_env_meta.environment_wrapper import GenerateEnvironmentWrapper
from learning.learning_utils.encoder_sampler_wrapper import EncoderSamplerWrapper
from learning.learning_utils.homer_train_encoding_function import TrainEncodingFunction
from learning.learning_utils.rl_discrete_latent_state_util import RLDiscreteLatentStateUtil
from learning.datastructures.transition import TransitionDatapoint
from model.policy.stationary_action_condition_policy import StationaryActionConditionPolicy
from utils.tensorboard import Tensorboard
from functools import partial


class DebugTrainEncodingFunction:
    """ A class for debugging the training of encoding function using oracle homing policies """

    def __init__(self, config, constants):

        self.config = config
        self.constants = constants

        # Train encoding function
        self.train_encoding_function = TrainEncodingFunction(config, constants)

        # Sampler for generating data for training the encoding function
        self.encoder_sampler = EncoderSamplerWrapper(constants)

        # Util
        self.util = RLDiscreteLatentStateUtil(config, constants)

    def generate_gold_homing_policies(self, env, env_name, horizon):

        if env_name == "combolock":
            return DebugTrainEncodingFunction.generate_combolock_gold_homing_policies(env, horizon)
        elif env_name == "stochcombolock":
            return DebugTrainEncodingFunction.generate_stochcombolock_gold_homing_policies(env, horizon)
        elif env_name == "diabcombolock":
            return DebugTrainEncodingFunction.generate_diabcombolock_gold_homing_policies(env, horizon)
        else:
            raise AssertionError("Unhandled environment name %r" % env_name)

    @staticmethod
    def generate_combolock_gold_homing_policies(env, horizon):

        homing_policy = dict()

        for step in range(1, horizon + 1):

            # Create policy to reach live branch
            live_state_policy = []
            for step_ in range(1, step + 1):

                def act_to_live(obs, mystep, myenv):
                    if obs[0][2 * (mystep - 1) + 0] == 1.0:  # In State (0, step - 1) which is dead
                        return 0
                    elif obs[0][2 * (mystep - 1) + 1] == 1.0:  # In State (1, step - 1) which is live
                        return myenv.env.opt[mystep - 1]
                    else:
                        raise AssertionError("Cannot be in any other state. Obs: %r, step: %r" % (obs, mystep))

                action_condition = partial(act_to_live, mystep=step_, myenv=env)
                policy_ = StationaryActionConditionPolicy(action_condition)
                live_state_policy.append(policy_)

            # Create policy to reach dead branch
            dead_state_policy = []
            for step_ in range(1, step + 1):

                def act_to_die(obs, mystep, myenv):
                    if obs[0][2 * (mystep - 1) + 0] == 1.0:  # In State (0, step - 1) which is dead
                        return 0
                    elif obs[0][2 * (mystep - 1) + 1] == 1.0:  # In State (1, step - 1) which is live
                        return 1 - myenv.env.opt[mystep - 1]
                    else:
                        raise AssertionError("Cannot be in any other state. Obs: %r, step: %r" % (obs, mystep))

                action_condition = partial(act_to_die, mystep=step_, myenv=env)
                policy_ = StationaryActionConditionPolicy(action_condition)
                dead_state_policy.append(policy_)

            homing_policy[step] = [live_state_policy, dead_state_policy]

        return homing_policy

    @staticmethod
    def generate_stochcombolock_gold_homing_policies(env, horizon):

        homing_policy = dict()

        for step in range(1, horizon + 1):

            # Create policy to reach live branch
            live_state_policy = dict()
            for step_ in range(1, step + 1):

                def act_to_live(obs, mystep, myenv):
                    if obs[0][3 * (mystep - 1) + 0] == 1.0:  # In State (0, step - 1) which is live
                        return myenv.env.opt_a[mystep - 1]
                    elif obs[0][3 * (mystep - 1) + 1] == 1.0:  # In State (1, step - 1) which is live
                        return myenv.env.opt_b[mystep - 1]
                    elif obs[0][3 * (mystep - 1) + 2] == 1.0:  # In State (2, step - 1) which is dead
                        return 0
                    else:
                        raise AssertionError("Cannot be in any other state. Obs: %r, step: %r" % (obs, mystep))

                action_condition = partial(act_to_live, mystep=step_, myenv=env)
                policy_ = StationaryActionConditionPolicy(action_condition)
                live_state_policy[step_] = policy_

            # Create policy to reach dead branch
            dead_state_policy = dict()
            for step_ in range(1, step + 1):

                def act_to_die(obs, mystep, myenv):
                    if obs[0][3 * (mystep - 1) + 0] == 1.0:  # In State (0, step - 1) which is live
                        return 1 - myenv.env.opt_a[mystep - 1]
                    elif obs[0][3 * (mystep - 1) + 1] == 1.0:  # In State (1, step - 1) which is live
                        return 1 - myenv.env.opt_b[mystep - 1]
                    elif obs[0][3 * (mystep - 1) + 2] == 1.0:  # In State (2, step - 1) which is dead
                        return 0
                    else:
                        raise AssertionError("Cannot be in any other state. Obs: %r, step: %r" % (obs, mystep))

                action_condition = partial(act_to_die, mystep=step_, myenv=env)
                policy_ = StationaryActionConditionPolicy(action_condition)
                dead_state_policy[step_] = policy_

            homing_policy[step] = [live_state_policy, dead_state_policy]

        return homing_policy

    @staticmethod
    def generate_diabcombolock_gold_homing_policies(env, horizon):

        homing_policy = dict()

        for step in range(1, horizon + 1):

            # Create policy to reach live branch
            live_state_policy = dict()
            for step_ in range(1, step + 1):

                def act_to_live(obs, mystep, myenv):
                    if obs[0][3 * (mystep - 1) + 0] == 1.0:  # In State (0, step - 1) which is live
                        return myenv.env.opt_a[mystep - 1]
                    elif obs[0][3 * (mystep - 1) + 1] == 1.0:  # In State (1, step - 1) which is live
                        return myenv.env.opt_b[mystep - 1]
                    elif obs[0][3 * (mystep - 1) + 2] == 1.0:  # In State (2, step - 1) which is dead
                        return 0
                    else:
                        raise AssertionError("Cannot be in any other state. Obs: %r, step: %r" % (obs, mystep))

                action_condition = partial(act_to_live, mystep=step_, myenv=env)
                policy_ = StationaryActionConditionPolicy(action_condition)
                live_state_policy[step_] = policy_

            # Create policy to reach dead branch
            dead_state_policy = dict()
            for step_ in range(1, step + 1):

                def act_to_die(obs, mystep, myenv):
                    if obs[0][3 * (mystep - 1) + 0] == 1.0:  # In State (0, step - 1) which is live
                        return (myenv.env.opt_a[mystep - 1] + 1) % myenv.env.num_actions
                    elif obs[0][3 * (mystep - 1) + 1] == 1.0:  # In State (1, step - 1) which is live
                        return (myenv.env.opt_b[mystep - 1] + 1) % myenv.env.num_actions
                    elif obs[0][3 * (mystep - 1) + 2] == 1.0:  # In State (2, step - 1) which is dead
                        return 0
                    else:
                        raise AssertionError("Cannot be in any other state. Obs: %r, step: %r" % (obs, mystep))

                action_condition = partial(act_to_die, mystep=step_, myenv=env)
                policy_ = StationaryActionConditionPolicy(action_condition)
                dead_state_policy[step_] = policy_

            homing_policy[step] = [live_state_policy, dead_state_policy]

        return homing_policy

    def _purge_observation(self, env_name, obs):
        """ Purge observation for combolocks """

        vec = np.copy(obs)
        horizon = self.config["horizon"]

        if env_name == "combolock":
            vec[2 * horizon + 2:] = vec[2 * horizon + 2:] * 0.0
        elif env_name == "stochcombolock" or env_name == "diabcombolock":
            vec[3 * horizon + 3:] = vec[3 * horizon + 3:] * 0.0
        else:
            raise AssertionError("Cannot handle")

        return vec

    def _purge_noise(self, env_name, dataset, purge_type="curr", logger=None):
        """
        Often times the observation has noise that can be purged for the purpose of ablation. Dataset is purged in place.
        :param env_name: Name of the environment
        :param dataset: Dataset to be purged of noise.
        :param purge_type: Is either "curr", "next" or "both". Indicating if only previous observation should be
        purged or current observation or both.
        :return nothing is returned.
        """

        assert env_name == "combolock" or env_name == "stochcombolock" or env_name == "diabcombolock", \
            "Only combolocks are supported"
        assert purge_type == "curr" or purge_type == "next" or purge_type == "both", "Only supported types"

        for datapoint in dataset:

            assert isinstance(datapoint, TransitionDatapoint), "Must be of type Transition Datapoint"

            if purge_type == "curr" or purge_type == "both":
                datapoint.curr_obs = self._purge_observation(env_name, datapoint.curr_obs)

            if purge_type == "next" or purge_type == "both":
                datapoint.next_obs = self._purge_observation(env_name, datapoint.next_obs)

        if logger is not None:
            logger.log("Purged the dataset with type %r" % purge_type)

    @staticmethod
    def do_train(config, constants, env_name, experiment_name, logger, use_pushover, debug):

        # Create the environment
        env = GenerateEnvironmentWrapper(env_name, config)
        logger.log("Environment Created")

        if env_name == "stochcombolock" or env_name == "diabcombolock":
            logger.log("Created Environment. First 5 actions Opt-A %r and Opt-B %r" %
                       (env.env.opt_a[0:5], env.env.opt_b[0:5]))
            print("Created Environment. First 5 actions Opt-A %r and Opt-B %r" %
                       (env.env.opt_a[0:5], env.env.opt_b[0:5]))
        elif env_name == "combolock":
            logger.log("Created Environment. First 5 actions %r" % env.env.opt[0:5])
            print("Created Environment. First 5 actions %r" % env.env.opt[0:5])
        else:
            raise AssertionError("Unhandled environment %s" % env_name)

        learner = DebugTrainEncodingFunction(config, constants)
        learner.train(env, env_name, experiment_name, logger, use_pushover, debug)

    def train(self, env, env_name, experiment_name, logger, use_pushover, debug):
        """ Performs the learning """

        horizon = self.config["horizon"]
        actions = self.config["actions"]
        num_samples = self.constants["encoder_training_num_samples"]

        tensorboard = Tensorboard(log_dir=self.config["save_path"])

        gold_homing_policies = self.generate_gold_homing_policies(env, env_name, horizon)
        encoding_function = None    # Learned encoding function for the current time step
        dataset = []                # Dataset of samples collected for training the encoder
        selection_weights = None    # A distribution over homing policies from the previous time step (can be None)
        observation_samples = None  # A set of observations observed on exploration
        success = True

        if env_name == "stochcombolock" or env_name == "diabcombolock":
            action_match = env.env.opt_a[1] == env.env.opt_b[1]
        elif env_name == "combolock":
            action_match = True  # There is nothing to match so true
        else:
            raise AssertionError("Unhandled environment %s" % env_name)

        assert horizon >= 2

        for step in range(2, 3):

            logger.log("Step %r out of %r " % (step, horizon))

            # Step 1: Create dataset for learning the encoding function. A single datapoint consists of a transition
            # (x, a, x') and a 0-1 label y. If y=1 then transition was observed and y=0 otherwise.
            time_collection_start = time.time()
            dataset = self.encoder_sampler.gather_samples(env, actions, step, gold_homing_policies, num_samples,
                                                          dataset, selection_weights)
            logger.log("Encoder: %r sample collected in %r sec" % (num_samples, time.time() - time_collection_start))

            # Optionally purge the dataset for ablation.
            # self._purge_noise(env_name, dataset, "curr", logger)

            # Step 2: Train a binary classifier on this dataset. The classifier f(x, a, x') is trained to predict
            # the probability that the transition (x, a, x') was observed. Importantly, the classifier has a special
            # structure f(x, a, x') = p(x, a, \phi(x')) where \phi maps x' to a set of discrete values.
            time_encoder_start = time.time()
            if not self.constants["bootstrap_encoder_model"]:
                encoding_function = None

            if self.config["encoder_training_type"] == "transfer":
                encoding_function, result_dict = self.train_encoding_function.do_train_with_discretized_models(
                    dataset, logger, tensorboard, debug, bootstrap_model=encoding_function)
            else:
                raise AssertionError("Unhandled training %r" % self.config["encoder_training_type"])

            logger.log("Encoder: Training time %r" % (time.time() - time_encoder_start))

            if debug:
                if self.config["feature_type"] == "image":

                    # Save the abstract state and an image
                    if observation_samples is not None:
                        self.util.save_abstract_state_figures(observation_samples, step)

                    # Save newly explored states
                    self.util.save_newly_explored_states(dataset, step)

            success = success and result_dict["success"]

            logger.log("Result: %r, Actions Match: %r" % (result_dict, action_match))
            print("Result: %r, Actions Match: %r" % (result_dict, action_match))

        return {"success": 1.0 if success else 0.0}
