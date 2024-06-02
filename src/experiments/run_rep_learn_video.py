import torch
import random
import numpy as np
import torch.multiprocessing as mp

from experiments.experiment_save import terminate
from experiments.experiment_header import get_header
from environments.intrepid_env_meta.make_env import MakeEnvironment
from learning.core_learner.prediction_video import AutoencoderVideo, NextFrameVideo
from learning.core_learner.temporal_contrastive_video import TemporalContrastiveVideo


def main():

    exp_setup = get_header()

    performance = []

    if exp_setup.config["seed"] == -1:
        seeds = list(range(1234, 1234 + 10))
        num_runs = len(seeds)
    else:
        seeds = [exp_setup.config["seed"]]
        num_runs = 1

    for exp_id in range(1, num_runs + 1):

        exp_setup.config["seed"] = seeds[exp_id - 1]
        exp_setup.config["env_seed"] = seeds[exp_id - 1] * 10
        exp_setup.logger.log(
            "========= STARTING EXPERIMENT %d (Seed = %d) ======== " % (exp_id, exp_setup.config["seed"]))

        # Set the random seed
        random.seed(exp_setup.config["seed"])
        np.random.seed(exp_setup.config["seed"])
        torch.manual_seed(exp_setup.config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(exp_setup.config["seed"])

        # Create a new environment
        make_env = MakeEnvironment()
        env = make_env.make(exp_setup)
        exp_setup.logger.log("Environment Created")

        # Save the environment for reproducibility
        # env.save_environment(experiment, trial_name=exp_id)
        # print("Saving Environment...")

        methods = [AutoencoderVideo, NextFrameVideo, TemporalContrastiveVideo]
        available_names = [method.NAME for method in methods]

        learning_alg = None
        for method in methods:

            if exp_setup.constants["rep_alg"] == method.NAME:
                learning_alg = method(exp_setup=exp_setup)
                exp_setup.logger.log("Created Learning Algorithm %s of type %r" % (method.NAME, method))
                break

        if learning_alg is None:
            raise AssertionError("Unhandled representation learning algorithm %s. Available names %r" %
                                 (exp_setup.constants["rep_alg"], available_names))
        else:
            # uses [64, 4, 84, 84] as input dim for pong
            policy_result = learning_alg.do_train(env=env)

        performance.append(policy_result)

    terminate(performance, exp_setup, seeds)


if __name__ == "__main__":

    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn')
    main()
