import os
import json
import logging
import argparse
import statistics
import torch.multiprocessing as mp

from analysis_tools.visualize_latent_dynamics import VisualizeDynamics
from setup_validator.core_validator import validate
from utils.multiprocess_logger import MultiprocessingLoggerManager
from environments.cerebral_env_meta.environment_wrapper import GenerateEnvironmentWrapper


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='diabcombolock', help="name of the environment e.g., montezuma")
    parser.add_argument("--num_processes", default=6, type=int,
                        help="number of policy search (PS) processes to be launched at a given time")
    parser.add_argument("--name", default="run-mbrl-decoder", help="Name of the experiment")
    parser.add_argument("--horizon", default=-1, type=int, help="Horizon")
    parser.add_argument("--env_seed", default=None, type=int, help="Environment Seed")
    parser.add_argument("--noise", default=None, type=str, help="Noise")
    parser.add_argument("--save_trace", default="False", help="Save traces")
    parser.add_argument("--trace_sample_rate", default=500, type=int, help="How often to save traces")
    parser.add_argument("--save_path", default="./results/", type=str, help="Folder where to save results")
    parser.add_argument("--debug", default="False", help="Debug the run")
    parser.add_argument("--pushover", default="False", help="Use pushover to send results on phone")
    args = parser.parse_args()

    env_name = args.env
    num_processes = args.num_processes
    exp_name = args.name

    experiment_name = "%s-%s-horizon-%d-noise-%s" % (exp_name, env_name, args.horizon, args.noise)
    experiment = "%s/%s" % (args.save_path, experiment_name)
    print("EXPERIMENT NAME: ", experiment_name)

    # Create the experiment folder
    if not os.path.exists(experiment):
        os.makedirs(experiment)

    # Define log settings
    log_path = experiment + '/train_mbrl_oracle_decoder.log'
    multiprocess_logging_manager = MultiprocessingLoggerManager(
        file_path=log_path, logging_level=logging.INFO)
    master_logger = multiprocess_logging_manager.get_logger("Master")
    master_logger.log("----------------------------------------------------------------")
    master_logger.log("                    STARING NEW EXPERIMENT                      ")
    master_logger.log("----------------------------------------------------------------")
    master_logger.log("Environment Name %r. Experiment Name %r" % (env_name, exp_name))

    # Read configuration and constant files. Configuration contain environment information and
    # constant file contains hyperparameters for the model and learning algorithm.
    with open("data/%s/config.json" % env_name) as f:
        config = json.load(f)
        # Add command line arguments. Command line arguments supersede file settings.
        if args.horizon != -1:
            config["horizon"] = args.horizon
        if args.noise is not None:
            config["noise"] = args.noise

        config["save_trace"] = args.save_trace == "True"
        config["trace_sample_rate"] = args.trace_sample_rate
        config["save_path"] = experiment
        config["exp_name"] = experiment_name
        config["env_seed"] = args.env_seed

        GenerateEnvironmentWrapper.adapt_config_to_domain(env_name, config)

    with open("data/%s/constants.json" % env_name) as f:
        constants = json.load(f)

    print(json.dumps(config, indent=2))

    # Validate the keys
    validate(config, constants)

    # log core experiment details
    master_logger.log("CONFIG DETAILS")
    for k, v in sorted(config.items()):
        master_logger.log("    %s --- %r" % (k, v))
    master_logger.log("CONSTANTS DETAILS")
    for k, v in sorted(constants.items()):
        master_logger.log("    %s --- %r" % (k, v))
    master_logger.log("START SCRIPT CONTENTS")
    with open(__file__) as f:
        for line in f.readlines():
            master_logger.log(">>> " + line.strip())
    master_logger.log("END SCRIPT CONTENTS")

    performance = []
    num_runs = 1
    for trial in range(1, num_runs + 1):

        master_logger.log("========= STARTING EXPERIMENT %d ======== " % trial)

        # Create a new environment
        env = GenerateEnvironmentWrapper(env_name, config)
        master_logger.log("Environment Created")
        print("Created Environment...")

        # Save the environment for reproducibility
        env.save_environment(experiment, trial_name=trial)
        print("Saving Environment...")

        learning_alg = VisualizeDynamics(config, constants)

        policy_result = learning_alg.train(experiment=experiment,
                                           env=env,
                                           env_name=env_name,
                                           experiment_name=experiment_name,
                                           logger=master_logger,
                                           use_pushover=args.pushover == "True",
                                           debug=args.debug == "True",
                                           do_reward_sensitive_learning=True)

        performance.append(policy_result)

    for key in performance[0]:  # Assumes the keys are same across all runes
        results = [result[key] for result in performance]

        if len(results) <= 1:
            stdev = 0.0
        else:
            stdev = statistics.stdev(results)

        master_logger.log("%r: Mean %r, Median %r, Std %r, Num runs %r, All performance %r" %
                          (key, statistics.mean(results), statistics.median(results), stdev,
                           num_runs, results))

        print("%r: Mean %r, Median %r, Std %r, Num runs %r, All performance %r" %
              (key, statistics.mean(results), statistics.median(results), stdev, num_runs, results))

    # Cleanup
    multiprocess_logging_manager.cleanup()


if __name__ == "__main__":

    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn')
    main()
