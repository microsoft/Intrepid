import os
import json
import time
import logging
import argparse

from setup_validator.core_validator import validate
from experiments.experimental_setup import ExperimentalSetup
from utils.multiprocess_logger import MultiprocessingLoggerManager


def get_header():

    # Read base config
    with open("data/base_config.json") as f:
        config = json.load(f)

    # Read base constants
    with open("data/base_constants.json") as f:
        constants = json.load(f)

    shared_vars = set(config.keys()).intersection(set(constants.keys()))
    assert len(shared_vars) == 0, \
        "Base config and constant file share following parameters %r. " \
        "This causes confusion in overwriting them with command line arguments. " % shared_vars

    parser = argparse.ArgumentParser()

    for k, v in config.items():
        parser.add_argument("--%s" % k, default=None, type=type(v), help="")

    for k, v in constants.items():
        parser.add_argument("--%s" % k, default=None, type=type(v), help="")

    # Algorithm and Environment Agnostic Base arguments
    parser.add_argument("--env", default='temporal_diabcombolock', help="name of the environment e.g., montezuma")
    parser.add_argument("--name", default="run-exp", help="Name of the experiment")
    parser.add_argument("--seed", default=1234, type=int, help="Random Generator Seed")
    parser.add_argument("--save_trace", default="False", help="Save traces")
    parser.add_argument("--saved_models", default="./saved_models", help="Models saved from previous run")
    parser.add_argument("--trace_sample_rate", default=500, type=int, help="How often to save traces")
    parser.add_argument("--save_path", default="./results/", type=str, help="Folder where to save results")
    parser.add_argument("--debug", default=-1, type=int, help="Debug the run")

    args = parser.parse_args()

    args.debug = args.debug > 0
    env_name = args.env

    if os.path.exists("%s/%s" % (args.save_path, args.name)):
        exp_name = args.name
    else:
        exp_name = "%s-%d" % (args.name, int(time.time()))

    experiment = "%s/%s" % (args.save_path, exp_name)

    # Create the experiment folder
    if not os.path.exists(experiment):
        os.makedirs(experiment)

    # Define log settings
    log_path = experiment + '/train.log'
    multiprocess_logging_manager = MultiprocessingLoggerManager(file_path=log_path, logging_level=logging.INFO)
    main_logger = multiprocess_logging_manager.get_logger("Main")
    main_logger.log("----------------------------------------------------------------")
    main_logger.log("                    STARING NEW EXPERIMENT                      ")
    main_logger.log("----------------------------------------------------------------")
    main_logger.log("Environment Name %r. Experiment Name %r" % (env_name, exp_name))

    # Update configuration and constant. The values have priority in the following order.
    # - command line argument values take first precedence
    # - environment file values take second precedence
    # - base values will take last precedence

    # Read configuration and constant files. Configuration contain environment information and
    # constant file contains hyperparameters for the model and learning algorithm.
    with open("data/%s/config.json" % env_name) as f:
        env_config = json.load(f)

    for k, v in env_config.items():
        config[k] = v       # Overwrite base_config values with environment specific config values

    with open("data/%s/constants.json" % env_name) as f:
        env_constants = json.load(f)

    for k, v in env_constants.items():
        constants[k] = v   # Overwrite base_constant values with environment specific constants

    # Lastly, if command line arguments are not none then they take top most priority
    for k, v in vars(args).items():

        if k in config and v is not None:
            assert type(v) == type(config[k])
            config[k] = v

        if k in constants and v is not None:
            assert type(v) == type(constants[k])
            constants[k] = v

    # TODO find a place to store seed and save_path
    config["seed"] = args.seed
    config["save_path"] = args.save_path

    # Validate the keys
    validate(config, constants)

    # log core experiment details
    main_logger.log("Config Values")
    for k, v in sorted(config.items()):
        main_logger.log("    %s --- %r" % (k, v))

    main_logger.log("Constant Values")
    for k, v in sorted(constants.items()):
        main_logger.log("    %s --- %r" % (k, v))

    main_logger.log("Start Script Contents")
    with open(__file__) as f:
        for line in f.readlines():
            main_logger.log(">>> " + line.strip())
    main_logger.log("End Script Contents")

    return ExperimentalSetup(config=config,
                             constants=constants,
                             experiment=experiment,
                             exp_name=exp_name,
                             env_name=env_name,
                             args=args,
                             debug=args.debug,
                             logger=main_logger,
                             logger_manager=multiprocess_logging_manager)
