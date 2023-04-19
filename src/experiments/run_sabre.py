import torch
import pickle
import random
import numpy as np
import torch.multiprocessing as mp

from experiments.experiment_save import terminate
from experiments.experiment_header import get_header
from environments.cerebral_env_meta.make_env import MakeEnvironment
from learning.core_learner.sabre import Sabre


def main():

    exp_setup = get_header()

    performance = []

    if exp_setup.config["seed"] == -1:
        seeds = list(range(1234, 1234 + 10))
        num_runs = len(seeds)
    else:
        seeds = [exp_setup.config["seed"]]
        num_runs = 1

    all_metrics = []

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

        reward_only = True
        learning_alg = Sabre(exp_setup)
        exp_setup.logger.log("Running SABRE: Reward Only %r" % reward_only)

        policy_result = learning_alg.train(env=env,
                                           exp_id=exp_id,
                                           reward_only=reward_only)

        # fname = "./pt/sabre-clean-1/" \
        #         "sabre_sabre-clean-1_hor_5_max_1000_sabreb_1_sabree_500_sabref_1_sabrem_100_sabren_5_see_1234/" \
        #         "sabre-1665697549/policy_disag_model.npy.npz"
        #
        # policy_result = learning_alg.do_train_from_disag(env=env,
        #                                                  model_fname=fname)

        performance.append(policy_result)
        all_metrics.append({
            "Traces": env.get_traces(),
            "Unsafe Action Metric": env.unsafe_actions_metric,
            "Total Unsafe Action": env.num_unsafe_actions,
            "Total Oracle Calls": env.num_oracle_calls,
            "Oracle Call Metric": env.num_oracle_calls_metric
        })

    # Save traces
    with open("%s/metrics.pickle" % exp_setup.experiment, "wb") as f:
        pickle.dump(all_metrics, f)

    terminate(performance, exp_setup, seeds)


if __name__ == "__main__":

    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn')
    main()
