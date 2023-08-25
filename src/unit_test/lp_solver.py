import time
import torch
import random
import numpy as np
import torch.multiprocessing as mp

from experiments.experiment_header import get_header
from environments.intrepid_env_meta.make_env import MakeEnvironment
from learning.learning_utils.linear_disag_model import LinearDisagModel
from utils.beautify_time import beautify


def main():

    exp_setup = get_header()

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

        state = (2, 3)
        safety_dataset = []
        true_safety_vals = []
        num_examples = 1000
        for _ in range(num_examples):
            for action in exp_setup.config["actions"]:
                safety_ftr = env.get_safety_ftr(state, action)
                true_safety_vals.append(env.gold_w @ safety_ftr + env.gold_b)
                label = env.safety_query(safety_ftr)
                safety_dataset.append((safety_ftr, 1.0 if label else -1.0, action))

        dataset_size = len(safety_dataset)
        random.shuffle(safety_dataset)

        print("Safe values are of mean %f, std %f" % (np.mean(true_safety_vals), np.std(true_safety_vals)))
        print("Min safety values is %f, Max safety value is %f" % (np.min(true_safety_vals), np.max(true_safety_vals)))
        hist = dict()
        buckets = 20
        min_val, max_val = np.min(true_safety_vals), np.max(true_safety_vals)
        grid_size = (max_val - min_val) / float(buckets)
        for i in range(0, buckets):
            hist[i] = 0.0
        for val in true_safety_vals:
            ix = int((val - min_val) / grid_size)
            if ix >= 20:
                ix = 19
            hist[ix] += 1

        for i in range(0, buckets):
            print("%f to %f => %d entries" % (min_val + i * grid_size, min_val + (i+1) * grid_size, hist[i]))
        exit(0)

        for p in [0.2, 0.4, 0.6, 0.8]:

            exp_setup.logger.log("Starting experiment with p=%f" % p)
            train_size = int(p * dataset_size)
            test_size = dataset_size - train_size
            disag_model = LinearDisagModel(safety_dataset[:train_size], env.safe_action)

            exp_setup.logger.log("Disagreement Model Created")
            time_s = time.time()
            rd_ctr = 0
            safe_rd_ctr = 0
            safe_ctr = 0
            unsafe_rd_ctr = 0
            unsafe_ctr = 0

            for safety_ftr, label, action in safety_dataset[train_size:]:
                rd_flag = disag_model.in_region_of_disag(safety_ftr, action)
                if rd_flag:
                    rd_ctr += 1

                if label > 0.0:
                    safe_ctr += 1
                    if rd_flag:
                        safe_rd_ctr += 1
                else:
                    unsafe_ctr += 1
                    if rd_flag:
                        unsafe_rd_ctr += 1

            in_rd_acc = (100.0 * rd_ctr) / float(test_size)
            in_safe_rd_acc = (100.0 * safe_rd_ctr) / float(safe_ctr)
            in_unsafe_rd_acc = (100.0 * unsafe_rd_ctr) / float(unsafe_ctr)

            exp_setup.logger.log("P=%f, Train size %d, Test size %d, Safe labels %d, Unsafe labels %d, "
                                 "RD %f%%, Safe-RD %f%%, Unsafe-RD %f%%. Time taken %s" %
                                 (p, train_size, test_size, safe_ctr, unsafe_ctr, in_rd_acc, in_safe_rd_acc,
                                  in_unsafe_rd_acc, beautify(time.time() - time_s)))


if __name__ == "__main__":

    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn')
    main()
