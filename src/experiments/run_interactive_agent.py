import torch
import random
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from experiments.experiment_save import terminate
from experiments.experiment_header import get_header
from environments.cerebral_env_meta.make_env import MakeEnvironment


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
        exp_setup.logger.log("========= STARTING EXPERIMENT %d (Seed = %d) ======== " % (exp_id, exp_setup.config["seed"]))

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

        exp_setup.config["actions"]
        total_return = 0.0
        obs, info = env.reset()

        imageio.imwrite("%s/img_0.png" % exp_setup.experiment, obs)

        plt.ion()
        plt.clf()
        plt.imshow(obs)
        plt.show()
        plt.pause(1)

        for h in range(exp_setup.config["horizon"]):
            while True:
                action_str = input("[Time Step %d] Enter action (or q to quit):" % (h + 1))
                # action_str = str(random.choice(actions))
                # action_str = str(env.get_optimal_action())
                try:
                    if action_str == "q":
                        break
                    else:
                        action = int(action_str)

                except ValueError:
                    print("Entered value has to be an action (integer) or q to denote quit")
                    continue

                else:
                    break

            if action_str == "q":
                break

            obs, reward, done, info = env.step(action)

            plt.clf()
            plt.imshow(obs)
            plt.show()
            plt.pause(1)

            imageio.imwrite("%s/img_%d.png" % (exp_setup.experiment, h + 1), obs)

            total_return += reward
            # pdb.set_trace()

        policy_result = {"total_return": total_return}
        performance.append(policy_result)

    terminate(performance, exp_setup, seeds)


if __name__ == "__main__":
    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
