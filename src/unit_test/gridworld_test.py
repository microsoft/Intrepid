import torch
import random
import numpy as np
import torch.multiprocessing as mp

from experiments.experiment_header import get_header
from environments.cerebral_env_meta.make_env import MakeEnvironment


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

        import imageio
        import matplotlib.pyplot as plt

        plt.ion()

        images = []

        for _ in range(0, 5):
            img, info = env.reset()
            plt.imshow(img)
            images.append(img)

            # print("Image shape is ", img.shape)
            for _ in range(0, exp_setup.config["horizon"]):
                action = random.choice(exp_setup.config["actions"])
                img, _, _, _ = env.step(action)
                images.append(img)
                # print("Image shape is ", img.shape)
                plt.imshow(img)
                plt.pause(0.05)

        imageio.mimsave("./gridworld.gif", images)

        import pdb

        pdb.set_trace()


if __name__ == "__main__":
    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
