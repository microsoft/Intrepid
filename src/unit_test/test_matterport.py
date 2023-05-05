import MatterSim
import numpy as np
import json
import os
from environments.cerebral_env_meta.environment_wrapper import (
    GenerateEnvironmentWrapper,
)


def first_test():
    env = MatterSim.Simulator()
    env.setCameraResolution(640, 480)
    env.setPreloadingEnabled(False)
    env.setDepthEnabled(False)
    env.setBatchSize(1)
    env.setCacheSize(2)

    env.setDatasetPath("/mnt/data/matterport/v1/scans")
    env.setNavGraphPath("/mnt/data/matterport/v1/connectivity/")

    env.initialize()
    house_id = "17DRP5sb8fy"
    room_id = "0f37bd0737e349de9d536263a4bdd60d"

    env.newEpisode([house_id], [room_id], [0], [0])

    def print_stuff():
        print(env.getState()[0].scanId)
        print(env.getState()[0].location.viewpointId)
        print(env.getState()[0].viewIndex)
        print(env.getState()[0].heading)
        print(env.getState()[0].elevation)
        print(env.getState()[0].step)
        print(env.getState()[0].navigableLocations)
        print(np.array(env.getState()[0].rgb, copy=False).shape)
        print()

    print_stuff()
    env.makeAction([1], [0], [0])
    print_stuff()

    env.newEpisode([house_id], [room_id], [0], [0])


def test_env():
    with open("../data/matterport/config.json") as f:
        config = json.load(f)

    config["save_trace"] = "True"
    config["trace_sample_rate"] = 500
    config["save_path"] = os.getenv("PT_OUTPUT_DIR")
    config["exp_name"] = "test"
    config["env_seed"] = 0
    config["policy_type"] = "linear"

    env = GenerateEnvironmentWrapper("matterport", config)
    env.reset()
    for _ in range(30):
        print("Stepping in env with action {}".format(1))
        obs, rew, done, info = env.step(1)
        print("Got:", rew, done, info["location"])
        print()
    env.reset()


if __name__ == "__main__":
    test_env()
