import pdb
import json
import numpy as np

from environments.matterport.matterport import Matterport

with open("data/matterport/config.json") as f:
    config = json.load(f)

env = Matterport(config)
img, info = env.reset()


def print_stuff():
    print(env.sim.getState()[0].scanId)
    print(env.sim.getState()[0].location.viewpointId)
    print(env.sim.getState()[0].viewIndex)
    print(env.sim.getState()[0].heading)
    print(env.sim.getState()[0].elevation)
    print(env.sim.getState()[0].step)
    print(env.sim.getState()[0].navigableLocations)
    print(np.array(env.sim.getState()[0].rgb, copy=False).shape)
    print()


print_stuff()
print("Taking action")
for i in range(0, config["horizon"]):
    img, reward, done, info = env.step(0)
print_stuff()

pdb.set_trace()