import random

from environments.rl_acid_env.slot_factored_mdp import SlotFactoredMDP

config = {"state_dim": 5, "grid_x": 3, "grid_y": 5, "horizon": 10}

mdp = SlotFactoredMDP(config)
obs, info = mdp.reset()
print("State \n", info["state"])

for _ in range(0, config["horizon"]):
    # pdb.set_trace()
    action = random.randint(0, config["grid_x"] * config["grid_y"] - 1)
    obs, reward, done, info = mdp.step(action)

    print("Action ", action)
    print("State \n", info["state"])
