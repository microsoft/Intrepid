import argparse
import gym
import json
import os
import random
import time

from gym_minigrid.window import Window
from environments.minigrid.gridworld_wrapper import GridWorldWrapper
from utils.beautify_time import beautify


def redraw(img):
    if not args.agent_view:
        img = env.render("rgb_array", tile_size=args.tile_size)

    window.show_img(img)


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, "mission"):
        print("Mission: %s" % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def step(action):
    obs, reward, done, info = env.step(action)
    print("step=%s, reward=%.2f" % (env.step_count, reward))

    if done:
        print("done!")
        reset()
    else:
        redraw(obs)


def key_handler(event):
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset()
        return

    if event.key == "left":
        step(env.actions.left)
        return
    if event.key == "right":
        step(env.actions.right)
        return
    if event.key == "up":
        step(env.actions.forward)
        return

    if event.key == "z":
        step(env.actions.left_forward)
        return
    if event.key == "x":
        step(env.actions.right_forward)
        return

    # Spacebar
    if event.key == " ":
        step(env.actions.toggle)
        return
    if event.key == "pageup":
        step(env.actions.pickup)
        return
    if event.key == "pagedown":
        step(env.actions.drop)
        return

    if event.key == "enter":
        step(env.actions.done)
        return


parser = argparse.ArgumentParser()
parser.add_argument("--env", help="gym environment to load", default=None)  #'MiniGrid-MultiRoom-N6-v0'
parser.add_argument("--seed", type=int, help="random seed to generate the environment with", default=-1)
parser.add_argument("--tile_size", type=int, help="size at which to render tiles", default=32)
parser.add_argument(
    "--agent_view",
    default=False,
    help="draw the agent sees (partially observable view)",
    action="store_true",
)

args = parser.parse_args()

if args.env is None:
    with open("./data/gridworld2/config.json", "r") as f:
        config = json.load(f)

    config["name"] = "gridworld2"
    config["save_trace"] = False
    config["trace_sample_rate"] = 50
    config["save_path"] = "./results/manual_control"
    config["exp_name"] = "manual_control"
    config["seed"] = 1234

    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])

    env = GridWorldWrapper(config)
    args.env = "MiniGrid-LavaTwoPathsEnv"
    print(env)
else:
    env = gym.make(args.env)

####################################
time_s = time.time()
for _ in range(0, 1000):
    env.reset(generate_obs=False)
    for h in range(0, 8):
        action = random.randint(0, 4)
        action = int(action)
        obs, reward, done, info = env.step(action, generate_obs=(h == 7))
time_taken = time.time() - time_s
print("Time taken for 1000 samples is %s." % beautify(time_taken))
print("Expected time that will take for 8 (H) x 500,000 is %s" % beautify(500 * 8 * time_taken))
exit(0)
####################################

print("Args tile size is ", args.tile_size)
window = Window("gym_minigrid - " + args.env)
window.reg_key_handler(key_handler)

# Blocking event loop
window.show(block=False)

while True:
    print("Resetting")
    obs, _ = env.reset()
    window.show_img(obs)

    for _ in range(0, config["horizon"]):
        # action = input("Take action ")
        action = random.randint(0, 4)
        action = int(action)
        obs, reward, done, info = env.step(action)
        print("Took action %d and got reward %f" % (action, reward))
        window.show_img(obs)
        time.sleep(0.2)
