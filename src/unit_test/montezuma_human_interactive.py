import gym
import time
import pickle
import matplotlib.pyplot as plt

from textwrap import wrap
from skimage.transform import resize

plt.ion()
env = gym.make("MontezumaRevengeDeterministic-v4")


def process_obs_and_show(obs, seq, ret):
    obs = obs[34 : 34 + 160, :160]
    obs = resize(obs, (500, 500, 3))
    seq_str = ", ".join([str(action) for action in seq])
    plt.clf()

    plt.title("\n".join(wrap("Trajectory [%s], return: %f" % (seq_str, ret), 90)), fontsize=8)
    plt.imshow(obs)
    plt.show()


def take_action(action):
    obs = None
    reward = 0
    for _ in range(4):
        obs, reward_, _, _ = env.step(action)
        reward += reward_
    return obs, reward


def play(seq):
    obs = env.reset()
    ret = 0
    for ix, action in enumerate(seq):
        obs, reward = take_action(action)
        ret += reward
    process_obs_and_show(obs, seq, ret)
    return obs, ret


seq = []
ret = 0
obs = env.reset()
process_obs_and_show(obs, seq, ret)

while True:
    cmd_str = input("Enter a number between 0 and 18, and press b to go back, and press q to quit\n\n")

    cmd_seq = [tk.strip() for tk in cmd_str.split(",")]
    cmd_seq = [tk for tk in cmd_seq if len(tk) > 0]

    for cmd in cmd_seq:
        if cmd == "b":
            if len(seq) > 0:
                # go back
                seq.pop()
                obs, ret = play(seq)
            else:
                print("No observation to backtrack\n\n")

        elif cmd == "q":
            with open(
                "key-montezuma-achieved-return-%d-%d.pkl" % (ret, int(time.time())),
                "wb",
            ) as f:
                pickle.dump({"seq": seq, "total_return": ret}, f)
            print("Quitting.")
            exit(0)

        elif cmd.startswith("load"):
            with open(cmd.split()[1], "rb") as f:
                data = pickle.load(f)
            obs, ret = play(data["seq"])

        else:
            try:
                action = int(cmd)
                obs, reward = take_action(action)
                seq.append(action)
                ret += reward
                process_obs_and_show(obs, seq, ret)
            except Exception:
                print("Enter b, q or a number")
                continue
