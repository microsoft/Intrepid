import random
import multiprocessing as mp

import torch

from learning.learning_utils.experience import Experience


class SharedReplayMemory:

    def __init__(self, state_dim, max_rollout_length, replay_memory_size):

        self.state_dim = state_dim
        self.max_rollout_length = max_rollout_length
        self.replay_memory_size = replay_memory_size

        self.rollouts = []
        for i in range(0, self.replay_memory_size):
            print(i)

            # Each experience contains a state, 1 float action and 1 float reward
            max_rollout_storage_length = (state_dim + 2) * max_rollout_length
            array = mp.Array('f', range(max_rollout_storage_length))
            self.rollouts.append(array)

        # Index of the row to be filled next
        self.index = mp.Value('i', 0)

        # If 0 then the memory is not full and 1 then it is full meaning each row contains a valid rollout
        self.memory_full = mp.Value('i', 0)

    def append(self, rollout):

        assert len(rollout) <= self.max_rollout_length, "Got a rollout which is longer than expected."

        self.rollouts[self.index.value][0] = len(rollout)
        pad = 1
        for i, experience in enumerate(rollout):

            state = experience.get_state()
            action = experience.get_action()
            reward = experience.get_reward()

            for j in range(0, self.state_dim):
                self.rollouts[self.index.value][pad + j] = state.view(-1)[j]
            self.rollouts[self.index.value][pad + self.state_dim] = float(action[-1]) # TODO
            self.rollouts[self.index.value][pad + self.state_dim + 1] = reward[-1]

            pad = pad + self.state_dim + 2

        self.index.value = self.index.value + 1
        if self.index.value == self.replay_memory_size:
            self.index.value = 0
            self.memory_full = 1

    def __len__(self):
        return self.index.value

    def sample(self):

        if self.memory_full == 0:
            index = random.randint(0, self.index.value - 1)
        else:
            index = random.randint(0, self.replay_memory_size - 1)

        rollout = []
        rollout_length = int(self.rollouts[index][0])

        pad = 1
        for i in range(0, rollout_length):

            state = torch.Tensor(self.state_dim)
            for j in range(0, self.state_dim):
                state[j] = self.rollouts[index][pad + j]
            action = int(self.rollouts[index][pad + self.state_dim])
            reward = float(self.rollouts[index][pad + self.state_dim + 1])

            pad = pad + self.state_dim + 2

            experience = Experience(state=state, action=action, reward=reward, next_state=None)
            rollout.append(experience)

        return rollout
