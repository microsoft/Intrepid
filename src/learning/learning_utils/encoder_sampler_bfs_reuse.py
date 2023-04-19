import torch
import random
from learning.learning_utils.transition import TransitionDatapoint

from utils.cuda import cuda_var
from learning.learning_utils.abstract_encoder_sampler import AbstractEncoderSampler


class EncoderSamplerBFSReUse(AbstractEncoderSampler):
    """ Sampling procedure: Collect (x, a, x') by rolling in with a uniformly chosen policy
     followed by taking an action uniformly. Unlike ReUse, we do a BFS for collecting the data which is efficient for
     deterministic problems. We iterate over every homing policy at previous time step and then take each action for
     every policy. For each sample (x, a, x') add k noise of the form (x, a, x") where
     x" are randomly sampled from other examples. k is tunable and defaults to 1.

     Each sample takes exactly 1 episode.
     """

    def __init__(self):
        AbstractEncoderSampler.__init__(self)

    @staticmethod
    def gather_samples(num_samples, env, actions, step, homing_policies, selection_weights=None, k=1):

        assert selection_weights is None, "Selection weights don't make sense with BFS"

        pos_dataset = []

        if step == 1:
            num_prev_policy = 1
        else:
            num_prev_policy = len(homing_policies[step - 1])

        while len(pos_dataset) < num_samples:
            # Without the while condition, we can collect a very tiny dataset that is not useful for learning
            for ix in range(0, num_prev_policy):
                for action in actions:
                    pos_dataset.append(
                        EncoderSamplerBFSReUse._gather_sample(env, actions, step, ix, action, homing_policies))

        num_pos = len(pos_dataset)

        neg_dataset = []
        for i in range(num_pos):
            for _ in range(k):

                # Find a negative example
                neg_data = pos_dataset[i].make_copy()

                # Make neg_data fake by replacing last observation and state with a randomly chosen example
                j = random.randint(0, num_pos - 1)
                chosen_sample = pos_dataset[j]

                neg_data.y = 0                                      # Marked as fake
                neg_data.next_obs = chosen_sample.next_obs          # Replaced last observation
                neg_data.next_state = chosen_sample.next_state      # Replaced last state
                neg_data.reward = None                              # Reward for fake transition makes little sense

                neg_dataset.append(neg_data)

        dataset = []
        dataset.extend(pos_dataset)
        dataset.extend(neg_dataset)

        # Shuffle the data to mix positive and negative samples
        random.shuffle(dataset)

        return dataset

    @staticmethod
    def _gather_sample(env, actions, step, ix, action, homing_policies):
        """ Gather sample by roll-in with the ix^th previous policy and then take the given action """

        start_obs, meta = env.reset()
        if step > 1:

            policy = homing_policies[step - 1][ix]
            obs = start_obs

            for step_ in range(1, step):
                obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
                policy_action = policy[step_].sample_action(obs_var)
                obs, reward, done, meta = env.step(policy_action)

            current_obs = obs
        else:
            ix = None       # Make ix None as when step = 1 we get ix = 1 as input
            current_obs = start_obs

        if meta is not None and "state" in meta:
            curr_state = meta["state"]
        else:
            curr_state = None

        action_prob = 1.0/float(max(1, len(actions)))

        next_obs, reward, done, meta = env.step(action)
        new_meta = meta

        if new_meta is not None and "state" in new_meta:
            next_state = new_meta["state"]
        else:
            next_state = None

        data_point = TransitionDatapoint(curr_obs=current_obs,
                                         action=action,
                                         next_obs=next_obs,
                                         y=1,
                                         curr_state=curr_state,
                                         next_state=next_state,
                                         action_prob=action_prob,
                                         policy_index=ix,
                                         step=step,
                                         reward=reward)

        return data_point
