import torch
import random
import utils.generic_policy as gp

from utils.cuda import cuda_var
from learning.learning_utils.abstract_encoder_sampler import AbstractEncoderSampler
from learning.learning_utils.transition import TransitionDatapoint


class EncoderSamplerForwardReUse(AbstractEncoderSampler):
    """ Sampling procedure: Collect (x, a, x') by rolling in with a uniformly chosen policy
     followed by taking an action uniformly. For each sample (x, a, x') add k noise of the form
     (\tilde{x}, \tilde{a}, x') where \tilde{x}, \tilde{a} are randomly sampled from another example.
     k is tunable and defaults to 1.

     Each sample takes exactly 1 episode.
     """

    def __init__(self):
        AbstractEncoderSampler.__init__(self)

    @staticmethod
    def gather_samples(num_samples, env, actions, step, homing_policies, selection_weights=None, k=1):

        pos_dataset = []
        for _ in range(num_samples):
            pos_dataset.append(
                EncoderSamplerForwardReUse._gather_sample(env, actions, step, homing_policies, selection_weights)
            )

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
                neg_data.curr_obs = chosen_sample.curr_obs          # Replaced curr observation
                neg_data.curr_state = chosen_sample.curr_state      # Replaced curr state
                neg_data.action = chosen_sample.action              # Replaced action
                neg_data.action_prob = chosen_sample.action_prob    # Replaced action probability
                neg_data.policy_index = chosen_sample.policy_index  # Replaced policy that generates curr_obs
                neg_data.reward = None                              # Reward for fake transition makes little sense

                neg_dataset.append(neg_data)

        dataset = []
        dataset.extend(pos_dataset)
        dataset.extend(neg_dataset)

        # Shuffle the data to mix positive and negative samples
        random.shuffle(dataset)

        return dataset

    @staticmethod
    def _gather_sample(env, actions, step, homing_policies, selection_weights=None):
        """ Gather sample using ALL_RANDOM style """

        start_obs, meta = env.reset()
        if step > 1:

            if selection_weights is None:
                # Select a homing policy for the previous time step randomly uniformly
                ix = random.randint(0, len(homing_policies[step - 1]) - 1)
                policy = homing_policies[step - 1][ix]
            else:
                # Select a homing policy for the previous time step using the given weights
                # policy = random.choices(homing_policies[step - 1], weights=selection_weights, k=1)[0]
                ix = gp.sample_action_from_prob(selection_weights)
                policy = homing_policies[step - 1][ix]
            obs = start_obs

            for step_ in range(1, step):
                obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
                action = policy[step_].sample_action(obs_var)
                obs, reward, done, meta = env.step(action)

            current_obs = obs
        else:
            ix = None
            current_obs = start_obs

        if meta is not None and "state" in meta:
            curr_state = meta["state"]
        else:
            curr_state = None

        deviation_action = random.choice(actions)
        action_prob = 1.0/float(max(1, len(actions)))

        next_obs, reward, done, meta = env.step(deviation_action)
        new_meta = meta

        if new_meta is not None and "state" in new_meta:
            next_state = new_meta["state"]
        else:
            next_state = None

        data_point = TransitionDatapoint(curr_obs=current_obs,
                                         action=deviation_action,
                                         next_obs=next_obs,
                                         y=1,
                                         curr_state=curr_state,
                                         next_state=next_state,
                                         action_prob=action_prob,
                                         policy_index=ix,
                                         step=step,
                                         reward=reward)

        return data_point
