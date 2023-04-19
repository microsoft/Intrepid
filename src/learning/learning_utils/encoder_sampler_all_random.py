import torch
import random
import utils.generic_policy as gp
from learning.learning_utils.transition import TransitionDatapoint

from utils.cuda import cuda_var
from learning.learning_utils.abstract_encoder_sampler import AbstractEncoderSampler


class EncoderSamplerAllRandom(AbstractEncoderSampler):
    """ Sampling procedure: Sample (x_1, a_1, x_1') and (x_2, a_2, x_2') by rolling in with a uniformly chosen policy
     followed by taking an action uniformly. Sample y ~ Bernoulli(1/2) and if y ==1 then (x_1, a_1, x_1') and
     otherwise (x_2, a_2, x_2').

     Each positive sample takes 1 episode and negative sample takes 2 episodes.
     """

    def __init__(self):
        AbstractEncoderSampler.__init__(self)

    @staticmethod
    def gather_samples(num_samples, env, actions, step, homing_policies, selection_weights=None):

        dataset = []
        for _ in range(num_samples):
            dataset.append(
                EncoderSamplerAllRandom._gather_sample(env, actions, step, homing_policies, selection_weights)
            )

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
        y = random.randint(0, 1)

        if y == 0:  # Add imposter
            next_obs, new_meta = EncoderSamplerAllRandom._gather_last_observation(env, actions, step,
                                                                                 homing_policies, selection_weights)
            reward = None      # Reward for imposter transition makes little sense
        elif y == 1:  # Take the action
            next_obs, reward, done, meta = env.step(deviation_action)
            new_meta = meta
        else:
            raise AssertionError("y can only be either 0 or 1")

        if new_meta is not None and "state" in new_meta:
            next_state = new_meta["state"]
        else:
            next_state = None

        data_point = TransitionDatapoint(curr_obs=current_obs,
                                         action=deviation_action,
                                         next_obs=next_obs,
                                         y=y,
                                         curr_state=curr_state,
                                         next_state=next_state,
                                         action_prob=action_prob,
                                         policy_index=ix,
                                         step=step,
                                         reward=reward)

        return data_point

    @staticmethod
    def _gather_last_observation(env, actions, step, homing_policies, selection_weights):

        start_obs, meta = env.reset()

        if step > 1:

            if selection_weights is None:
                # Select a homing policy for the previous time step randomly uniformly
                policy = random.choice(homing_policies[step - 1])
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

        action = random.choice(actions)
        new_obs, reward, done, meta = env.step(action)

        return new_obs, meta
