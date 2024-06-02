import torch
import random
import utils.generic_policy as gp
from learning.datastructures.transition import TransitionDatapoint

from utils.cuda import cuda_var


class EncoderSamplerIK:
    """Sampling procedure: Collect (x, a, x') by rolling in with a uniformly chosen policy
    followed by taking an action uniformly. Each sample takes exactly 1 episode.
    """

    def __init__(self):
        pass

    @staticmethod
    def gather_samples(num_samples, env, actions, step, homing_policies, selection_weights=None):
        dataset = []
        for _ in range(num_samples):
            dataset.append(EncoderSamplerIK._gather_sample(env, actions, step, homing_policies, selection_weights))

        return dataset

    @staticmethod
    def _gather_sample(env, actions, step, homing_policies, selection_weights=None):
        """Gather sample using ALL_RANDOM style"""

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
        action_prob = 1.0 / float(max(1, len(actions)))

        next_obs, reward, done, meta = env.step(deviation_action)
        new_meta = meta

        if new_meta is not None and "state" in new_meta:
            next_state = new_meta["state"]
        else:
            next_state = None

        data_point = TransitionDatapoint(
            curr_obs=current_obs,
            action=deviation_action,
            next_obs=next_obs,
            y=1,
            curr_state=curr_state,
            next_state=next_state,
            action_prob=action_prob,
            policy_index=ix,
            step=step,
            reward=reward,
        )

        return data_point
