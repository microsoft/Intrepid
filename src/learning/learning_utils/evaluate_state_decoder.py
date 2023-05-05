import torch
import random

from utils.cuda import cuda_var


class EvaluateStateDecoder:
    """
    Evaluate state decoder on permutation invariant metric
    """

    def __init__(self, exp_setup):
        self.num_eval_samples = 5000
        self.logger = exp_setup.logger

    @staticmethod
    def _rollout(env, step, homing_policies):
        start_obs, meta = env.reset()

        # Select a homing policy for the previous time step randomly uniformly
        ix = random.randint(0, len(homing_policies[step]) - 1)
        policy = homing_policies[step][ix]
        obs = start_obs

        for step_ in range(1, step + 1):
            obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
            action = policy[step_].sample_action(obs_var)
            obs, reward, done, meta = env.step(action)

        return (
            obs,
            meta["state"]
            if "endogenous_state" not in meta
            else meta["endogenous_state"],
        )

    def evaluate(self, env, step, policy_cover, encoding_function):
        # Collect data
        succ = 0
        state_dist = dict()

        for it in range(0, self.num_eval_samples):
            # Sample two independent roll-outs
            obs1, state1 = self._rollout(env, step, policy_cover)
            obs2, state2 = self._rollout(env, step, policy_cover)

            if state1 not in state_dist:
                state_dist[state1] = 1
            else:
                state_dist[state1] += 1

            if state2 not in state_dist:
                state_dist[state2] = 1
            else:
                state_dist[state2] += 1

            # Compute the encoding
            abstract_state1 = encoding_function.encode_observations(obs1)
            abstract_state2 = encoding_function.encode_observations(obs2)

            if (state1 == state2 and abstract_state1 == abstract_state2) or (
                state1 != state2 and abstract_state1 != abstract_state2
            ):
                succ += 1

            if it % 100 == 99:
                acc = (succ * 100.0) / float(it + 1)
                self.logger.log(
                    "Evaluate State Decoder: After %d many samples. The accuracy is %f%%"
                    % (it, acc)
                )

        state_dist = {
            k: (v * 100.0) / float(2 * self.num_eval_samples)
            for k, v in state_dist.items()
        }
        self.logger.log(
            "Evaluate State Decoder: State distribution %r" % sorted(state_dist.items())
        )

        acc = (succ * 100.0) / float(self.num_eval_samples)
        self.logger.log(
            "Evaluate State Decoder: After %d many samples. The accuracy is %f%%"
            % (self.num_eval_samples, acc)
        )

        return acc
