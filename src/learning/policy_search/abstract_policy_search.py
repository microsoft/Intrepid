import torch
import random
import numpy as np

from utils.cuda import cuda_var


class AbstractPolicySearch:

    def __init__(self):
        pass

    @staticmethod
    def _gather_sample(env, actions, horizon, step, homing_policies, learned_policy, reward_func):

        start_obs, meta = env.reset()

        if step > 1:

            # Randomly a select a homing policy for the previous time step
            policy_index = random.randint(0, len(homing_policies[step - 1]) - 1)
            # policy = random.choice(homing_policies[step - 1])
            policy = homing_policies[step - 1][policy_index]
            obs = start_obs

            for step_ in range(1, step):
                obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
                action = policy[step_].sample_action(obs_var)
                obs, reward, done, meta = env.step(action)

            current_obs = obs
        else:
            policy_index = None
            current_obs = start_obs

        if meta is not None and "state" in meta:
            current_state = meta["state"]
        else:
            current_state = None

        total_reward = 0.0
        deviation_action = random.choice(actions)
        action_prob = 1 / float(len(actions))

        obs, reward, done, meta = env.step(deviation_action)

        if meta is not None and "state" in meta:
            new_state = meta["state"]
        else:
            new_state = None

        if reward_func is None:
            total_reward += reward
        else:
            reward = reward_func(obs, step)
            total_reward += reward

        for k in range(step + 1, horizon + 1):
            # Sample actions actions with respect to the policy
            obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
            action = learned_policy[k].sample_action(obs_var)
            obs, reward, done, meta = env.step(action)

            if reward_func is None:
                total_reward += reward
            else:
                reward = reward_func(obs, k)
                total_reward += reward

        datapoint = (current_obs, action_prob, deviation_action, total_reward, current_state, new_state, policy_index)

        return datapoint

    @staticmethod
    def _evaluate_intermediate_policy(deviation_policy, env, actions, horizon, step, homing_policies, learned_policy,
                                      reward_func):

        start_obs, meta = env.reset()

        if step > 1:

            # Randomly a select a homing policy for the previous time step
            policy = random.choice(homing_policies[step - 1])
            obs = start_obs

            for step_ in range(1, step):
                obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
                action = policy[step_].sample_action(obs_var)
                obs, reward, done, meta = env.step(action)

            current_obs = obs
        else:
            current_obs = start_obs

        total_reward = 0.0
        # Use the deviation policy to generate action
        obs_var = cuda_var(torch.from_numpy(current_obs)).float().view(1, -1)
        action = deviation_policy.get_argmax_action(obs_var)

        obs, reward, done, meta = env.step(action)

        if reward_func is None:
            total_reward += reward
        else:
            reward = reward_func(obs, step)
            total_reward += reward

        for k in range(step + 1, horizon + 1):
            # Sample actions actions with respect to the policy
            obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
            action = learned_policy[k].get_argmax_action(obs_var)
            obs, reward, done, meta = env.step(action)

            if reward_func is None:
                total_reward += reward
            else:
                reward = reward_func(obs, k)
                total_reward += reward

        return total_reward

    @staticmethod
    def _evaluate__learned_policy(env, horizon, reward_func, learned_policy, logger, num_rollouts, actions):

        mean_reward = 0.0
        action_distribution = {}
        for act in actions:
            action_distribution[act] = 0

        for _ in range(0, num_rollouts):

            total_reward = 0.0
            obs, meta = env.reset()

            for step in range(1, horizon + 1):
                obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
                action = learned_policy[step].get_argmax_action(obs_var)

                obs, reward, done, meta = env.step(action)

                if reward_func is None:
                    total_reward += reward
                else:
                    reward = reward_func(obs, step)
                    total_reward += reward

                if step == horizon:
                    action_distribution[action] += 1.0

            mean_reward += total_reward

        mean_reward /= float(max(1, num_rollouts))
        num_actions_taken = sum([action_distribution[key] for key in action_distribution])
        for act in action_distribution:
            action_distribution[act] = (action_distribution[act] / float(num_actions_taken)) * 100.0

        logger.log("Evaluating Learned Policy with %d rollouts. Action distribution %r. Empirical Total Reward: %r" %
                   (num_rollouts, action_distribution, mean_reward))

        return mean_reward

    @staticmethod
    def log_intermediate_results(dataset, optimal_policy_step, env, actions, horizon,
                                 step, homing_policies, learned_policy,
                                 reward_func, logger, num_eval_samples, batch_size):

        # Print prediction errors for each transition
        dataset_size = len(dataset)
        batches = [dataset[i:i + batch_size] for i in range(0, dataset_size, batch_size)]

        transition = dict()
        counts = dict()
        predicted_rewards_dict = dict()
        predicted_max_rewards = dict()

        for batch in batches:

            observation_batch = cuda_var(torch.cat([torch.from_numpy(np.array(point[0])).view(1, -1)
                                                    for point in batch], dim=0)).float()
            predicted_rewards = optimal_policy_step.gen_q_val(observation_batch)

            for i, dp in enumerate(batch):

                key = "%r -> %r -> %r" % (dp[4], dp[2], dp[5])
                predicted_rewards_numpy = predicted_rewards[i].cpu().data.numpy()

                if key in transition:
                    counts[key] += 1.0
                    transition[key] = np.append(transition[key], dp[3])
                    predicted_rewards_dict[key] = np.vstack([predicted_rewards_dict[key], predicted_rewards_numpy])
                else:
                    counts[key] = 1.0
                    transition[key] = np.array([dp[3]])
                    predicted_rewards_dict[key] = predicted_rewards_numpy

        for key in sorted(transition):
            mean_results = predicted_rewards_dict[key].mean(0).tolist()
            std_results = predicted_rewards_dict[key].std(0).tolist()
            mean_results_str = ", ".join(
                ["%r (std: %r)" % (round(mean, 2), round(std, 2)) for (mean, std) in zip(mean_results, std_results)])

            logger.log("CB:: %r, Count %r, Mean Reward %r (Std %r), Predicted Reward %r" %
                       (key, counts[key],
                        round(transition[key].mean(), 4),
                        round(transition[key].std(), 4),
                        mean_results_str
                        ))

        # Evaluate the deviation policy
        mean_total_reward = 0.0
        for _ in range(0, num_eval_samples):
            sampled_total_reward = AbstractPolicySearch._evaluate_intermediate_policy(
                optimal_policy_step, env, actions, horizon, step, homing_policies, learned_policy, reward_func)
            mean_total_reward += sampled_total_reward
        mean_total_reward = mean_total_reward/float(num_eval_samples)

        logger.log("Intermediate Evaluation on Step %r of Horizon %r. With %d rollouts the mean total reward is %r " %
                   (step, horizon, num_eval_samples, mean_total_reward))
