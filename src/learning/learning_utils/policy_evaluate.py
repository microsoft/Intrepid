import torch

from utils.cuda import cuda_var


def evaluate(
    env, policy, horizon, logger, train_episodes, sum_train_reward, regret=True
):
    """Compute mean total reward and the number of episodes to reach half regret of the optimal policy"""

    optimal_policy_v = env.get_optimal_value()

    if optimal_policy_v is None or not regret:
        # Evaluate the policy based on fixed number of episodes and computing total reward
        return evaluate_for_policy_value(env, policy, horizon, logger, train_episodes)
    else:
        # Evaluate the policy based on total number of episodes needed to reach half of optimal regret
        return evaluate_for_half_regret(
            env,
            policy,
            horizon,
            optimal_policy_v,
            logger,
            train_episodes,
            sum_train_reward,
        )


def generate_failure_result(env, train_samples, regret=True):
    """When the agent fails in the middle of training. This function allows it to return the intermediate result"""

    optimal_policy_v = env.get_optimal_value()
    if optimal_policy_v is None or not regret:
        return {
            "train_episodes": train_samples,
            "test_episodes": 0,
            "policy_val": float("-inf"),
        }

    else:
        return {
            "total_episodes_half_regret": float("inf"),
            "train_episodes": train_samples,
            "test_episodes": 0,
            "policy_val": float("-inf"),
        }


def evaluate_for_policy_value(
    env, policy, horizon, logger, train_episodes, num_eval_episodes=100
):
    """Evaluate the policy based on estimate of value function"""

    cumm_reward = 0.0

    for _ in range(1, num_eval_episodes + 1):
        # Rollin for steps
        obs, meta = env.reset()

        for step in range(1, horizon + 1):
            obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)

            if type(policy) == dict:
                action = policy[step].sample_action(obs_var)
            else:
                action = policy.sample_action(obs_var, step - 1)
            obs, reward, done, meta = env.step(action)
            cumm_reward += reward

    policy_val = cumm_reward / float(max(1, num_eval_episodes))
    logger.log("Estimated policy value %f" % policy_val)

    return {
        "total_train_episodes": train_episodes,
        "total_test_episodes": num_eval_episodes,
        "est_policy_val": policy_val,
    }


def evaluate_for_half_regret(
    env,
    policy,
    horizon,
    optimal_policy_v,
    logger,
    train_episodes,
    sum_train_reward,
    max_episodes=500000,
):
    """Evaluate the policy based on number of samples needed to reach half the regret"""

    max_episodes -= train_episodes
    opt_policy_cumm_ret = optimal_policy_v * train_episodes
    policy_cumm_ret = sum_train_reward
    test_episode = 0
    logger.log(
        "Evaluating Policy. V* = %r, At init (num train episodes %d): Total reward %r (Optimal) vs %r (System)"
        % (optimal_policy_v, train_episodes, opt_policy_cumm_ret, policy_cumm_ret)
    )

    for test_episode in range(1, max_episodes + 1):
        # Rollin for steps
        start_obs, meta = env.reset()
        obs = start_obs
        total_reward = 0.0

        for step in range(1, horizon + 1):
            obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
            if type(policy) == dict:
                action = policy[step].sample_action(obs_var)
            else:
                action = policy.sample_action(obs_var, step - 1)
            obs, reward, done, meta = env.step(action)
            total_reward += reward

        opt_policy_cumm_ret += optimal_policy_v
        policy_cumm_ret += total_reward
        ratio = policy_cumm_ret / float(max(1.0, opt_policy_cumm_ret))
        policy_val = (policy_cumm_ret - sum_train_reward) / float(max(1, test_episode))

        if test_episode % 10000 == 0:
            logger.log(
                "(Total Episodes %d) Total reward %r (Optimal) vs %r (System). Ratio %r"
                % (
                    train_episodes + test_episode,
                    opt_policy_cumm_ret,
                    policy_cumm_ret,
                    ratio,
                )
            )

        if policy_cumm_ret >= 0.5 * opt_policy_cumm_ret:
            logger.log(
                "Exceeded half of V^* after %r test episodes. Total test+train episodes %r. Estimate V %r"
                % (test_episode, train_episodes + test_episode, policy_val)
            )
            print(
                "Exceeded half of V^* after %r test episodes. Total test+train episodes %r. Estimate V %r"
                % (test_episode, train_episodes + test_episode, policy_val)
            )

            result = {
                "total_episodes_half_regret": train_episodes + test_episode,
                "train_episodes": train_episodes,
                "test_episodes": test_episode,
                "policy_val": policy_val,
            }

            return result

    policy_val = (policy_cumm_ret - sum_train_reward) / float(max(1, test_episode))

    logger.log(
        "Exceeded max steps. Learned/V* ratio is %r"
        % (policy_cumm_ret / max(0.00001, float(opt_policy_cumm_ret)))
    )
    print(
        "Exceeded max steps. Learned/V* ratio is %r"
        % (policy_cumm_ret / max(0.00001, float(opt_policy_cumm_ret)))
    )

    result = {
        "total_episodes_half_regret": float("inf"),
        "train_episodes": train_episodes,
        "test_episodes": test_episode,
        "policy_val": policy_val,
    }

    return result
