import numpy as np


class ValueIteration:
    """
    Performs Bellman Optimal Q-iteration on Tabular MDP
    """

    def __init__(self):
        pass

    def do_value_iteration(self, tabular_mdp, reward_func=None, min_reward_val=0.0):
        actions = tabular_mdp.actions
        num_actions = len(actions)
        q_values = dict()

        for h in range(tabular_mdp.horizon, -1, -1):
            states = tabular_mdp.get_states(h)

            for state in states:
                state_with_timestep = (h, state)

                q_values[state_with_timestep] = np.repeat(min_reward_val, num_actions).astype(np.float32)

                for action in actions:
                    if h == tabular_mdp.horizon:
                        q_values[state_with_timestep][action] = 0.0
                    else:
                        q_val = 0.0
                        for new_state, prob_val in tabular_mdp.get_transitions(state, action):
                            if reward_func is None:
                                # Use the environment reward function
                                reward = tabular_mdp.get_reward(state, action, new_state, h)
                            else:
                                # Use the given reward function
                                reward = reward_func(state, action, new_state, h)

                            q_val += prob_val * (reward + tabular_mdp.gamma * q_values[(h + 1, new_state)].max())

                        q_values[state_with_timestep][action] = q_val

        return q_values
