import math
import numpy as np


class QLearningBonus:
    """
    Performs Q-learning with bonus described here: Is Q-learning Provably Efficient? NeurIPS 2018.

    Assumes rewards are in [0, 1].

    """

    def __init__(self):

        self.max_epsiodes = 1000
        self.failure_prob = 0.01
        self.c = 1

    def train(self, env, action_space, horizon):

        # This value should be of the form log(S A T / p) where p is failure probability
        iota = math.log((2 * 162 * horizon) / self.failure_prob)

        num_actions = len(action_space)
        q_val = dict()
        n_val = dict()
        v_val = dict()

        for h in range(0, horizon):
            # Given input as s, returns a numpy array over actions giving current estimated q_values.
            # Initialized optimistically by H. Since reward is 1 and, therefore, max value of Q_h is H - h =< H.
            q_val[h] = dict()

            # Given input as s, returns a numpy array over actions giving current estimated v_values.
            # Initialized optimistically by H. Since reward is 1 and, therefore, max value of V_h is H - h =< H.
            v_val[h] = dict()

            # Given input as s, returns a numpy array over actions giving current visitation counts.
            # Initialized optimistically by 0.
            n_val[h] = dict()

        # V_values are also computable for terminal states
        v_val[horizon] = dict()

        # unique states
        unique_states = set()

        # unique state-action pairs
        unique_state_actions = set()

        for k in range(0, self.max_epsiodes):

            x = env.reset()
            unique_states.add(x)

            for h in range(0, horizon):

                if x in q_val[h]:
                    a = np.argmax(q_val[h][x])
                else:
                    q_val[h][x] = horizon * np.ones(num_actions).astype(np.float32)
                    a = np.random.choice(action_space)

                # Take action a
                new_x, r, done, info = env.step(a)

                unique_state_actions.add((x, a))

                if x not in n_val[h]:
                    n_val[h][x] = np.zeros(num_actions).astype(np.float32)

                # Update frequency
                n_val[h][x][a] = n_val[h][x][a] + 1

                count = n_val[h][x][a]

                # Set bonus and learning rate
                alpha_t = (horizon + 1) / float(horizon + count)
                bonus = self.c * horizon * math.sqrt(horizon * iota / float(count))

                if new_x not in v_val[h+1]:
                    v_val[h + 1][new_x] = horizon * np.ones(num_actions).astype(np.float32)

                # Update q_val and v_val
                q_val[h][x][a] = (1 - alpha_t) * q_val[h][x][a] + alpha_t * (r + v_val[h + 1][new_x] + bonus)
                v_val[h][x] = min(horizon, np.max(q_val[h][x]))

                # Update current observation x
                x = new_x
                unique_states.add(x)

                if (k + 1) % 100 == 0:

                    print("Episode %d: Total number of states (across time) discovered %d; "
                          "Total number of state-actions (across time) discovered %d" %
                          (k, len(unique_states), len(unique_state_actions)))

