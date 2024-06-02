import math
import time
import pickle
import random
import numpy as np

from learning.datastructures.episode import Episode
from learning.datastructures.count_probability import CountProbability


class FQIOracleDecoder:
    """Performs FQI Based Reinforcement Learning with oracle access to the decoder.
    Interactively samples a single episode in each iteration, which is used to refine
    the optimal policy based on counts"""

    def __init__(self, config, constants):
        self.horizon = config["horizon"]
        self.constants = constants
        self.actions = config["actions"]
        self.num_actions = config["num_actions"]
        self.max_iter = 1000000

    def generate_episode(self, env, q_values):
        """Generate a single episode by taking actions using the q_values"""

        start_obs, meta = env.reset()
        state = (0, meta["state"])
        episode = Episode(state=state, observation=start_obs)

        for step_ in range(0, self.horizon):
            if state in q_values:
                action = self.actions[q_values[state].argmax()]  # We separate between action and action indices
            else:
                action = random.choice(self.actions)  # Take uniform actions for states without q_values

            obs, reward, done, meta = env.step(action)
            state = (step_ + 1, meta["state"])

            episode.add(action=action, reward=reward, new_state=state, new_obs=obs)

            if done:
                break

        episode.terminate()

        return episode

    def q_value_iteration(self, q_values, model, counts, states_visited):
        for state in states_visited:
            if state not in q_values:
                # We set Q-values of these states to 1.0 which is the maximum optimistic reward the agent can get
                timestep = state[0]  # We append states with time step information. Start state has timestep 0
                q_values[state] = np.repeat(1.0 * (self.horizon - timestep), self.num_actions)

        for it in range(10):
            for state in states_visited:
                timestep = state[0]

                for action in self.actions:
                    # Compute the bonus reward
                    count_ = 0.0 if (state, action) not in counts else counts[(state, action)]
                    bonus_reward = 1.0 / math.sqrt(max(1, count_))

                    # Perform a single Bellman Operator update
                    # D(s, a) = R(s, a) + \sum_{s'} T(s'|s,a) max_{a'} Q(s', a')        , Inductive case
                    # D(s, a) = R(s, a)                                                 , Base Case
                    # Q(s, a) = \alpha Q(s, a) + (1 - \alpha) D(s, a)                   , Updates

                    if timestep == self.horizon - 1:
                        q_values[state][action] = bonus_reward

                    else:
                        if (state, action) in model:
                            # This means we have taken this action in this state before

                            prob_values = model[(state, action)].get_probability()
                            future_return = 0.0

                            for new_state, prob_val in prob_values:
                                future_return += prob_val * q_values[new_state].max()

                            # print("Bonus reward is %f and future_return is %f " % (bonus_reward, future_return))
                            q_values[state][action] = bonus_reward + future_return

                        else:
                            # We haven't taken this action in this state before
                            # Therefore, we keep q_values to their optimistic setting
                            pass

    def refine_model(self, model, episode):
        """
        Update the model based on the episode by doing count based estimation
        :param model: A dictionary from (state, action) to count based probability
        :param episode: A single episode object
        :return: None
        """

        transitions = episode.get_transitions()
        for state, action, new_states in transitions:
            if (state, action) not in model:
                model[(state, action)] = CountProbability()

            model[(state, action)].acc(new_states)

    def train(
        self,
        experiment,
        env,
        env_name,
        experiment_name,
        logger,
        use_pushover,
        debug,
        do_reward_sensitive_learning=False,
    ):
        """Execute HOMER algorithm on an environment using
        :param experiment:
        :param env:
        :param env_name:
        :param experiment_name:
        :param logger:
        :param use_pushover: True/False based on whether pushover is used
        :param debug:
        :param do_reward_sensitive_learning:
        :return:
        """

        counts = dict()  # Count is the number of times a state and action has been visited
        states_visited = set()  # States which have been visited
        q_values = dict()  # Dictionary from state to a numpy array over actions indicating Q values
        model = dict()
        time_start = time.time()

        for it in range(1, self.max_iter + 1):
            # Sample an episode with the current policy
            episode = self.generate_episode(env, q_values)

            # Refine the count
            for state, action in episode.get_state_action_pairs():
                if (state, action) in counts:
                    counts[(state, action)] += 1.0
                else:
                    counts[(state, action)] = 1.0

            self.refine_model(model, episode)

            for state in episode.get_states():
                states_visited.add(state)

            # Refine the value iteration
            self.q_value_iteration(q_values, model, counts, states_visited)

            if it % 100 == 0:
                # print("\n\n")
                # print("States visited are \n", states_visited)
                # for key, value in sorted(model.items()):
                #     print("Model: %r -> %r" % (key, value.get_probability()))
                # for key, value in sorted(q_values.items()):
                #     print("Q values: %r -> %r" % (key, value.tolist()))

                logger.log(
                    "Episodes %d, Number of states visited %d, Time taken %f sec"
                    % (it, len(states_visited), round(time.time() - time_start, 1))
                )
                print(
                    "Episodes %d, Number of states visited %d, Time taken %f sec"
                    % (it, len(states_visited), round(time.time() - time_start, 1))
                )

            if len(states_visited) == 3 * self.horizon + 2:
                print("Success. Reached all %d dataset" % (3 * self.horizon + 2))
                logger.log("Success. Reached all %d dataset" % (3 * self.horizon + 2))
                break

        # Save the model
        with open("%s/model.pkl" % experiment, "wb") as fp:
            pickle.dump(model, fp)

        return {"num_states_visited": len(states_visited)}
