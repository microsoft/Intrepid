import math
import time
import pickle
import random
import numpy as np

from learning.learning_utils.episode import Episode
from learning.learning_utils.count_probability import CountProbability


class MBRLOracleDecoder:
    """ Performs Model Based Reinforcement Learning with oracle access to the decoder.
        Interactively samples a single episode in each iteration, which is used to refine
        the optimal policy based on counts """

    def __init__(self, exp_setup):

        self.horizon = exp_setup.config["horizon"]
        self.constants = exp_setup.constants
        self.actions = exp_setup.config["actions"]
        self.num_actions = exp_setup.config["num_actions"]
        self.gamma = exp_setup.config["gamma"]
        self.reward_type = exp_setup.constants["reward_type"]
        self.count_type = exp_setup.constants["count_type"]
        self.logger = exp_setup.logger
        self.debug = exp_setup.debug
        self.experiment = exp_setup.experiment
        self.max_iter = 1000000
        self.start_state = None   # TODO Temp

    def generate_episode(self, env, q_values):
        """ Generate a single episode by taking actions using the q_values """

        start_obs, meta = env.reset()
        state = (0, meta["state"])
        self.start_state = meta["state"]
        # state = meta["state"]

        episode = Episode(state=state,
                          observation=start_obs)

        for step_ in range(0, self.horizon):

            if state in q_values:
                action = self.actions[q_values[state].argmax()]     # We separate between action and action indices
            else:
                action = random.choice(self.actions)        # Take uniform actions for states without q_values

            obs, reward, done, meta = env.step(action)
            state = (step_ + 1, meta["state"])      # TODO do not concatenate timestep info to state
            # state = meta["state"]

            episode.add(action=action,
                        reward=reward,
                        new_state=state,
                        new_obs=obs)

            if done:
                break

        episode.terminate()

        return episode

    def _get_bonus_reward(self, counts, state, action):

        if self.count_type == "state":
            count = 0.0 if state[1] not in counts else counts[state[1]]
        elif self.count_type == "state-action":
            count = 0.0 if (state[1], action) not in counts else counts[(state[1], action)]
        else:
            raise AssertionError("Unhandled count type %r" % self.count_type)

        if self.reward_type == "deterministic":
            bonus_reward = 0.0 if count > 0 else 1.0
        elif self.reward_type == "stochastic":
            bonus_reward = 1.0 / math.sqrt(max(1, count))
        else:
            raise AssertionError("Unhandled reward type %r" % self.reward_type)

        return bonus_reward

    def _update_counts(self, counts, episode):

        if self.count_type == "state":
            for state in episode.get_states():

                state = state[1]            # Ignore timestep information
                if state in counts:
                    counts[state] += 1.0
                else:
                    counts[state] = 1.0

        elif self.count_type == "state-action":
            for (state, action) in episode.get_state_action_pairs():

                state = state[1]            # Ignore timestep information
                if (state, action) in counts:
                    counts[(state, action)] += 1.0
                else:
                    counts[(state, action)] = 1.0
        else:
            raise AssertionError("Unhandled count type %r" % self.count_type)

    def q_value_iteration(self, q_values, model, counts, states_visited):

        for state in states_visited:
            for h in range(self.horizon, -1, -1):
                state_with_timstep = (h, state)
                if state_with_timstep not in q_values:
                    # We set Q-values of these states to 1.0 which is the maximum optimistic reward the agent can get
                    # timestep = state[0]   # We append states with time step information. Start state has timestep 0
                    q_values[state_with_timstep] = np.repeat(1.0 * (self.horizon - h + 1), self.num_actions)
                    # q_values[state_with_timstep] = np.repeat(0.0, self.num_actions)

        for h in range(self.horizon, -1, -1):
            max_val = 0.0
            for state in states_visited:

                state_with_timestep = (h, state)

                # TODO visualize the dynamics
                # TODO play the greedy method and see if you reach the state which you are trying to reach

                for action in self.actions:

                    # TODO: Use Paul Mineiro's confidence interval
                    bonus_reward = self._get_bonus_reward(counts, state_with_timestep, action)

                    if h == self.horizon:
                        q_values[state_with_timestep][action] = bonus_reward
                    else:

                        if (state, action) not in model:
                            q_values[state_with_timestep][action] = bonus_reward + self.horizon - h
                        else:
                            prob_values = model[(state, action)].get_probability()
                            future_return = 0.0

                            for (new_state, prob_val) in prob_values:

                                future_return += prob_val * q_values[(h + 1, new_state)].max()

                            # print("Bonus reward is %f and future_return is %f " % (bonus_reward, future_return))
                            q_values[state_with_timestep][action] = bonus_reward + self.gamma * future_return

                    max_val = max(max_val, q_values[state_with_timestep][action])

            self.logger.log("Q-Value Iteration: Time step %d, Max Q_Values %f" % (h, max_val))

            # TODO remove timestep -> infinite discounted horizon
            # TODO John's optimization procedure: backup from states which have a reward,
            # TODO Miro's suggestion is to use a queue
            # TODO Store backward dynamics. Do dynamic's programming on the change.

    def refine_model(self, model, episode):
        """
        Update the model based on the episode by doing count based estimation
        :param model: A dictionary from (state, action) to count based probability
        :param episode: A single episode object
        :return: None
        """

        transitions = episode.get_transitions()
        for (state, action, new_state) in transitions:

            state, new_state = state[1], new_state[1]           # TODO

            if (state, action) not in model:
                model[(state, action)] = CountProbability()

            model[(state, action)].acc(new_state)

    def _print_model(self, model, states_visited):

        states_index = list(states_visited)

        for ix, state in enumerate(states_index):
            for action in range(0, self.num_actions):

                if (state, action) not in model:
                    self.logger.log("Transition %d x %d -> NA" % (ix, action))

                else:
                    prob_values = model[(state, action)].get_probability()

                    s = ""
                    for (new_state, prob_val) in prob_values:
                        s += "%d:%f\t" % (states_index.index(new_state), round(prob_val, 2))
                    self.logger.log("Transition %d x %d -> %s" % (ix, action, s))

    def train(self, env, exp_id=1):
        """ Execute MBRL algorithm on an environment
        """

        counts = dict()                 # Count is the number of times a state and action has been visited
        states_visited = set()          # States which have been visited
        state_action_visited = set()    # State action pair
        model = dict()                  # Dictionary from state x actions to a list of tuples {(state, prob)}
                                        # indicating the next state and probability of doing there
        q_values = dict()               # Dictionary from state to a numpy array over actions indicating Q values
        time_start = time.time()
        sum_total_return = 0
        max_return = 0

        for it in range(1, self.max_iter + 1):

            # Sample an episode with the current policy
            episode = self.generate_episode(env, q_values)

            # Compute return
            ret = 0.0
            for (state, action) in episode.get_state_action_pairs():
                bonus_reward = self._get_bonus_reward(counts, state, action)

                ret += bonus_reward

            # Refine the count
            for (state, action) in episode.get_state_action_pairs():
                state_action_visited.add((state[1], action))  # Ignoring the timestep information

            for state in episode.get_states():
                states_visited.add(state[1])    # Ignoring the timestep information

            self._update_counts(counts, episode)

            # Refine the model
            self.refine_model(model, episode)

            # Update the return statistics
            sum_total_return += episode.get_return()
            max_return = max(max_return, episode.get_return())

            if it % 10 == 0 and self.debug:
                self.logger.log("Episodes %d. Received return %f" % (it, ret))
                self._print_model(model, states_visited)

            if it % 10 == 0:
                # Refine the value iteration
                self.q_value_iteration(q_values, model, counts, states_visited)

            if it % 10 == 0:

                mean_return = sum_total_return / float(max(1, it))
                time_taken = round(time.time() - time_start, 1)

                self.logger.log("Episodes %d, Number of states visited %d, Number of state action pair %d, "
                                "Max return %f, Mean return %f, Time taken %f sec" %
                                (it, len(states_visited), len(state_action_visited),
                                 max_return, mean_return, time_taken))

        # Save the model
        with open("%s/model.pkl" % self.experiment, "wb") as fp:
            pickle.dump(model, fp)

        return {
            "num_states_visited": len(states_visited)
        }
