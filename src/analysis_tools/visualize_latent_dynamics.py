import math
import time
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from learning.learning_utils.episode import Episode
from learning.learning_utils.count_probability import CountProbability


class Index(object):
    ind = 0

    def __init__(self, visualize_dynamics, bpause, bskip, bexit):
        self.visualize_dynamics = visualize_dynamics
        self.bpause = bpause
        self.bskip = bskip
        self.bexit = bexit

    def pause(self, event):
        self.bpause.label.set_text("Paused")

        try:
            input("Enter an input to continue or esc to exit")
        except Exception:
            pass
        self.bpause.label.set_text("Pause")

    def skip(self, event):
        self.visualize_dynamics.visualize = not self.visualize_dynamics.visualize

        if self.visualize_dynamics.visualize:
            self.bskip.label.set_text("Skip")
        else:
            self.bskip.label.set_text("Visualize")

    def exit_code(self, event):
        print("Exiting.")
        exit(0)


class VisualizeDynamics:
    """Performs Model Based Reinforcement Learning with oracle access to the decoder.
    Interactively samples a single episode in each iteration, which is used to refine
    the optimal policy based on counts"""

    def __init__(self, config, constants):
        self.horizon = config["horizon"]
        self.constants = constants
        self.actions = config["actions"]
        self.num_actions = config["num_actions"]
        self.gamma = config["gamma"]
        self.reward_type = constants["reward_type"]
        self.count_type = constants["count_type"]
        self.max_iter = 1000000
        self.start_state = None  # TODO Temp

        self.visualize = True
        self.pause_time = 0.1

        # Data
        self.state_obs_map = dict()

        # plt.title("Visualize Model Based RL")
        self.f, self.axarr = plt.subplots(1, 3, figsize=(12, 6))

        axpause = plt.axes([0.3, 0.05, 0.1, 0.075])
        bpause = Button(axpause, "Pause")

        axskip = plt.axes([0.41, 0.05, 0.1, 0.075])
        bskip = Button(axskip, "Skip")

        axexit = plt.axes([0.52, 0.05, 0.1, 0.075])
        bexit = Button(axexit, "Exit")

        self.callback = Index(self, bpause, bskip, bexit)

        bpause.on_clicked(self.callback.pause)
        bskip.on_clicked(self.callback.skip)
        bexit.on_clicked(self.callback.exit_code)

    def generate_episode(self, env, q_values):
        """Generate a single episode by taking actions using the q_values"""

        start_obs, meta = env.reset()
        state = (0, meta["state"])
        self.start_state = meta["state"]
        # state = meta["state"]
        episode = Episode(state=state, observation=start_obs)

        for step_ in range(0, self.horizon):
            if state in q_values:
                action = self.actions[q_values[state].argmax()]  # We separate between action and action indices
            else:
                action = random.choice(self.actions)  # Take uniform actions for states without q_values

            obs, reward, done, meta = env.step(action)
            state = (
                step_ + 1,
                meta["state"],
            )  # TODO do not concatenate timestep info to state
            # state = meta["state"]

            episode.add(action=action, reward=reward, new_state=state, new_obs=obs)

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
                state = state[1]  # Ignore timestep information
                if state in counts:
                    counts[state] += 1.0
                else:
                    counts[state] = 1.0

        elif self.count_type == "state-action":
            for state, action in episode.get_state_action_pairs():
                state = state[1]  # Ignore timestep information
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

                            for new_state, prob_val in prob_values:
                                future_return += prob_val * q_values[(h + 1, new_state)].max()

                            # print("Bonus reward is %f and future_return is %f " % (bonus_reward, future_return))
                            q_values[state_with_timestep][action] = bonus_reward + self.gamma * future_return

                    max_val = max(max_val, q_values[state_with_timestep][action])

            print("Q-Value Iteration: Time step %d, Max Q_Values %f" % (h, max_val))

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
        for state, action, new_state in transitions:
            state, new_state = state[1], new_state[1]  # TODO

            if (state, action) not in model:
                model[(state, action)] = CountProbability()

            model[(state, action)].acc(new_state)

    @staticmethod
    def _get_transition(state, action, states_visited, model):
        state_id = states_visited[state][0]

        if (state, action) not in model:
            return "Learned Model: %d x %d -> NA" % (state_id, action), []

        else:
            prob_values = model[(state, action)].get_probability()

            s = None
            new_states = []
            for new_state, prob_val in prob_values:
                new_state_id = states_visited[new_state][0]
                if s is None:
                    s = "State %d:%r" % (new_state_id, round(prob_val, 2))
                else:
                    s += ",  State %d:%r" % (new_state_id, round(prob_val, 2))
                new_states.append(new_state)

            return (
                "Learned Model: State %d    x    Action %d    =>    %s" % (state_id, action, s),
                new_states,
            )

    def print_model(self, model, states_visited):
        for state in states_visited:
            for action in range(0, self.num_actions):
                strrep, _ = self._get_transition(state, action, states_visited, model)
                print(strrep)

    def visualize_episode(self, episode_id, episode_return, episode, states_visited, model):
        states = episode.get_states()
        actions = episode.get_actions()
        observations = episode.get_observations()
        num_states = len(states_visited)

        for i, action in enumerate(actions):
            for k in range(0, len(self.axarr)):
                self.axarr[k].cla()
                self.axarr[k].axis("off")

            curr_state = states[i]
            curr_state = curr_state[1]  # Ignore the time step information
            curr_state_id, curr_state_count = states_visited[curr_state]
            if curr_state_count == 1:
                curr_state_label = "[novel state]"
            else:
                curr_state_label = "[count: %d]" % curr_state_count
            curr_obs = observations[i]

            next_state = states[i + 1]
            next_state = next_state[1]  # Ignore the time step information
            next_state_id, new_state_count = states_visited[next_state]
            if new_state_count == 1:
                next_state_label = "[novel state]"
            else:
                next_state_label = "[count: %d]" % new_state_count
            next_obs = observations[i + 1]

            curr_obs = curr_obs.squeeze()
            next_obs = next_obs.squeeze()

            transition_strrep, possible_states = self._get_transition(curr_state, action, states_visited, model)

            self.axarr[0].imshow(curr_obs)

            self.axarr[0].text(
                5,
                -30,
                "Episode ID: %d,    Explore-Return: %r,    Num states: %d" % (episode_id, episode_return, num_states),
            )
            self.axarr[0].text(5, -20, transition_strrep)
            self.axarr[1].imshow(next_obs)

            special_case = False
            if len(possible_states) > 1:
                other_state = None
                for another_state in possible_states:
                    if another_state != next_state:
                        other_state = another_state
                        break

                if other_state in self.state_obs_map:
                    self.axarr[2].imshow(self.state_obs_map[other_state].squeeze())
                    special_case = True

            if special_case:
                other_state_id, other_state_count = states_visited[other_state]
                other_state_label = "[count: %d]" % other_state_count
                self.axarr[0].text(
                    5,
                    -10,
                    "State %d %s    x    Action %d    =>    State %d %s               Other state %d %s"
                    % (
                        curr_state_id,
                        curr_state_label,
                        action,
                        next_state_id,
                        next_state_label,
                        other_state_id,
                        other_state_label,
                    ),
                )

            else:
                self.axarr[0].text(
                    5,
                    -10,
                    "State %d %s    x    Action %d    =>    State %d %s"
                    % (
                        curr_state_id,
                        curr_state_label,
                        action,
                        next_state_id,
                        next_state_label,
                    ),
                )

            plt.show()
            if special_case:
                plt.pause(3.0)
            else:
                plt.pause(self.pause_time)

            if not self.visualize:
                return

    @staticmethod
    def pause_fn():
        pass

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
        states_visited = dict()  # States which have been visited
        state_action_visited = set()  # State action pair
        model = dict()  # Dictionary from state x actions to a list of tuples {(state, prob)}
        # indicating the next state and probability of doing there
        q_values = dict()  # Dictionary from state to a numpy array over actions indicating Q values
        time_start = time.time()
        sum_total_return = 0
        max_return = 0
        num_state = 0

        plt.ion()

        for it in range(1, self.max_iter + 1):
            # Sample an episode with the current policy
            episode = self.generate_episode(env, q_values)

            # Compute return
            ret = 0.0
            for state, action in episode.get_state_action_pairs():
                bonus_reward = self._get_bonus_reward(counts, state, action)

                ret += bonus_reward
            print("Received Return ", ret)

            # Refine the count
            for state, action in episode.get_state_action_pairs():
                state_action_visited.add((state[1], action))  # Ignoring the timestep information

            for state, obs in zip(episode.get_states(), episode.get_observations()):
                if state[1] not in self.state_obs_map:
                    self.state_obs_map[state[1]] = obs

            for state_with_timestep in episode.get_states():
                state = state_with_timestep[1]  # Ignoring the timestep information

                if state not in states_visited:
                    states_visited[state] = [num_state + 1, 1]
                    num_state += 1
                else:
                    state_id, old_count = states_visited[state]
                    states_visited[state] = [state_id, old_count + 1]

            self._update_counts(counts, episode)

            # Refine the model
            self.refine_model(model, episode)

            # Update the return statistics
            sum_total_return += episode.get_return()
            max_return = max(max_return, episode.get_return())

            # Visualize Episode
            if self.visualize:
                self.visualize_episode(it, ret, episode, states_visited, model)
            else:
                plt.show()
                plt.pause(0.001)

            if it % 10 == 0:
                print("Episodes %d" % it)

            if it % 10 == 0:
                self.print_model(model, states_visited)
                # Refine the value iteration
                self.q_value_iteration(q_values, model, counts, states_visited)

            if it % 10 == 0:
                mean_return = sum_total_return / float(max(1, it))
                time_taken = round(time.time() - time_start, 1)

                logger.log(
                    "Episodes %d, Number of states visited %d, Number of state action pair %d, "
                    "Max return %f, Mean return %f, Time taken %f sec"
                    % (
                        it,
                        len(states_visited),
                        len(state_action_visited),
                        max_return,
                        mean_return,
                        time_taken,
                    )
                )
                print(
                    "Episodes %d, Number of states visited %d, Number of state action pair %d, "
                    "Max return %f, Mean return %f, Time taken %f sec"
                    % (
                        it,
                        len(states_visited),
                        len(state_action_visited),
                        max_return,
                        mean_return,
                        time_taken,
                    )
                )

        # Save the model
        with open("%s/model.pkl" % experiment, "wb") as fp:
            pickle.dump(model, fp)

        return {"num_states_visited": len(states_visited)}
