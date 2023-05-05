class Episode:
    """Represents a single episode"""

    def __init__(self, state, observation, gamma=1.0):
        """
        Create an episode with the start_state state and start_observation observation
        :param state: start state
        :param observation: start observation
        """

        self._terminated = False
        self._states = [state]
        self._actions = []
        self._rewards = []
        self._observations = [observation]
        self._return = 0.0
        self._discount = 1.0
        self.gamma = gamma

    def add(self, action, reward, new_obs, new_state):
        """
        Add the result of a single action. Episodes are created incrementally so this action is being taken on the
        last state/observation in self.states which is nonempty
        :param action: Action taken
        :param reward: Reward received for taking that action
        :param new_obs:  New observation
        :param new_state: New state
        :return:
        """

        if self._terminated:
            raise AssertionError("Cannot add to a terminate episode")

        self._actions.append(action)
        self._rewards.append(reward)
        self._observations.append(new_obs)
        self._states.append(new_state)

        self._return = self._return + self._discount * reward
        self._discount = self._discount * self.gamma

    def terminate(self):
        """
        Terminate an episode
        :return:
        """
        self._terminated = True

    def get_states(self):
        """
        :return: All states in this episode
        """
        return self._states

    def get_actions(self):
        """
        :return: All actions in this episode
        """
        return self._actions

    def get_rewards(self):
        """
        :return: All rewards in this episode
        """
        return self._rewards

    def get_observations(self):
        """
        :return: All observations in this episode
        """
        return self._observations

    def get_return(self):
        """
        :return: Return the total discounted return in this episode
        """
        return self._return

    def get_state_action_pairs(self):
        """
        :return: Return an iterator of state and actions
        """

        return zip(self._states[:-1], self._actions)

    def get_obs_state_pairs(self):
        """
        :return: Return an iterator of state and observations
        """
        return zip(self._states, self._observations)

    def get_obs_action_pairs(self):
        """
        :return: Return an iterator of obs and actions
        """
        return zip(self._observations[:-1], self._actions)

    def get_transitions(self):
        """
        :return: Return all state transitions in this episode in the form (s, a, s')
        """
        return zip(self._states[:-1], self._actions, self._states[1:])

    def get_observation_transitions(self):
        """
        :return: Return all observation transitions in this episode in the form (x, a, x')
        """
        return zip(self._observations[:-1], self._actions, self._observations[1:])

    def get_state_observation_transitions(self):
        """
        :return: Return all state and observation transitions in this episode in the form (s, x, a, s', x')
        """
        return zip(
            self._states[:-1],
            self._observations[:-1],
            self._actions,
            self._states[1:],
            self._observations[1:],
        )

    def get_multi_step_observation_transitions(self, k):
        """
        :return: Return multi-step state and observation transitions in this episode in the form (x_h, a_h, x_{h+k})
        """
        klst = [k] * len(self._observations[:-k])
        return zip(self._observations[:-k], self._actions, self._observations[k:], klst)

    def get_acro_transitions(self, k):
        """
        :return: Return multi-step state and observation transitions in this episode in the form (x_h, a_h, x_{h+1}, x_{h+k})
        """
        # print('obs: {}'.format(self._observations))
        # print('k: {}; [:-k]: {}; [1:-(k-1)]: {}; [k:]: {}'.format(k,len(self._observations[:-k]), len(self._observations[1:-(k-1)]), len(self._observations[k:])))
        # raise Exception
        klst = [k] * len(self._observations[:-k])
        if k == 1:
            obs_n = self._observations[1:]
        else:
            obs_n = self._observations[1 : -(k - 1)]
        return zip(
            self._observations[:-k], self._actions, obs_n, self._observations[k:], klst
        )

    def get_len(self):
        """
        :return returns the length of the episode which is the number of actions taken
        """
        return len(self._actions)

    def get_transitions_at_step(self, step):
        """
        :param step: A time step h starting with 0
        :return: Transition (x_h, a_h, x_{h+1}) and None if the step exceeds what the episode has
        """

        if step + 1 >= len(self._observations):
            return None
        else:
            return (
                self._observations[step],
                self._actions[step],
                self._rewards[step],
                self._observations[step + 1],
            )

    def is_terminated(self):
        """
        :return True or False based on whether the episode is terminated or not
        """
        return self._terminated

    def __str__(self):
        return "%r -> " % (self._states[0],) + "-> ".join(
            [
                "%r -> %r, %r" % (action, reward, state)
                for (action, reward, state) in zip(
                    self._actions, self._rewards, self._states[1:]
                )
            ]
        )
