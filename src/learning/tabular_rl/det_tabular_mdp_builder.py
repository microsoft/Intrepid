from learning.tabular_rl.abstract_tabular_mdp import AbstractTabularMDP


class DetTabularMDPBuilder(AbstractTabularMDP):
    """
    Builder class to construct a deterministic tabular MDP
    """

    def __init__(self, actions, horizon, gamma=1.0):

        AbstractTabularMDP.__init__(self, actions, horizon, gamma)

        self.actions = actions
        self.horizon = horizon
        self.gamma = gamma

        # States reached at different time step
        # timestep -> [state1, state2, ...]
        self._states = dict()

        # (state, action) -> [(new_state, 1.0)]
        self._transitions = dict()

        # (state, action) -> scalar_value
        self._rewards = dict()

        self._finalize = False

    def add_state(self, state, timestep):

        assert not self._finalize, "This MDP has been finalized so new states cannot be added to it."

        if timestep not in self._states:
            self._states[timestep] = []

        self._states[timestep].append(state)

    def add_transition(self, state, action, new_state):

        assert not self._finalize, "This MDP has been finalized so new transitions cannot be added to it."

        if (state, action) in self._transitions:
            return

        self._transitions[(state, action)] = [(new_state, 1.0)]

    def add_reward(self, state, action, reward):

        assert not self._finalize, "This MDP has been finalized so new rewards cannot be added to it."

        if (state, action) in self._rewards:
            return

        self._rewards[(state, action)] = reward

    def finalize(self):
        self._finalize = True

    def get_states(self, timestep):
        return self._states[timestep]

    def num_states(self, timestep):
        return len(self._states[timestep])

    def get_transitions(self, state, action):
        return self._transitions[(state, action)]

    def get_reward(self, state, action, next_state, step):
        return self._rewards[(state, action)]
