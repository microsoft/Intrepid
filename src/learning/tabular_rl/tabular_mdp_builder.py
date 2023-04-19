from learning.learning_utils.count_conditional_probability import CountConditionalProbability
from learning.tabular_rl.abstract_tabular_mdp import AbstractTabularMDP
from utils.average import AverageUtil


class TabularMDPBuilder(AbstractTabularMDP):
    """
    Builder class to construct a general tabular MDP
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
        self._transitions = CountConditionalProbability()

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

        self._transitions.add(new_state, (state, action))

    def add_reward(self, state, action, new_state, reward):

        assert not self._finalize, "This MDP has been finalized so new rewards cannot be added to it."

        if (state, action, new_state) not in self._rewards:
            self._rewards[(state, action, new_state)] = AverageUtil()

        self._rewards[(state, action, new_state)].acc(reward)

    def finalize(self):
        self._finalize = True

    def get_states(self, timestep):
        return self._states[timestep]

    def num_states(self, timestep):
        return len(self._states[timestep])

    def get_transitions(self, state, action):

        res = self._transitions.get_entry((state, action))

        if res is None:
            raise AssertionError("No transition data found for this case")
        else:
            return res.get_probability()

    def get_reward(self, state, action, next_state, step):

        if (state, action, next_state) not in self._rewards:
            raise AssertionError("No transition data found for this case")
        else:
            return self._rewards[(state, action, next_state)].get_mean()
