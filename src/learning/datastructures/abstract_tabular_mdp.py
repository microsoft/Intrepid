class AbstractTabularMDP:
    def __init__(self, actions, horizon, gamma=1.0):
        self.actions = actions
        self.horizon = horizon
        self.gamma = gamma

    def get_states(self, timestep):
        raise NotImplementedError()

    def get_transitions(self, state, action):
        raise NotImplementedError()

    def get_reward(self, state, action, next_state, step):
        raise NotImplementedError()
