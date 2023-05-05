from model.policy.abstract_nonstationary import AbstractNonStationaryPolicy


class OpenLoopPolicy(AbstractNonStationaryPolicy):
    def __init__(self, actions=None, path_id=None):
        AbstractNonStationaryPolicy.__init__(self)

        # List of actions
        if actions is None:
            self._actions = []
        else:
            self._actions = list(actions)

        # ID of this path
        self.path_id = path_id

        # Last action that formed this path
        self.action = None if len(self._actions) == 0 else self._actions[-1]

        # ID of the parent
        self.parent_path_id = None

    def extend(self, action, path_id=None):
        policy = self.clone()
        policy._actions.append(action)

        policy.parent_path_id = policy.path_id
        policy.path_id = path_id
        policy.action = action

        return policy

    def num_timesteps(self):
        return len(self._actions)

    def action_type(self):
        """
        :return: Type of action returned by the policy
        """
        raise NotImplementedError()

    def sample_action(self, observation, timestep):
        return self._actions[timestep]

    def get_argmax_action(self, observation, timestep):
        return self._actions[timestep]

    def clone(self):
        policy = OpenLoopPolicy()
        policy._actions = list(self._actions)

        policy.path_id = self.path_id
        policy.action = self.action
        policy.parent_path_id = self.parent_path_id

        return policy

    def __str__(self):
        if self.parent_path_id is None or self.action is None or self.path_id is None:
            return "NA"
        else:
            return "[%d -> %r -> %d]" % (self.parent_path_id, self.action, self.path_id)
