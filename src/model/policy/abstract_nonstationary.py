class AbstractNonStationaryPolicy:

    def __init__(self):
        pass

    def action_type(self):
        """
        :return: Type of action returned by the policy
        """
        raise NotImplementedError()

    def sample_action(self, observation, timestep):
        """
        :param observation: Observation of the world
        :param timestep: time step at which observation is observed
        :return: a numpy vector denoting the probability distribution over actions
        """
        raise NotImplementedError()

    def get_argmax_action(self, observation, timestep):
        """
        :param observation: Observation of the world
        :param timestep: time step at which observation is observed
        :return: action representation (can be integer, real number, real-valued vector, or object of some class)
        """
        raise NotImplementedError()
