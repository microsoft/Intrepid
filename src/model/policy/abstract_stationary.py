class AbstractStationaryPolicy:

    def __init__(self):
        pass

    def action_type(self):
        """
        :return: Type of action returned by the policy
        """
        raise NotImplementedError()

    def sample_action(self, observation):
        """
        :param observation: Observation of the world
        :return: a numpy vector denoting the probability distribution over actions
        """
        raise NotImplementedError()

    def get_argmax_action(self, observation):
        """
        :param observation: Observation of the world
        :return: action representation (can be integer, real number, real-valued vector, or object of some class)
        """
        raise NotImplementedError()
