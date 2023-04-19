class AbstractEncoderSampler:

    def __init__(self):
        pass

    @staticmethod
    def gather_samples(num_samples, env, actions, step, homing_policies):
        """ Gather samples given the environment, action space, the step at which the sample has to be
        gathered and the homing policies for the given step """
        raise NotImplementedError()