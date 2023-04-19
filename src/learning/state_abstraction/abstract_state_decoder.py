class AbstractStateDecoder:

    def __init__(self):
        pass

    def calc_loss(self, model, sample, **kwargs):
        """
        :model
        :sample
        :**kwargs
        """
        raise NotImplementedError()
