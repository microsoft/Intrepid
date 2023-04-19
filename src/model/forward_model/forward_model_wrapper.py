from model.forward_model.conv_forward_model import ConvForwardModel


class ForwardDynamicsWrapper:
    """ Wrapper for forward dynamics models """

    @staticmethod
    def get_forward_dynamics_model(model_name, bootstrap_model=None, **kwargs):

        models = [ConvForwardModel.NAME]

        if model_name == ConvForwardModel.NAME:
            return ConvForwardModel(**kwargs, bootstrap_model=bootstrap_model)

        else:
            raise NotImplementedError("Model %s is not implemented. Implemented models are linear, %r" %
                                      (model_name, models))
