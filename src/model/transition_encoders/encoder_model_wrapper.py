from model.transition_encoders.encoder_model import *
from model.transition_encoders.compositional_encoder_model import *


class EncoderModelWrapper:
    """ Wrapper for encoder model """

    @staticmethod
    def get_encoder_model(model_type, config, constants, bootstrap_model=None):

        if model_type == "backwardmodel":
            return BackwardEncoderModel(config, constants, bootstrap_model)
        elif model_type == "forwardmodel":
            return ForwardEncoderModel(config, constants, bootstrap_model)
        elif model_type == "compbackwardmodel":
            return CompositionalEncoderModel(config, constants, bootstrap_model)
        else:
            raise NotImplementedError("Did not implement %r" % model_type)
