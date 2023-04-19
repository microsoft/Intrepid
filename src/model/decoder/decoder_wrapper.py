from model.decoder.conv_decoder import ConvDecoder
from model.decoder.conv_decoder2 import ConvDecoder2
from model.decoder.conv_decoder_ai2thor import ConvDecoderAI2Thor
from model.decoder.feedforward_decoder import FeedForwardDecoder


class DecoderModelWrapper:
    """ Wrapper for decoder models """

    @staticmethod
    def get_decoder(model_name, bootstrap_model=None, **kwargs):

        models = [FeedForwardDecoder.NAME, ConvDecoder.NAME, ConvDecoder2.NAME]

        if model_name == FeedForwardDecoder.NAME:
            return FeedForwardDecoder(**kwargs, bootstrap_model=bootstrap_model)

        elif model_name == ConvDecoder.NAME:
            return ConvDecoder(**kwargs, bootstrap_model=bootstrap_model)

        elif model_name == ConvDecoder2.NAME:
            return ConvDecoder2(**kwargs, bootstrap_model=bootstrap_model)

        elif model_name == ConvDecoderAI2Thor.NAME:
            return ConvDecoderAI2Thor(**kwargs, bootstrap_model=bootstrap_model)

        else:
            raise NotImplementedError("Model %s is not implemented. Implemented models are linear, %r" %
                                      (model_name, models))
