from model.encoder.conv2_encoder import Conv2Encoder
from model.encoder.conv3_encoder import Conv3Encoder
from model.encoder.conv4_encoder import Conv4Encoder
from model.encoder.conv_encoder import ConvEncoder
from model.encoder.feedforward_encoder import FeedForwardEncoder


class EncoderModelWrapper:
    """Wrapper for encoder models"""

    @staticmethod
    def get_encoder(model_name, bootstrap_model=None, **kwargs):
        models = [
            FeedForwardEncoder,
            ConvEncoder,
            Conv2Encoder,
            Conv3Encoder,
            Conv4Encoder,
        ]
        model_names = [model.NAME for model in models]

        for model in models:
            if model_name == model.NAME:
                return model(**kwargs, bootstrap_model=bootstrap_model)

        raise NotImplementedError(
            "Model %s is not implemented. Implemented models are %s"
            % (model_name, model_names)
        )
