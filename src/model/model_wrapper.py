from model.decoder.decoder_wrapper import DecoderModelWrapper
from model.encoder.encoder_wrapper import EncoderModelWrapper
from model.classifiers.classifier_model_wrapper import ClassifierModelWrapper


class ModelWrapper:

    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, model_name, config, constants, bootstrap_model=None, **kwargs):

        if model_type == "classifier":
            return ClassifierModelWrapper.get_classifier(model_name=model_name,
                                                         num_class=kwargs["num_class"],
                                                         config=config,
                                                         constants=constants,
                                                         bootstrap_model=None)

        elif model_type == "encoder":
            return EncoderModelWrapper.get_encoder(model_name=model_name,
                                                   bootstrap_model=bootstrap_model,
                                                   **kwargs)

        elif model_type == "decoder":
            return DecoderModelWrapper.get_decoder(model_name=model_name,
                                                   bootstrap_model=bootstrap_model,
                                                   **kwargs)

        else:
            raise NotImplementedError()