from model.classifiers.convm_classifier import ConvMClassifier
from model.classifiers.conv3_classifier import Conv3Classifier
from model.classifiers.conv2_classifier import Conv2Classifier
from model.classifiers.feedforward_classifier import FeedForwardClassifier
from model.classifiers.linear_classifier import LinearClassifier


class ClassifierModelWrapper:
    """Wrapper for classification model"""

    @staticmethod
    def get_classifier(model_name, num_class, config, constants, bootstrap_model=None):
        if model_name == "linear":
            return LinearClassifier(num_class, config, constants, bootstrap_model)

        elif model_name == "ff":
            return FeedForwardClassifier(num_class, config, constants, bootstrap_model)

        elif model_name == "conv2":
            return Conv2Classifier(num_class, config, constants, bootstrap_model)

        elif model_name == "conv3":
            return Conv3Classifier(num_class, config, constants, bootstrap_model)

        elif model_name == "convm":
            return ConvMClassifier(num_class, config, constants, bootstrap_model)

        else:
            raise NotImplementedError(
                "Model %s is not implemented. Implemented models are linear, "
                "ff, conv2, covn3, and convm" % model_name
            )
