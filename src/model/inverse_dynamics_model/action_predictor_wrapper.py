from model.inverse_dynamics_model.action_predictor import (
    ActionPredictor,
    ActionPredictorFlatNN,
    ActionPredictorCNN1,
    ActionPredictorCNN2,
    ActionPredictorCNN3,
    ActionPredictorCNN4,
    ActionPredictorCNN5,
    ActionPredictorCNN6,
    ActionPredictorCNN7,
)


class InverseDynamicsWrapper:
    FF = range(1)

    def __init__(self):
        pass

    @staticmethod
    def get_model(config, constants, bootstrap_model=None):
        model_type_str = constants["model_type"]
        if model_type_str == "ff":
            return ActionPredictor(config, constants, bootstrap_model)
        elif model_type_str == "flat":
            return ActionPredictorFlatNN(config, constants, bootstrap_model)
        elif model_type_str == "conv1":
            return ActionPredictorCNN1(config, constants, bootstrap_model)
        elif model_type_str == "conv2":
            return ActionPredictorCNN2(config, constants, bootstrap_model)
        elif model_type_str == "conv3":
            return ActionPredictorCNN3(config, constants, bootstrap_model)
        elif model_type_str == "conv4":
            return ActionPredictorCNN4(config, constants, bootstrap_model)
        elif model_type_str == "conv5":
            return ActionPredictorCNN5(config, constants, bootstrap_model)
        elif model_type_str == "conv6":
            return ActionPredictorCNN6(config, constants, bootstrap_model)
        elif model_type_str == "conv7":
            return ActionPredictorCNN7(config, constants, bootstrap_model)
        else:
            raise AssertionError("Unhandled model type %r" % model_type_str)
