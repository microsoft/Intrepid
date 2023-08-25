from model.inverse_dynamics.encoded_mlp import EncodedMLP
from model.inverse_dynamics.simple_feed_forward import SimpleFeedForwardIK
from model.inverse_dynamics.tensor_inverse_dynamics import TensorInverseDynamics


class InverseDynamicsWrapper:
    """Wrapper for inverse dynamics models"""

    @staticmethod
    def get_inv_dynamics_model(model_name, bootstrap_model=None, **kwargs):
        models = [EncodedMLP.NAME, SimpleFeedForwardIK.NAME, TensorInverseDynamics.NAME]

        if model_name == EncodedMLP.NAME:
            return EncodedMLP(**kwargs, bootstrap_model=bootstrap_model)

        elif model_name == SimpleFeedForwardIK.NAME:
            return SimpleFeedForwardIK(**kwargs, bootstrap_model=bootstrap_model)

        elif model_name == TensorInverseDynamics.NAME:
            return TensorInverseDynamics(**kwargs, bootstrap_model=bootstrap_model)

        else:
            raise NotImplementedError("Model %s is not implemented. Implemented models are linear, %r" % (model_name, models))
