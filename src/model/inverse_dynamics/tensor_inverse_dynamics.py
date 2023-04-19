import torch
import torch.nn as nn
from utils.gumbel import gumbel_sample


class TensorInverseDynamics(nn.Module):

    NAME = "tensor-inv-dyn"

    def __init__(self, exp_setup, bootstrap_model=None):

        super(TensorInverseDynamics, self).__init__()

        self.temperature = 1.0
        self.action_dim = exp_setup.config["num_actions"]
        self.dim = exp_setup.constants["hidden_dim"]

        self.tensor_W = nn.Parameter(torch.randn(self.dim, self.action_dim, self.dim) * 0.01)

        if torch.cuda.is_available():
            self.cuda()

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

    def get_action_dim(self):
        return self.action_dim

    def get_latent_action(self, prev_encoding, obs_encoding):

        batch_size = prev_encoding.size(0)

        x = torch.matmul(prev_encoding,
                         self.tensor_W.view(self.dim, self.action_dim * self.dim))
        x = x.view(batch_size, self.action_dim, self.dim)

        x = (obs_encoding[:, None, :] * x).sum(2)           # batch x num_actions

        # Compute probability using Gumbel softmax
        prob, log_prob = gumbel_sample(x, self.temperature)

        return prob, log_prob
