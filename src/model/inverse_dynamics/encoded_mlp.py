import torch
import torch.nn as nn
from utils.gumbel import gumbel_sample


class EncodedMLP(nn.Module):
    NAME = "encoded-mlp"

    def __init__(self, exp_setup, bootstrap_model=None):
        super(EncodedMLP, self).__init__()

        self.temperature = 1.0
        self.action_dim = exp_setup.config["num_actions"]
        self.dim = exp_setup.constants["hidden_dim"]

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.action_dim),
        )

        self.mlp_h = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim), nn.LeakyReLU(), nn.Linear(self.dim, 256)
        )

        if torch.cuda.is_available():
            self.cuda()

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

    def get_action_dim(self):
        return self.action_dim

    def get_latent_action(self, prev_encoding, obs_encoding):
        x = torch.cat([prev_encoding, obs_encoding], dim=1)  # batch x (2 dim)
        logits = self.mlp(x)  # batch x action_dim

        # Compute probability using Gumbel softmax
        prob, log_prob = gumbel_sample(logits, self.temperature)

        h = self.mlp_h(x)

        return prob, log_prob, h
