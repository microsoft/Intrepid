import torch
import torch.nn as nn


class FeedForwardEncoder(nn.Module):

    NAME = "ff"

    def __init__(self, num_inputs, inp_dim, out_dim, hidden_dim, bootstrap_model=None):

        super(FeedForwardEncoder, self).__init__()

        self.num_inputs = num_inputs
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.num_inputs * self.inp_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )

        if torch.cuda.is_available():
            self.cuda()

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

    def forward(self, **inputs):
        return self.encode(**inputs)

    def encode(self, **inputs):

        vec = torch.cat(inputs, dim=1)
        return self.model(vec)
