import torch
import torch.nn as nn


class FeedForwardDecoder(nn.Module):
    NAME = "ff"

    def __init__(self, num_inputs, inp_dim, out_dim, hidden_dim, bootstrap_model=None):
        super(FeedForwardDecoder, self).__init__()

        self.num_inputs = num_inputs
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.out_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.num_inputs * self.inp_dim),
        )

        if torch.cuda.is_available():
            self.cuda()

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

    def forward(self, vec):
        return self.decode(vec)

    def decode(self, vec):
        output = self.model(vec).data.cpu()
        outputs = []
        for i in range(0, self.num_inputs):
            outputs.append(output[i * self.inp_dim : (i + 1) * self.inp_dim])

        return outputs
