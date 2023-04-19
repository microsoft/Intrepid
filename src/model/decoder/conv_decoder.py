import torch
import torch.nn as nn


class ConvDecoder(nn.Module):

    NAME = "conv"

    def __init__(self, height, width, channel, out_dim, bootstrap_model=None):

        super(ConvDecoder, self).__init__()

        self.height = height
        self.channel = channel
        self.width = width

        self.out_dim = out_dim

        self.linear_layer = nn.Linear(out_dim, 32 * 2 * 2)

        self.model = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, (2, 2), 2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 16, (4, 4), 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 16, (4, 4), 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, self.channel, (6, 6), 2)
        )

        if torch.cuda.is_available():
            self.cuda()

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

    def forward(self, vec):
        return self.decode(vec)

    def decode(self, vec):

        batch_size = vec.size(0)
        out = self.linear_layer(vec).view(batch_size, 32, 2, 2)
        return self.model(out)
