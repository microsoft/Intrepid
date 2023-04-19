import torch
import torch.nn as nn


class Conv2Encoder(nn.Module):

    NAME = "conv2"

    def __init__(self, height, width, channel, out_dim, bootstrap_model=None):

        super(Conv2Encoder, self).__init__()

        self.height = height
        self.channel = channel
        self.width = width

        self.out_dim = out_dim

        self.model = nn.Sequential(
            nn.Conv2d(self.channel, 16, (8, 8), 4),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, (4, 4), 2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, (4, 4), 1),
            nn.Flatten(),
            nn.Linear(128, out_dim)
        )

        if torch.cuda.is_available():
            self.cuda()

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

    def forward(self, img):
        return self.encode(img)

    def encode(self, img):
        return self.model(img)
