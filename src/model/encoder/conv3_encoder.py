import torch
import torch.nn as nn

from utils.conv_util import get_conv_out_size


class Conv3Encoder(nn.Module):
    """
    Recommended convolution for gridworld
    """

    NAME = "conv3"

    def __init__(self, height, width, channel, out_dim, bootstrap_model=None):
        super(Conv3Encoder, self).__init__()

        self.height = height
        self.channel = channel
        self.width = width

        self.out_dim = out_dim

        # Note that the dynamic size below is calculated based on the model. If the model changes, then the size
        # will also change
        kernel_size1 = (5, 5)
        stride1 = (2, 2)

        kernel_size2 = (5, 5)
        stride2 = (2, 2)

        kernel_size3 = (5, 5)
        stride3 = (1, 1)

        dynamic_size_h1, dynamic_size_w1 = get_conv_out_size(self.height, self.width, kernel_size=kernel_size1, stride=stride1)

        dynamic_size_h2, dynamic_size_w2 = get_conv_out_size(
            dynamic_size_h1, dynamic_size_w1, kernel_size=kernel_size2, stride=stride2
        )

        dynamic_size_h3, dynamic_size_w3 = get_conv_out_size(
            dynamic_size_h2, dynamic_size_w2, kernel_size=kernel_size3, stride=stride3
        )

        self.n_channels_out = 32
        self.dynamic_size = dynamic_size_h3 * dynamic_size_w3 * self.n_channels_out

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=kernel_size1, stride=stride1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size2, stride=stride2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, self.n_channels_out, kernel_size=kernel_size3, stride=stride3),
            nn.BatchNorm2d(self.n_channels_out),
            nn.Flatten(),
            nn.Linear(self.dynamic_size, out_dim),
        )

        if torch.cuda.is_available():
            self.cuda()

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

    def forward(self, img):
        return self.encode(img)

    def encode(self, img):
        # print(img.shape)
        return self.model(img)
