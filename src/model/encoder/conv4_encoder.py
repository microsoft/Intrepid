import torch
import torch.nn as nn

from utils.conv_util import get_conv_out_size


class Conv4Encoder(nn.Module):
    NAME = "conv4"

    def __init__(self, height, width, channel, out_dim, bootstrap_model=None):
        super(Conv4Encoder, self).__init__()

        self.height = height
        self.channel = channel
        self.width = width

        self.out_dim = out_dim

        # Note that the dynamic size below is calculated based on the model. If the model changes, then the size
        # will also change
        kernel_size1 = (8, 8)
        stride1 = (4, 4)

        kernel_size2 = (4, 4)
        stride2 = (2, 2)

        dynamic_size_h1, dynamic_size_w1 = get_conv_out_size(
            self.height, self.width, kernel_size=kernel_size1, stride=stride1
        )

        dynamic_size_h2, dynamic_size_w2 = get_conv_out_size(
            dynamic_size_h1, dynamic_size_w1, kernel_size=kernel_size2, stride=stride2
        )

        self.n_channels_out = 32
        self.dynamic_size = dynamic_size_h2 * dynamic_size_w2 * self.n_channels_out

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(16, self.n_channels_out, (4, 4), 2),
            nn.ReLU(),
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
