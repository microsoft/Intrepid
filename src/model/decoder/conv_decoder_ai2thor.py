import torch
import torch.nn as nn


class ConvDecoderAI2Thor(nn.Module):

    NAME = "conv-ai2thor"

    def __init__(self, height, width, channel, out_dim, bootstrap_model=None):

        super(ConvDecoderAI2Thor, self).__init__()

        self.height = height
        self.channel = channel
        self.width = width

        self.out_dim = out_dim

        self.linear_layer = nn.Linear(out_dim, 32 * 2 * 4)

        self.model = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(32, 128, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16,3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8,3, stride=1, padding=1),
            nn.Upsample(size=(56, 56 * 2)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, self.channel, 3, stride=1, padding=1)
        )

        if torch.cuda.is_available():
            self.cuda()

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

    def forward(self, vec):
        return self.decode(vec)

    def decode(self, vec):

        batch_size = vec.size(0)
        out = self.linear_layer(vec).view(batch_size, 32, 2, 4)
        # print(out.shape)
        # raise Exception('done')
        return self.model(out)
