import torch
import torch.nn as nn


class ConvForwardModel(nn.Module):
    NAME = "ConvForward"

    def __init__(self, exp_setup, bootstrap_model=None):
        super(ConvForwardModel, self).__init__()

        self.num_actions = exp_setup.config["num_actions"]
        self.latent_action_vec_dim = 256
        self.height, self.width, self.channel = exp_setup.config["obs_dim"]
        self.encoder_dim = exp_setup.constants["hidden_dim"]
        # self.reshape_layer = nn.Linear(self.encoder_dim + self.num_actions, 32 * 2 * 2)
        self.reshape_layer = nn.Linear(
            self.encoder_dim + self.latent_action_vec_dim, 256 * 2 * 2
        )

        self.model = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
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
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.Upsample(size=(56, 56)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, self.channel, 3, stride=1, padding=1),
        )

        if torch.cuda.is_available():
            self.cuda()

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

    def forward(self, obs_encoding, latent_action_vec):
        batch = obs_encoding.size(0)
        vec = torch.cat([obs_encoding, latent_action_vec], dim=1)  # batch x dim
        vec = self.reshape_layer(vec).resize(batch, 256, 2, 2)
        return self.model(vec)
