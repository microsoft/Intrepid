import torch
import torch.nn as nn

from model.misc.robot_car.positional_encoding import positionalencoding1d


class ImageToPatches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.P = patch_size

    def forward(self, x):
        P = self.P
        B, C, H, W = x.shape  # [B,C,H,W]                 4D Image
        x = x.reshape(B, C, H // P, P, W // P, P)  # [B,C, H//P, P, W//P, P]   6D Patches
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H//P, W//P, C, P, P]  6D Swap Axes
        x = x.reshape(B, H // P * W // P, C * P * P)  # [B, H//P * W//P, C*P*P]   3D Patches
        # [B, n_tokens, n_pixels]
        return x


class PerPatchMLP(nn.Module):
    def __init__(self, n_pixels, n_channel):
        super().__init__()
        self.mlp = nn.Linear(n_pixels, n_channel)

    def forward(self, x):
        return self.mlp(x)  # x*w:  [B, n_tokens, n_pixels] x [n_pixels, n_channel]
        #       [B, n_tokens, n_channel]


class TokenMixingMLP(nn.Module):
    def __init__(self, n_tokens, n_channel, n_hidden):
        super().__init__()
        self.layer_norm = nn.LayerNorm([n_tokens, n_channel])
        self.mlp1 = nn.Linear(n_tokens, n_hidden)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(n_hidden, n_tokens)

    def forward(self, X):
        z = self.layer_norm(X)  # z:    [B, n_tokens, n_channel]
        z = z.permute(0, 2, 1)  # z:    [B, n_channel, n_tokens]
        z = self.gelu(self.mlp1(z))  # z:    [B, n_channel, n_hidden]
        z = self.mlp2(z)  # z:    [B, n_channel, n_tokens]
        z = z.permute(0, 2, 1)  # z:    [B, n_tokens, n_channel]
        U = X + z  # U:    [B, n_tokens, n_channel]
        return U


class ChannelMixingMLP(nn.Module):
    def __init__(self, n_tokens, n_channel, n_hidden):
        super().__init__()
        self.layer_norm = nn.LayerNorm([n_tokens, n_channel])
        self.mlp3 = nn.Linear(n_channel, n_hidden)
        self.gelu = nn.GELU()
        self.mlp4 = nn.Linear(n_hidden, n_channel)

    def forward(self, U):
        z = self.layer_norm(U)  # z: [B, n_tokens, n_channel]
        z = self.gelu(self.mlp3(z))  # z: [B, n_tokens, n_hidden]
        z = self.mlp4(z)  # z: [B, n_tokens, n_channel]
        Y = U + z  # Y: [B, n_tokens, n_channel]
        return Y


class OutputMLP(nn.Module):
    def __init__(self, n_tokens, n_channel, n_output):
        super().__init__()
        self.layer_norm = nn.LayerNorm([n_tokens, n_channel])
        self.out_mlp = nn.Linear(n_channel, n_output)

    def forward(self, x):
        x = self.layer_norm(x)  # x: [B, n_tokens, n_channel]
        x = x.mean(dim=1)  # x: [B, n_channel]
        return self.out_mlp(x)  # x: [B, n_output]


class MLP_Mixer(nn.Module):
    def __init__(self, n_layers, n_channel, n_hidden, n_output, image_size_h, image_size_w, patch_size,
                 n_image_channel):
        super().__init__()

        n_tokens = (image_size_h // patch_size) * (image_size_w // patch_size)
        n_pixels = n_image_channel * patch_size ** 2

        self.ImageToPatch = ImageToPatches(patch_size=patch_size)
        self.PerPatchMLP = PerPatchMLP(n_pixels, n_channel)
        self.MixerStack = nn.Sequential(*[
            nn.Sequential(
                TokenMixingMLP(n_tokens, n_channel, n_hidden),
                ChannelMixingMLP(n_tokens, n_channel, n_hidden)
            ) for _ in range(n_layers)
        ])
        self.OutputMLP = OutputMLP(n_tokens, n_channel, n_output)

        self.pe = positionalencoding1d(n_channel, n_tokens).unsqueeze(0)

    def forward(self, x):
        x = self.ImageToPatch(x)
        x = self.PerPatchMLP(x)
        # pe_use = self.pe.repeat(x.shape[0], 1, 1)
        # x += pe_use
        x = self.MixerStack(x)
        return self.OutputMLP(x)

    def to(self, device):
        self.ImageToPatch = self.ImageToPatch.to(device)
        self.PerPatchMLP = self.PerPatchMLP.to(device)
        self.MixerStack = self.MixerStack.to(device)
        self.OutputMLP = self.OutputMLP.to(device)
        self.pe = self.pe.to(device)
        return self


if __name__ == "__main__":
    mix = MLP_Mixer(n_layers=2, n_channel=32, n_hidden=32, n_output=256, image_size_h=100 * 3, image_size_w=100 * 1,
                    patch_size=10, n_image_channel=3)
    batch_size = 128
    x = torch.randn(batch_size, 3, 100 * 3, 100)
    y = mix(x)
    print(x.shape, y.shape)