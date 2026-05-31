import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    70x70 PatchGAN discriminator.
    """

    def __init__(self, input_channels=1):
        super().__init__()

        self.model = nn.Sequential(

            # C64
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # C128
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # C256
            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # C512
            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=1,
                padding=1
            ),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Output PatchGAN
            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        return self.model(x)