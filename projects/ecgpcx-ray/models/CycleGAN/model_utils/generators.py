import torch
import torch.nn as nn

from .blocks import ResidualBlock


class Generator(nn.Module):
    """
    CycleGAN ResNet Generator

    Architecture:
    c7s1-64
    d128
    d256
    9 x Residual Blocks
    u128
    u64
    c7s1-3
    """

    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        n_residuals=6
    ):
        super().__init__()

        model = []

        # Initial convolution block
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2

        for _ in range(2):
            model += [
                nn.Conv2d(
                    in_features,
                    out_features,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]

            in_features = out_features
            out_features *= 2

        # Residual blocks
        for _ in range(n_residuals):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2

        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]

            in_features = out_features
            out_features //= 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, output_channels, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)