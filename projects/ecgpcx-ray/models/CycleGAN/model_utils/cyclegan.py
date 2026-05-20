from .generators import Generator
from .discriminators import Discriminator


class CycleGAN:
    """
    Wrapper class for CycleGAN architecture.
    """

    def __init__(self, image_channels=1):

        # Generators
        self.G_H2P = Generator(
            input_channels=image_channels,
            output_channels=image_channels
        )

        self.G_P2H = Generator(
            input_channels=image_channels,
            output_channels=image_channels
        )

        # Discriminators
        self.D_H = Discriminator(
            input_channels=image_channels
        )

        self.D_P = Discriminator(
            input_channels=image_channels
        )