import random
import torch


class ImageBuffer:
    """
    Replay buffer from CycleGAN paper.
    Helps stabilize discriminator training.
    """

    def __init__(self, max_size=50):

        assert max_size > 0

        self.max_size = max_size
        self.data = []

    def push_and_pop(self, images):

        output = []

        for image in images:

            image = torch.unsqueeze(image.detach(), 0)

            if len(self.data) < self.max_size:

                self.data.append(image)
                output.append(image)

            else:

                if random.random() > 0.5:

                    idx = random.randint(0, self.max_size - 1)

                    old_image = self.data[idx].clone()

                    self.data[idx] = image

                    output.append(old_image)

                else:
                    output.append(image)

        return torch.cat(output)