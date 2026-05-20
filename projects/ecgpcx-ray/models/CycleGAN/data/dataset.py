from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset


class CycleGANDataset(Dataset):

    def __init__(
        self,
        healthy_dir,
        pneumonia_dir,
        transform=None
    ):

        self.healthy_paths = sorted(
            list(Path(healthy_dir).glob("*"))
        )

        self.pneumonia_paths = sorted(
            list(Path(pneumonia_dir).glob("*"))
        )

        self.transform = transform

    def __len__(self):

        return max(
            len(self.healthy_paths),
            len(self.pneumonia_paths)
        )

    def __getitem__(self, idx):

        healthy_path = self.healthy_paths[
            idx % len(self.healthy_paths)
        ]

        pneumonia_path = random.choice(
            self.pneumonia_paths
        )

        healthy_image = Image.open(
            healthy_path
        ).convert("L")

        pneumonia_image = Image.open(
            pneumonia_path
        ).convert("L")

        if self.transform:

            healthy_image = self.transform(
                healthy_image
            )

            pneumonia_image = self.transform(
                pneumonia_image
            )

        return {
            "healthy": healthy_image,
            "pneumonia": pneumonia_image
        }