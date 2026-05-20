from torch.utils.data import DataLoader

from data.dataset import CycleGANDataset

from .transforms import (
    get_train_transforms,
    get_val_transforms
)


def get_train_dataloader(
    healthy_dir,
    pneumonia_dir,
    batch_size=4,
    image_size=128,
    num_workers=4
):

    dataset = CycleGANDataset(
        healthy_dir=healthy_dir,
        pneumonia_dir=pneumonia_dir,
        transform=get_train_transforms(
            image_size=image_size
        )
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


def get_val_dataloader(
    healthy_dir,
    pneumonia_dir,
    batch_size=4,
    image_size=256,
    num_workers=4
):

    dataset = CycleGANDataset(
        healthy_dir=healthy_dir,
        pneumonia_dir=pneumonia_dir,
        transform=get_val_transforms(
            image_size=image_size
        )
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )