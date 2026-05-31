from pathlib import Path

import matplotlib.pyplot as plt
import torch

from torchvision.utils import make_grid


def denormalize(x):

    return (x * 0.5) + 0.5


def save_training_progress(
    images,
    epoch,
    save_dir="outputs/progress",
    max_images=4
):

    Path(save_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    real_H = denormalize(
        images["real_healthy"][:max_images]
    )

    fake_P = denormalize(
        images["fake_pneumonia"][:max_images]
    )

    rec_H = denormalize(
        images["recovered_healthy"][:max_images]
    )

    real_P = denormalize(
        images["real_pneumonia"][:max_images]
    )

    fake_H = denormalize(
        images["fake_healthy"][:max_images]
    )

    rec_P = denormalize(
        images["recovered_pneumonia"][:max_images]
    )

    healthy_row = torch.cat(
        [real_H, fake_P, rec_H],
        dim=0
    )

    pneumonia_row = torch.cat(
        [real_P, fake_H, rec_P],
        dim=0
    )

    grid = make_grid(
        torch.cat(
            [healthy_row, pneumonia_row],
            dim=0
        ),
        nrow=max_images,
        padding=2
    )

    plt.figure(figsize=(12, 8))

    plt.imshow(
        grid.permute(1, 2, 0).squeeze(),
        cmap="gray"
    )

    plt.axis("off")

    plt.title(
        f"Epoch {epoch}"
    )

    save_path = (
        Path(save_dir) /
        f"epoch_{epoch:03d}.png"
    )

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()