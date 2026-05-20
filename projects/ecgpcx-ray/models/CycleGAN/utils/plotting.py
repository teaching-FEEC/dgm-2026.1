from pathlib import Path

import matplotlib.pyplot as plt


def plot_losses(
    history,
    save_dir="outputs/losses",
    filename="loss_curve.png"
):

    Path(save_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(12, 6))

    for loss_name, values in history.items():

        plt.plot(
            values,
            label=loss_name,
            linewidth=2
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "CycleGAN Training Losses"
    )

    plt.yscale("log")

    plt.grid(True)

    plt.legend()

    save_path = (
        Path(save_dir) /
        filename
    )

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()