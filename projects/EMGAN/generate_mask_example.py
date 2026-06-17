"""
Generate a side-by-side lung mask example image for the README.

Usage (from projects/ecgpcx-ray/):
    python generate_mask_example.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np

IMG_PATH = Path("data/processed/train/healthy/00000205_000.png")
MASK_PATH = Path("data/processed/train/healthy_masks/00000205_000.png")
OUT_PATH = Path("images/lung_mask_example.png")


def main():
    img = Image.open(IMG_PATH).convert("L")
    mask = Image.open(MASK_PATH).convert("L")

    img_arr = np.asarray(img)
    mask_arr = np.asarray(mask)

    # Create an RGB overlay: lung region tinted blue on the original image
    img_rgb = np.stack([img_arr] * 3, axis=-1)
    overlay = img_rgb.copy()
    lung_px = mask_arr > 127
    overlay[lung_px, 0] = np.clip(img_arr[lung_px].astype(int) * 0.4, 0, 255).astype(np.uint8)
    overlay[lung_px, 1] = np.clip(img_arr[lung_px].astype(int) * 0.6, 0, 255).astype(np.uint8)
    overlay[lung_px, 2] = np.clip(img_arr[lung_px].astype(int) * 1.0 + 80, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_arr, cmap="gray")
    axes[0].set_title("Original X-ray", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(mask_arr, cmap="gray")
    axes[1].set_title("Lung Mask (binary)", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=13)
    axes[2].axis("off")
    patch = mpatches.Patch(color=(0.2, 0.4, 1.0), label="Lung region")
    axes[2].legend(handles=[patch], loc="lower right", fontsize=9)

    fig.suptitle(
        "TorchXRayVision PSPNet lung segmentation",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
