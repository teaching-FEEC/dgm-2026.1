"""
Visualization of CycleGAN image generation for the README.

Creates a two-row figure:
  Row 1 – Healthy → Pneumonia: N pairs of (Real Healthy | Generated Pneumonia)
  Row 2 – Pneumonia → Healthy: N pairs of (Real Pneumonia | Generated Healthy)

Usage:
    python visualize_generation.py \\
        --checkpoint checkpoints/epoch_199.pt \\
        --healthy_dir   ../../data/processed/test/healthy \\
        --pneumonia_dir ../../data/processed/test/pneumonia \\
        [--n_examples 4] [--image_size 128] [--seed 42] [--device cpu]
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from model_utils.cyclegan import CycleGAN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def load_images(directory: str | Path, n: int, transform, seed: int) -> torch.Tensor:
    paths = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")
    rng = random.Random(seed)
    selected = rng.sample(paths, min(n, len(paths)))
    return torch.stack([transform(Image.open(p).convert("L")) for p in selected])


def to_displayable(tensor: torch.Tensor) -> "np.ndarray":
    """[-1, 1] tensor (1, H, W) → float32 numpy array [0, 1] (H, W)."""
    import numpy as np
    return np.clip((tensor.squeeze(0).numpy() * 0.5) + 0.5, 0, 1)


def style_axis(ax, spine_color: str, linewidth: float = 2.5) -> None:
    """Color the border of an axis and remove ticks."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
        spine.set_linewidth(linewidth)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate README visualization for CycleGAN translations."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the .pt checkpoint file.")
    parser.add_argument("--healthy_dir", required=True,
                        help="Directory with test healthy X-ray images.")
    parser.add_argument("--pneumonia_dir", required=True,
                        help="Directory with test pneumonia X-ray images.")
    parser.add_argument("--n_examples", type=int, default=4,
                        help="Number of example pairs per row (default: 4).")
    parser.add_argument("--image_size", type=int, default=128,
                        help="Spatial size to resize images (default: 128).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for image selection (default: 42).")
    parser.add_argument("--output_dir", default="outputs/img_generation",
                        help="Directory to save the output PNG.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    n = args.n_examples

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = CycleGAN(image_channels=1)
    model.G_H2P.load_state_dict(ckpt["G_H2P"])
    model.G_P2H.load_state_dict(ckpt["G_P2H"])
    G_H2P = model.G_H2P.to(device).eval()
    G_P2H = model.G_P2H.to(device).eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')} checkpoint.")

    # ------------------------------------------------------------------
    # Load test images
    # ------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    healthy_imgs   = load_images(args.healthy_dir,   n, transform, args.seed).to(device)
    pneumonia_imgs = load_images(args.pneumonia_dir, n, transform, args.seed).to(device)

    # ------------------------------------------------------------------
    # Generate translations
    # ------------------------------------------------------------------
    with torch.no_grad():
        gen_pneumonia = G_H2P(healthy_imgs)    # Healthy  → Pneumonia
        gen_healthy   = G_P2H(pneumonia_imgs)  # Pneumonia → Healthy

    healthy_imgs   = healthy_imgs.cpu()
    pneumonia_imgs = pneumonia_imgs.cpu()
    gen_pneumonia  = gen_pneumonia.cpu()
    gen_healthy    = gen_healthy.cpu()

    # ------------------------------------------------------------------
    # Build figure
    #
    # Layout (each cell = one image):
    #
    #   INPUT      GENERATED   INPUT      GENERATED   ...   ← column headers (row 0)
    #   Real H1 |  Gen P1  |  Real H2 |  Gen P2  |   ...   ← row 0: H→P
    #   Real P1 |  Gen H1  |  Real P2 |  Gen H2  |   ...   ← row 1: P→H
    #
    # Row labels shown as the y-axis label of the leftmost axis of each row.
    # Blue border  → input image (real)
    # Orange border → generated image (translation output)
    # ------------------------------------------------------------------

    INPUT_COLOR     = "#2c7bb6"   # steel-blue for real / input images
    GENERATED_COLOR = "#d7191c"   # red for generated / output images
    LABEL_COLOR_H2P = "#2c7bb6"
    LABEL_COLOR_P2H = "#d7191c"

    img_px   = args.image_size
    cell_in  = img_px / 72        # inches per cell (72 dpi baseline)
    cell_h   = cell_in * 1.3

    fig_w = 2 * n * cell_in * 4 + 1.6   # 4× upscale + label margin
    fig_h = 2 * cell_h * 4 + 1.2

    fig, axes = plt.subplots(
        2, 2 * n,
        figsize=(fig_w, fig_h),
    )

    # Adjust layout: leave room on the left for the row-label text
    fig.subplots_adjust(
        left=0.10, right=0.98,
        top=0.88, bottom=0.04,
        wspace=0.06, hspace=0.14,
    )

    # ── Column headers (only displayed on row 0) ──────────────────────
    for i in range(n):
        axes[0, 2 * i].set_title(
            "Input", fontsize=10, fontweight="bold",
            color=INPUT_COLOR, pad=5,
        )
        axes[0, 2 * i + 1].set_title(
            "Generated", fontsize=10, fontweight="bold",
            color=GENERATED_COLOR, pad=5,
        )

    # ── Row 0: Healthy → Pneumonia ────────────────────────────────────
    for i in range(n):
        axes[0, 2 * i].imshow(to_displayable(healthy_imgs[i]),
                               cmap="gray", vmin=0, vmax=1)
        axes[0, 2 * i + 1].imshow(to_displayable(gen_pneumonia[i]),
                                   cmap="gray", vmin=0, vmax=1)
        style_axis(axes[0, 2 * i],     INPUT_COLOR)
        style_axis(axes[0, 2 * i + 1], GENERATED_COLOR)

    # Row label
    axes[0, 0].set_ylabel(
        "Healthy → Pneumonia",
        fontsize=12, fontweight="bold", color=LABEL_COLOR_H2P,
        labelpad=10, rotation=90, va="center",
    )

    # ── Row 1: Pneumonia → Healthy ────────────────────────────────────
    for i in range(n):
        axes[1, 2 * i].imshow(to_displayable(pneumonia_imgs[i]),
                               cmap="gray", vmin=0, vmax=1)
        axes[1, 2 * i + 1].imshow(to_displayable(gen_healthy[i]),
                                   cmap="gray", vmin=0, vmax=1)
        style_axis(axes[1, 2 * i],     INPUT_COLOR)
        style_axis(axes[1, 2 * i + 1], GENERATED_COLOR)

    # Row label
    axes[1, 0].set_ylabel(
        "Pneumonia → Healthy",
        fontsize=12, fontweight="bold", color=LABEL_COLOR_P2H,
        labelpad=10, rotation=90, va="center",
    )

    # ── Legend patches ─────────────────────────────────────────────────
    import matplotlib.patches as mpatches
    input_patch     = mpatches.Patch(color=INPUT_COLOR,     label="Input (Real)")
    generated_patch = mpatches.Patch(color=GENERATED_COLOR, label="Generated (CycleGAN)")
    fig.legend(
        handles=[input_patch, generated_patch],
        loc="upper right", fontsize=10, framealpha=0.9,
        bbox_to_anchor=(0.98, 0.97),
    )

    # ── Title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "CycleGAN — Counterfactual Image Generation",
        fontsize=14, fontweight="bold", y=0.96,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "generation_examples.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    main()
