"""
Heatmap visualization of CycleGAN image changes — full test-set sweep.

Iterates over every image in the test set, grouped in batches of 2 healthy
and 2 pneumonia images, saving one PNG per batch.

Each PNG shows two rows of triplets:
    Row 1 – Healthy  → Pneumonia : [Real Healthy | Generated Pneumonia | Change Heatmap]
    Row 2 – Pneumonia → Healthy  : [Real Pneumonia | Generated Healthy  | Change Heatmap]

The heatmap is |real − generated| (Gaussian-smoothed) overlaid on the real
image so that anatomical context is preserved.

Usage:
    python visualize_heatmap.py \\
        --checkpoint checkpoints/epoch_199.pt \\
        --healthy_dir   ../../data/processed/test/healthy \\
        --pneumonia_dir ../../data/processed/test/pneumonia \\
        [--image_size 128] [--colormap hot] [--overlay_alpha 0.55] \\
        [--blur_sigma 2] [--output_dir outputs/img_generation] [--device cpu]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.cm as mplcm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from model_utils.cyclegan import CycleGAN


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXTS        = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
BATCH_SIZE  = 2          # images per class per output file

INPUT_COLOR = "#2c7bb6"
GEN_COLOR   = "#d7191c"
HEAT_COLOR  = "#4dac26"

ROW_LABELS = [
    ("Healthy\n→ Pneumonia", INPUT_COLOR),
    ("Pneumonia\n→ Healthy",  GEN_COLOR),
]
COL_TITLES = ["Input (Real)", "Generated", "Change Heatmap"]
COL_COLORS = [INPUT_COLOR, GEN_COLOR, HEAT_COLOR]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_paths(directory: str | Path) -> list[Path]:
    paths = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")
    return paths


def chunked(lst: list, size: int):
    """Yield successive non-overlapping chunks of `size`; skip the last incomplete one."""
    for i in range(0, len(lst) - size + 1, size):
        yield lst[i : i + size]


def load_batch(paths: list[Path], transform, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [transform(Image.open(p).convert("L")) for p in paths]
    ).to(device)


def to_gray(tensor: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor (1, H, W) → float32 numpy (H, W) in [0, 1]."""
    return np.clip(tensor.squeeze(0).numpy() * 0.5 + 0.5, 0, 1)


def make_overlay(
    real: np.ndarray,
    generated: np.ndarray,
    cmap_name: str,
    alpha: float,
    blur_sigma: float,
) -> np.ndarray:
    """
    Blend an absolute-difference heatmap over the real grayscale image.

    Returns an (H, W, 3) RGB array in [0, 1].
    """
    diff = np.abs(real - generated)
    if blur_sigma > 0:
        diff = gaussian_filter(diff, sigma=blur_sigma)
    diff_norm = diff / (diff.max() + 1e-8)

    cmap        = mplcm.get_cmap(cmap_name)
    heatmap_rgb = cmap(diff_norm)[..., :3]
    gray_rgb    = np.stack([real, real, real], axis=-1)
    return np.clip((1.0 - alpha) * gray_rgb + alpha * heatmap_rgb, 0, 1)


def style_axis(ax, spine_color: str, lw: float = 2.5) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
        spine.set_linewidth(lw)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def render_batch(
    healthy_imgs:   torch.Tensor,   # (2, 1, H, W)
    pneumonia_imgs: torch.Tensor,   # (2, 1, H, W)
    gen_pneumonia:  torch.Tensor,   # (2, 1, H, W)
    gen_healthy:    torch.Tensor,   # (2, 1, H, W)
    args,
    save_path: Path,
    batch_idx: int,
    n_batches: int,
) -> None:
    n = BATCH_SIZE  # 2

    row_data = [
        (healthy_imgs,   gen_pneumonia),
        (pneumonia_imgs, gen_healthy),
    ]

    # Pre-compute overlays
    overlays = []
    for real_batch, gen_batch in row_data:
        row_overlays = []
        for i in range(n):
            overlay = make_overlay(
                to_gray(real_batch[i]),
                to_gray(gen_batch[i]),
                cmap_name=args.colormap,
                alpha=args.overlay_alpha,
                blur_sigma=args.blur_sigma,
            )
            row_overlays.append(overlay)
        overlays.append(row_overlays)

    # ------------------------------------------------------------------
    # Layout: 2 rows × (3*n image cols + 1 colorbar col)
    # ------------------------------------------------------------------
    n_img_cols = 3 * n
    fig_w = 3.0 * n_img_cols + 0.6
    fig_h = 3.0 * 2 + 1.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(
        nrows=2, ncols=n_img_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_img_cols + [0.05],
        wspace=0.05, hspace=0.18,
        left=0.10, right=0.96,
        top=0.87,  bottom=0.04,
    )

    axes = np.array([
        [fig.add_subplot(gs[r, c]) for c in range(n_img_cols)]
        for r in range(2)
    ])
    cbar_ax = fig.add_subplot(gs[:, -1])

    # Column headers (only on row 0)
    for i in range(n):
        for k, (title, color) in enumerate(zip(COL_TITLES, COL_COLORS)):
            axes[0, 3 * i + k].set_title(
                title, fontsize=9, fontweight="bold", color=color, pad=5,
            )

    # Fill image cells
    for r, (real_batch, gen_batch) in enumerate(row_data):
        for i in range(n):
            real_np = to_gray(real_batch[i])
            gen_np  = to_gray(gen_batch[i])

            axes[r, 3 * i    ].imshow(real_np,        cmap="gray", vmin=0, vmax=1)
            axes[r, 3 * i + 1].imshow(gen_np,         cmap="gray", vmin=0, vmax=1)
            axes[r, 3 * i + 2].imshow(overlays[r][i])

            style_axis(axes[r, 3 * i    ], INPUT_COLOR)
            style_axis(axes[r, 3 * i + 1], GEN_COLOR)
            style_axis(axes[r, 3 * i + 2], HEAT_COLOR)

        axes[r, 0].set_ylabel(
            ROW_LABELS[r][0],
            fontsize=11, fontweight="bold", color=ROW_LABELS[r][1],
            labelpad=10, rotation=90, va="center",
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=mplcm.get_cmap(args.colormap),
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Normalised |Δ pixel|", fontsize=9, labelpad=6)
    cbar.ax.tick_params(labelsize=8)

    # Legend
    patches = [
        mpatches.Patch(color=INPUT_COLOR, label="Input (Real)"),
        mpatches.Patch(color=GEN_COLOR,   label="Generated (CycleGAN)"),
        mpatches.Patch(color=HEAT_COLOR,  label=f"Change Heatmap (α={args.overlay_alpha})"),
    ]
    fig.legend(handles=patches, loc="upper right", fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.95, 0.97))

    fig.suptitle(
        f"CycleGAN — Change Heatmap  [{batch_idx + 1}/{n_batches}]",
        fontsize=13, fontweight="bold", y=0.95,
    )

    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Heatmap of per-pixel changes for every image in the test set."
    )
    p.add_argument("--checkpoint",    required=True,
                   help="Path to .pt checkpoint file.")
    p.add_argument("--healthy_dir",   required=True,
                   help="Directory with test healthy X-ray images.")
    p.add_argument("--pneumonia_dir", required=True,
                   help="Directory with test pneumonia X-ray images.")
    p.add_argument("--image_size",    type=int,   default=128,
                   help="Resize images to this size (default: 128).")
    p.add_argument("--colormap",      default="hot",
                   help="Matplotlib colormap for the heatmap (default: hot).")
    p.add_argument("--overlay_alpha", type=float, default=0.55,
                   help="Heatmap opacity over the real image (default: 0.55).")
    p.add_argument("--blur_sigma",    type=float, default=2.0,
                   help="Gaussian blur sigma for the diff map (default: 2.0).")
    p.add_argument("--output_dir",    default="outputs/img_generation",
                   help="Directory to save output PNGs.")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = CycleGAN(image_channels=1)
    model.G_H2P.load_state_dict(ckpt["G_H2P"])
    model.G_P2H.load_state_dict(ckpt["G_P2H"])
    G_H2P = model.G_H2P.to(device).eval()
    G_P2H = model.G_P2H.to(device).eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')} checkpoint.")

    # ------------------------------------------------------------------
    # Collect all image paths and split into batches of BATCH_SIZE
    # ------------------------------------------------------------------
    healthy_paths   = collect_paths(args.healthy_dir)
    pneumonia_paths = collect_paths(args.pneumonia_dir)

    healthy_batches   = list(chunked(healthy_paths,   BATCH_SIZE))
    pneumonia_batches = list(chunked(pneumonia_paths, BATCH_SIZE))

    n_batches = min(len(healthy_batches), len(pneumonia_batches))

    skipped_h = len(healthy_paths)   - len(healthy_batches)   * BATCH_SIZE
    skipped_p = len(pneumonia_paths) - len(pneumonia_batches) * BATCH_SIZE

    print(f"\nHealthy images   : {len(healthy_paths)}  → {len(healthy_batches)} complete batches"
          + (f"  ({skipped_h} skipped)" if skipped_h else ""))
    print(f"Pneumonia images : {len(pneumonia_paths)}  → {len(pneumonia_batches)} complete batches"
          + (f"  ({skipped_p} skipped)" if skipped_p else ""))
    print(f"Output files     : {n_batches}  (limited by the smaller class)\n")

    # ------------------------------------------------------------------
    # Transform (same as training eval — no augmentation)
    # ------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx in range(n_batches):
        h_paths = healthy_batches[batch_idx]
        p_paths = pneumonia_batches[batch_idx]

        healthy_imgs   = load_batch(h_paths, transform, device)
        pneumonia_imgs = load_batch(p_paths, transform, device)

        with torch.no_grad():
            gen_pneumonia = G_H2P(healthy_imgs)
            gen_healthy   = G_P2H(pneumonia_imgs)

        save_path = out_dir / f"change_heatmap_{batch_idx + 1:03d}.png"

        render_batch(
            healthy_imgs.cpu(),
            pneumonia_imgs.cpu(),
            gen_pneumonia.cpu(),
            gen_healthy.cpu(),
            args,
            save_path,
            batch_idx,
            n_batches,
        )

        print(f"  [{batch_idx + 1:>{len(str(n_batches))}}/{n_batches}]  {save_path.name}"
              f"  (H: {h_paths[0].name}, {h_paths[1].name}"
              f"  |  P: {p_paths[0].name}, {p_paths[1].name})")

    print(f"\nDone. {n_batches} files saved to {out_dir}/")


if __name__ == "__main__":
    main()