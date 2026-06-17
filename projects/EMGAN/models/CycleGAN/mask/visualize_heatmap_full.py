"""
Heatmap visualization of mask-guided CycleGAN — full-image difference.

Uses the same mask-guided model (2-channel input) and keeps the hard-mask
post-processing (pixels outside the lung are identical to the original).
The heatmap is computed over the FULL image: inside the lung it shows the
actual pixel-level changes; outside it is 0 because the images are identical.

Each PNG shows two rows:
    Row 0 — Healthy  → Pneumonia: [Input | Generated | Change Heatmap (full) | Lung Mask]
    Row 1 — Pneumonia → Healthy:  [Input | Generated | Change Heatmap (full) | Lung Mask]

Usage (from projects/ecgpcx-ray/):
    python models/CycleGAN/mask/visualize_heatmap_full.py \\
        --checkpoint models/CycleGAN/mask/checkpoints/epoch_199.pt \\
        --healthy_dir          data/processed/test/healthy \\
        --pneumonia_dir        data/processed/test/pneumonia \\
        --healthy_masks_dir    data/processed/test/healthy_masks \\
        --pneumonia_masks_dir  data/processed/test/pneumonia_masks \\
        --healthy_images   00020293_001.png 00022733_000.png \\
        --pneumonia_images 00022877_014.png 00022201_000.png \\
        --device cuda
"""

import argparse
import sys
from pathlib import Path

import matplotlib.cm as mplcm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision.transforms.functional import InterpolationMode

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils.cyclegan import CycleGAN

EXTS       = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
BATCH_SIZE = 2

INPUT_COLOR = "#2c7bb6"
GEN_COLOR   = "#d7191c"
HEAT_COLOR  = "#4dac26"
MASK_COLOR  = "#9c27b0"

COL_TITLES = ["Input (Real)", "Generated", "Change Heatmap (full image)"]
COL_COLORS = [INPUT_COLOR, GEN_COLOR, HEAT_COLOR]
ROW_LABELS = [
    ("Healthy\n→ Pneumonia", INPUT_COLOR),
    ("Pneumonia\n→ Healthy",  GEN_COLOR),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_paths(directory):
    paths = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")
    return paths


def chunked(lst, size):
    for i in range(0, len(lst) - size + 1, size):
        yield lst[i: i + size]


def load_batch(img_paths, masks_dir, image_size, device):
    masks_dir = Path(masks_dir)
    imgs, masks = [], []
    for p in img_paths:
        img  = TF.resize(Image.open(p).convert("L"), (image_size, image_size))
        mask = TF.resize(
            Image.open(masks_dir / p.name).convert("L"),
            (image_size, image_size),
            interpolation=InterpolationMode.NEAREST,
        )
        img_t  = TF.normalize(TF.to_tensor(img), [0.5], [0.5])
        mask_t = (TF.to_tensor(mask) > 0.5).float()
        imgs.append(img_t)
        masks.append(mask_t)
    return torch.stack(imgs).to(device), torch.stack(masks).to(device)


def to_gray(tensor):
    return np.clip(tensor.squeeze(0).numpy() * 0.5 + 0.5, 0, 1)


def make_heatmap_overlay_full(real, generated, cmap_name, alpha, blur_sigma):
    """Compute |real - generated| over the FULL image (no mask restriction)."""
    diff = np.abs(real - generated)          # full image, zeros outside lung
    if blur_sigma > 0:
        diff = gaussian_filter(diff, sigma=blur_sigma)
    diff_norm   = diff / (diff.max() + 1e-8)
    heatmap_rgb = mplcm.get_cmap(cmap_name)(diff_norm)[..., :3]
    gray_rgb    = np.stack([real, real, real], axis=-1)
    return np.clip((1.0 - alpha) * gray_rgb + alpha * heatmap_rgb, 0, 1)


def style_axis(ax, spine_color, lw=2.5):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
        spine.set_linewidth(lw)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def render_batch(h_imgs, p_imgs, gen_P, gen_H, h_masks, p_masks,
                 args, save_path, batch_idx, n_batches, n=BATCH_SIZE):
    row_data = [
        (h_imgs, gen_P, h_masks),
        (p_imgs, gen_H, p_masks),
    ]

    n_img_cols = 3 * n
    fig = plt.figure(figsize=(3.0 * n_img_cols + 0.6, 3.0 * 2 + 1.4))
    gs  = GridSpec(
        nrows=2, ncols=n_img_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_img_cols + [0.05],
        wspace=0.05, hspace=0.18,
        left=0.10, right=0.96, top=0.87, bottom=0.04,
    )
    axes    = np.array([
        [fig.add_subplot(gs[r, c]) for c in range(n_img_cols)]
        for r in range(2)
    ])
    cbar_ax = fig.add_subplot(gs[:, -1])

    for i in range(n):
        for k, (title, color) in enumerate(zip(COL_TITLES, COL_COLORS)):
            axes[0, 3 * i + k].set_title(
                title, fontsize=8, fontweight="bold", color=color, pad=5
            )

    for r, (real_batch, gen_batch, _) in enumerate(row_data):
        for i in range(n):
            real_np = to_gray(real_batch[i].cpu())
            gen_np  = to_gray(gen_batch[i].cpu())

            heat_np = make_heatmap_overlay_full(
                real_np, gen_np, args.colormap, args.overlay_alpha, args.blur_sigma
            )

            axes[r, 3 * i    ].imshow(real_np, cmap="gray", vmin=0, vmax=1)
            axes[r, 3 * i + 1].imshow(gen_np,  cmap="gray", vmin=0, vmax=1)
            axes[r, 3 * i + 2].imshow(heat_np)

            style_axis(axes[r, 3 * i    ], INPUT_COLOR)
            style_axis(axes[r, 3 * i + 1], GEN_COLOR)
            style_axis(axes[r, 3 * i + 2], HEAT_COLOR)

        axes[r, 0].set_ylabel(
            ROW_LABELS[r][0], fontsize=10, fontweight="bold",
            color=ROW_LABELS[r][1], labelpad=10, rotation=90, va="center",
        )

    sm = plt.cm.ScalarMappable(
        cmap=mplcm.get_cmap(args.colormap),
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Normalised |Δ pixel|", fontsize=8, labelpad=6)
    cbar.ax.tick_params(labelsize=7)

    patches = [
        mpatches.Patch(color=INPUT_COLOR, label="Input (Real)"),
        mpatches.Patch(color=GEN_COLOR,   label="Generated (hard mask applied)"),
        mpatches.Patch(color=HEAT_COLOR,  label=f"Full-image heatmap — 0 outside lung (α={args.overlay_alpha})"),
    ]
    fig.legend(handles=patches, loc="upper right", fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.95, 0.97))

    fig.suptitle(
        f"CycleGAN (Mask-Guided) — Full-image Change Heatmap  [{batch_idx + 1}/{n_batches}]",
        fontsize=12, fontweight="bold", y=0.95,
    )

    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Full-image heatmap for mask-guided CycleGAN (hard mask kept)."
    )
    p.add_argument("--checkpoint",          required=True)
    p.add_argument("--healthy_dir",         required=True)
    p.add_argument("--pneumonia_dir",       required=True)
    p.add_argument("--healthy_masks_dir",   required=True)
    p.add_argument("--pneumonia_masks_dir", required=True)
    p.add_argument("--image_size",    type=int,   default=128)
    p.add_argument("--colormap",      default="hot")
    p.add_argument("--overlay_alpha", type=float, default=0.55)
    p.add_argument("--blur_sigma",    type=float, default=2.0)
    p.add_argument("--output_dir",    default="models/CycleGAN/mask/outputs/img_generation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--healthy_images",   nargs="+", default=None,
                   help="Specific healthy filenames. If omitted, sweeps the full directory.")
    p.add_argument("--pneumonia_images", nargs="+", default=None,
                   help="Specific pneumonia filenames. Must match length of --healthy_images.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = CycleGAN(image_channels=1, use_mask=True)
    model.G_H2P.load_state_dict(ckpt["G_H2P"])
    model.G_P2H.load_state_dict(ckpt["G_P2H"])
    G_H2P = model.G_H2P.to(device).eval()
    G_P2H = model.G_P2H.to(device).eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')} checkpoint.")

    healthy_dir   = Path(args.healthy_dir)
    pneumonia_dir = Path(args.pneumonia_dir)
    out_dir       = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.healthy_images is not None or args.pneumonia_images is not None:
        if args.healthy_images is None or args.pneumonia_images is None:
            raise ValueError("--healthy_images and --pneumonia_images must both be provided.")
        if len(args.healthy_images) != len(args.pneumonia_images):
            raise ValueError(
                f"--healthy_images ({len(args.healthy_images)}) and "
                f"--pneumonia_images ({len(args.pneumonia_images)}) must have the same length."
            )
        healthy_paths   = [healthy_dir   / name for name in args.healthy_images]
        pneumonia_paths = [pneumonia_dir / name for name in args.pneumonia_images]
        for p in healthy_paths + pneumonia_paths:
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")

        h_imgs, h_masks = load_batch(healthy_paths,   args.healthy_masks_dir,   args.image_size, device)
        p_imgs, p_masks = load_batch(pneumonia_paths, args.pneumonia_masks_dir, args.image_size, device)

        with torch.no_grad():
            gen_P = G_H2P(torch.cat([h_imgs, h_masks], dim=1))
            gen_P = gen_P * h_masks + h_imgs * (1 - h_masks)   # hard mask kept
            gen_H = G_P2H(torch.cat([p_imgs, p_masks], dim=1))
            gen_H = gen_H * p_masks + p_imgs * (1 - p_masks)   # hard mask kept

        save_path = out_dir / "change_heatmap_full_selected.png"
        render_batch(h_imgs, p_imgs, gen_P, gen_H, h_masks, p_masks,
                     args, save_path, batch_idx=0, n_batches=1, n=len(healthy_paths))
        print(f"\nSaved: {save_path}")

    else:
        healthy_paths   = collect_paths(args.healthy_dir)
        pneumonia_paths = collect_paths(args.pneumonia_dir)

        healthy_batches   = list(chunked(healthy_paths,   BATCH_SIZE))
        pneumonia_batches = list(chunked(pneumonia_paths, BATCH_SIZE))
        n_batches = min(len(healthy_batches), len(pneumonia_batches))

        print(f"\nHealthy   : {len(healthy_paths)} → {len(healthy_batches)} batches")
        print(f"Pneumonia : {len(pneumonia_paths)} → {len(pneumonia_batches)} batches")
        print(f"Output    : {n_batches} files\n")

        for batch_idx in range(n_batches):
            h_imgs, h_masks = load_batch(healthy_batches[batch_idx],   args.healthy_masks_dir,   args.image_size, device)
            p_imgs, p_masks = load_batch(pneumonia_batches[batch_idx], args.pneumonia_masks_dir, args.image_size, device)

            with torch.no_grad():
                gen_P = G_H2P(torch.cat([h_imgs, h_masks], dim=1))
                gen_P = gen_P * h_masks + h_imgs * (1 - h_masks)
                gen_H = G_P2H(torch.cat([p_imgs, p_masks], dim=1))
                gen_H = gen_H * p_masks + p_imgs * (1 - p_masks)

            save_path = out_dir / f"change_heatmap_full_{batch_idx + 1:03d}.png"
            render_batch(h_imgs, p_imgs, gen_P, gen_H, h_masks, p_masks,
                         args, save_path, batch_idx, n_batches)

            h_names = [p.name for p in healthy_batches[batch_idx]]
            p_names = [p.name for p in pneumonia_batches[batch_idx]]
            print(f"  [{batch_idx + 1}/{n_batches}]  {save_path.name}"
                  f"  (H: {', '.join(h_names)}  |  P: {', '.join(p_names)})")

        print(f"\nDone. {n_batches} files saved to {out_dir}/")


if __name__ == "__main__":
    main()
