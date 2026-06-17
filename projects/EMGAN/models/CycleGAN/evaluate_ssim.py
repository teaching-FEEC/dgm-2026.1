"""
SSIM evaluation script for CycleGAN (no mask) counterfactual generation.

Generates translations on-the-fly from a checkpoint and computes SSIM between
each original test image and its generated counterfactual:
  - Original healthy  (test) vs. Generated pneumonia (H→P)
  - Original pneumonia (test) vs. Generated healthy   (P→H)

Uses the reduced (balanced) test set from data/processed/test/.

Usage:
    python evaluate_ssim.py \\
        --checkpoint    checkpoints/epoch_199.pt \\
        --healthy_dir   ../../data/processed/test/healthy \\
        --pneumonia_dir ../../data/processed/test/pneumonia \\
        [--batch_size 16] [--image_size 128] [--device cuda]
        [--output outputs/ssim.json]
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_utils.cyclegan import CycleGAN

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self, img_dir: Path, image_size: int):
        self.paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXTS)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        self.names = [p.name for p in self.paths]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("L"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_ssim_direction(
    img_dir: Path,
    generator: torch.nn.Module,
    image_size: int,
    batch_size: int,
    device: torch.device,
    direction: str,
) -> list[dict]:
    ds     = ImageFolderDataset(img_dir, image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  {direction}: {len(ds)} images.")

    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    rows    = []
    img_idx = 0
    for batch in tqdm(loader, desc=f"SSIM {direction}", leave=False):
        batch = batch.to(device)
        with torch.no_grad():
            generated = generator(batch)

        for orig_t, gen_t in zip(batch, generated):
            score = float(ssim_fn(orig_t.unsqueeze(0), gen_t.unsqueeze(0)).item())
            rows.append({
                "filename":  ds.names[img_idx],
                "direction": direction,
                "ssim":      score,
            })
            img_idx += 1

    return rows


def summarize(rows: list[dict]) -> dict:
    scores = np.array([r["ssim"] for r in rows], dtype=np.float64)
    return {
        "n":    len(scores),
        "mean": float(scores.mean()),
        "std":  float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "min":  float(scores.min()),
        "max":  float(scores.max()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SSIM evaluation for CycleGAN (no mask).")
    p.add_argument("--checkpoint",    required=True,
                   help="Path to .pt checkpoint (e.g. checkpoints/epoch_199.pt).")
    p.add_argument("--healthy_dir",   required=True,
                   help="Directory with test healthy X-ray images.")
    p.add_argument("--pneumonia_dir", required=True,
                   help="Directory with test pneumonia X-ray images.")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--image_size",  type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output",     default=None,
                   help="JSON output path (default: outputs/ssim.json).")
    p.add_argument("--output_csv", default=None,
                   help="CSV output path (default: outputs/ssim.csv).")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}\n")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = CycleGAN(image_channels=1)
    model.G_H2P.load_state_dict(ckpt["G_H2P"])
    model.G_P2H.load_state_dict(ckpt["G_P2H"])
    G_H2P = model.G_H2P.to(device).eval()
    G_P2H = model.G_P2H.to(device).eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')} checkpoint.\n")

    rows_h2p = compute_ssim_direction(
        Path(args.healthy_dir),   G_H2P,
        args.image_size, args.batch_size, device, "H→P",
    )
    rows_p2h = compute_ssim_direction(
        Path(args.pneumonia_dir), G_P2H,
        args.image_size, args.batch_size, device, "P→H",
    )
    all_rows  = rows_h2p + rows_p2h
    stats_h2p = summarize(rows_h2p)
    stats_p2h = summarize(rows_p2h)
    stats_all = summarize(all_rows)

    print("\n" + "=" * 55)
    print("SSIM Results (CycleGAN — no mask)")
    print("=" * 55)
    print(f"  H→P  — mean: {stats_h2p['mean']:.4f}  std: {stats_h2p['std']:.4f}"
          f"  min: {stats_h2p['min']:.4f}  max: {stats_h2p['max']:.4f}  n: {stats_h2p['n']}")
    print(f"  P→H  — mean: {stats_p2h['mean']:.4f}  std: {stats_p2h['std']:.4f}"
          f"  min: {stats_p2h['min']:.4f}  max: {stats_p2h['max']:.4f}  n: {stats_p2h['n']}")
    print(f"  All  — mean: {stats_all['mean']:.4f}  std: {stats_all['std']:.4f}"
          f"  min: {stats_all['min']:.4f}  max: {stats_all['max']:.4f}  n: {stats_all['n']}")
    print("=" * 55)

    # JSON
    output_json = Path(args.output) if args.output else Path("outputs/ssim.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({
            "H2P": stats_h2p,
            "P2H": stats_p2h,
            "combined": stats_all,
            "checkpoint":    str(args.checkpoint),
            "healthy_dir":   str(args.healthy_dir),
            "pneumonia_dir": str(args.pneumonia_dir),
            "image_size":    args.image_size,
            "use_mask":      False,
            "timestamp":     datetime.datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)
    print(f"\nJSON saved to: {output_json}")

    # CSV
    output_csv = Path(args.output_csv) if args.output_csv else Path("outputs/ssim.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "direction", "ssim"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"CSV  saved to: {output_csv}")


if __name__ == "__main__":
    main()
