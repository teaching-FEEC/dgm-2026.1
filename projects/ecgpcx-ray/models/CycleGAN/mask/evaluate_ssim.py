"""
SSIM evaluation script for mask-guided CycleGAN counterfactual generation.

Computes Structural Similarity Index (SSIM) between original and generated images:
  - Original healthy  (test) vs. Generated pneumonia (H→P)
  - Original pneumonia (test) vs. Generated healthy   (P→H)

Images are selected from a split CSV and matched by filename to the generated dirs.
  - H→P: rows with Label == 0  (No Finding / healthy)
  - P→H: rows with Finding Labels == 'Pneumonia' exactly (pure pneumonia)

Usage:
    python mask/evaluate_ssim.py \\
        --split_csv          ../../data/test_split.csv \\
        --healthy_dir        ../../data/processed/test/healthy \\
        --pneumonia_dir      ../../data/processed/test/pneumonia \\
        --generated_h2p_dir  ../../data/generated/healthy_to_pneumonia \\
        --generated_p2h_dir  ../../data/generated/pneumonia_to_healthy \\
        [--image_size 128] [--device cuda] [--output mask/outputs/ssim_mask.json]
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from tqdm import tqdm

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split_filenames(split_csv: Path) -> tuple[set[str], set[str]]:
    """Return (healthy_filenames, pneumonia_filenames) from the split CSV.

    Healthy  : Label == 0
    Pneumonia: Finding Labels == 'Pneumonia' (pure, no co-findings)
    """
    df = pd.read_csv(split_csv)
    healthy   = set(df.loc[df["Label"] == 0, "Image Index"].tolist())
    pneumonia = set(df.loc[df["Finding Labels"] == "Pneumonia", "Image Index"].tolist())
    return healthy, pneumonia


def list_images(folder: Path) -> dict[str, Path]:
    """Return {filename: path} for all images in folder."""
    return {
        p.name: p
        for p in sorted(folder.iterdir())
        if p.suffix.lower() in EXTS
    }


def load_grayscale_tensor(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((image_size, image_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)


def compute_ssim_pairs(
    original_dir: Path,
    generated_dir: Path,
    allowed_filenames: set[str],
    image_size: int,
    device: torch.device,
    direction: str,
) -> list[dict]:
    originals  = list_images(original_dir)
    generateds = list_images(generated_dir)

    candidates = set(originals) & set(generateds) & allowed_filenames
    shared = sorted(candidates)
    if not shared:
        raise FileNotFoundError(
            f"No matching filenames after applying split filter for {direction}.\n"
            f"  original_dir : {original_dir}\n"
            f"  generated_dir: {generated_dir}"
        )

    print(f"  {direction}: {len(shared)} pairs "
          f"(split: {len(allowed_filenames)}, originals: {len(originals)}, "
          f"generated: {len(generateds)}).")

    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    rows = []
    for name in tqdm(shared, desc=f"SSIM {direction}", leave=False):
        orig = load_grayscale_tensor(originals[name],  image_size).to(device)
        gen  = load_grayscale_tensor(generateds[name], image_size).to(device)
        score = float(ssim_fn(orig, gen).item())
        rows.append({"filename": name, "direction": direction, "ssim": score})

    return rows


def summarize(rows: list[dict]) -> dict:
    scores = np.array([r["ssim"] for r in rows], dtype=np.float64)
    return {
        "n": len(scores),
        "mean": float(scores.mean()),
        "std":  float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "min":  float(scores.min()),
        "max":  float(scores.max()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SSIM evaluation for mask-guided CycleGAN."
    )
    p.add_argument("--split_csv",         required=True,
                   help="Path to test_split.csv used to select images.")
    p.add_argument("--healthy_dir",       required=True,
                   help="Directory with original test healthy images.")
    p.add_argument("--pneumonia_dir",     required=True,
                   help="Directory with original test pneumonia images.")
    p.add_argument("--generated_h2p_dir", required=True,
                   help="Directory with generated H→P images (same filenames as healthy_dir).")
    p.add_argument("--generated_p2h_dir", required=True,
                   help="Directory with generated P→H images (same filenames as pneumonia_dir).")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None,
                   help="JSON output path (default: mask/outputs/ssim_mask.json).")
    p.add_argument("--output_csv", default=None,
                   help="CSV output path for per-image scores (default: mask/outputs/ssim_mask.csv).")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}\n")

    print(f"Loading split: {args.split_csv}")
    healthy_names, pneumonia_names = load_split_filenames(Path(args.split_csv))
    print(f"  Healthy (Label==0):              {len(healthy_names)} images")
    print(f"  Pneumonia (Finding=='Pneumonia'): {len(pneumonia_names)} images\n")

    rows_h2p = compute_ssim_pairs(
        Path(args.healthy_dir),
        Path(args.generated_h2p_dir),
        healthy_names,
        args.image_size, device,
        direction="H→P",
    )
    rows_p2h = compute_ssim_pairs(
        Path(args.pneumonia_dir),
        Path(args.generated_p2h_dir),
        pneumonia_names,
        args.image_size, device,
        direction="P→H",
    )
    all_rows = rows_h2p + rows_p2h

    stats_h2p = summarize(rows_h2p)
    stats_p2h = summarize(rows_p2h)
    stats_all = summarize(all_rows)

    print("\n" + "=" * 55)
    print("SSIM Results (mask-guided CycleGAN)")
    print("=" * 55)
    print(f"  H→P  — mean: {stats_h2p['mean']:.4f}  std: {stats_h2p['std']:.4f}"
          f"  min: {stats_h2p['min']:.4f}  max: {stats_h2p['max']:.4f}  n: {stats_h2p['n']}")
    print(f"  P→H  — mean: {stats_p2h['mean']:.4f}  std: {stats_p2h['std']:.4f}"
          f"  min: {stats_p2h['min']:.4f}  max: {stats_p2h['max']:.4f}  n: {stats_p2h['n']}")
    print(f"  All  — mean: {stats_all['mean']:.4f}  std: {stats_all['std']:.4f}"
          f"  min: {stats_all['min']:.4f}  max: {stats_all['max']:.4f}  n: {stats_all['n']}")
    print("=" * 55)

    # JSON output
    output_json = Path(args.output) if args.output else Path("mask/outputs/ssim_mask.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "H2P": stats_h2p,
        "P2H": stats_p2h,
        "combined": stats_all,
        "split_csv":          str(args.split_csv),
        "healthy_dir":        str(args.healthy_dir),
        "pneumonia_dir":      str(args.pneumonia_dir),
        "generated_h2p_dir":  str(args.generated_h2p_dir),
        "generated_p2h_dir":  str(args.generated_p2h_dir),
        "image_size": args.image_size,
        "use_mask": True,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved to: {output_json}")

    # CSV output
    output_csv = Path(args.output_csv) if args.output_csv else Path("mask/outputs/ssim_mask.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "direction", "ssim"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"CSV  saved to: {output_csv}")


if __name__ == "__main__":
    main()
