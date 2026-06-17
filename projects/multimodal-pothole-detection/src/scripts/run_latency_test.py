"""Latency benchmark for Point-E inference.

Iterates over images in a directory, runs inference for each one, records the
wall-clock time per sample, and writes summary statistics to a JSON file.
Predictions are discarded immediately — nothing is saved to disk.

Typical usage
-------------
python -m src.scripts.run_latency_test \\
    --checkpoint  artifacts/run_001/checkpoints/checkpoint_best.pt \\
    --images_dir  data/processed/point_e_ready/images \\

    --output      reports/latency/latency_results.json \\
    --limit       20
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from src.models.point_e_model import PotholePointE

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Latency benchmark for Point-E inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the latency results JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all images in the directory)",
    )
    parser.add_argument("--base_model", type=str, default="base40M")
    parser.add_argument("--points_base", type=int, default=1024)
    parser.add_argument("--guidance_base", type=float, default=3.0)
    return parser


def _compute_stats(times: list[float]) -> dict:
    """Compute summary statistics from a list of per-sample inference times (seconds)."""
    import statistics

    sorted_times = sorted(times)
    n = len(sorted_times)
    p95_idx = max(0, int(0.95 * n) - 1)
    p99_idx = max(0, int(0.99 * n) - 1)

    return {
        "n_samples": n,
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "stdev_s": statistics.stdev(times) if n > 1 else 0.0,
        "min_s": sorted_times[0],
        "max_s": sorted_times[-1],
        "p95_s": sorted_times[p95_idx],
        "p99_s": sorted_times[p99_idx],
        "total_s": sum(times),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(
            f"Output file already exists at {output_path}. Remove it before rerunning."
        )

    # Collect image paths
    all_images = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    if not all_images:
        raise RuntimeError(f"No valid images found in {images_dir}")

    if args.limit is not None:
        all_images = all_images[: args.limit]

    print(f"Images to benchmark : {len(all_images)}")
    print(f"Checkpoint          : {checkpoint_path}")

    # Load model (upsampler off — same config used for final eval)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device              : {device}")

    model = PotholePointE(
        device=device,
        base_model_name=args.base_model,
        custom_base_weights_path=str(checkpoint_path),
        num_points_base=args.points_base,
        guidance_scale_base=args.guidance_base,
        use_upsampler=False,
    )

    per_sample: list[dict] = []
    skipped = 0

    print("\nRunning inference (predictions are discarded)...\n")
    for img_path in all_images:
        stem = img_path.stem

        img = Image.open(img_path).convert("RGB")

        t0 = time.perf_counter()
        _ = model.predict(img, 1.0)   # scale irrelevant — result discarded
        elapsed = time.perf_counter() - t0

        per_sample.append({"sample_id": stem, "inference_time_s": elapsed})
        print(f"  {stem}  {elapsed:.3f}s")

    if not per_sample:
        raise RuntimeError("No samples were benchmarked.")

    times = [s["inference_time_s"] for s in per_sample]
    stats = _compute_stats(times)

    print(f"\n{'='*50}")
    print(f"  Samples benchmarked : {stats['n_samples']}")
    print(f"  Mean latency        : {stats['mean_s']:.3f}s")
    print(f"  Median latency      : {stats['median_s']:.3f}s")
    print(f"  Std dev             : {stats['stdev_s']:.3f}s")
    print(f"  Min / Max           : {stats['min_s']:.3f}s / {stats['max_s']:.3f}s")
    print(f"  P95 / P99           : {stats['p95_s']:.3f}s / {stats['p99_s']:.3f}s")
    print(f"  Total time          : {stats['total_s']:.1f}s")
    if skipped:
        print(f"  Skipped             : {skipped}")
    print(f"{'='*50}\n")

    results = {
        "checkpoint": str(checkpoint_path),
        "images_dir": str(images_dir),
        "limit": args.limit,
        "device": str(device),
        "base_model": args.base_model,
        "skipped_no_scale": skipped,
        "stats": stats,
        "per_sample": per_sample,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"Results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
