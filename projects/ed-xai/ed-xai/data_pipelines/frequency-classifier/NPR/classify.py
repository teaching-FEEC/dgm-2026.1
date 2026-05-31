import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from npr_model import resnet50

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_EVERY = 500


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def npr_to_fakeclue(score):
    """Map NPR sigmoid score to FakeClue label.

    NPR: high score = fake (class 1)
    FakeClue: 0 = fake, 1 = real
    """
    if score > 0.5:
        return 0
    return 1


# ---------------------------------------------------------------------------
# Checkpoint / resume support
# ---------------------------------------------------------------------------

def _load_checkpoint_results(checkpoint_path, total):
    if not checkpoint_path.exists():
        return [None] * total, 0, 0
    with open(checkpoint_path) as f:
        data = json.load(f)
    results = data["results"]
    if len(results) < total:
        results.extend([None] * (total - len(results)))
    elif len(results) > total:
        results = results[:total]
    start_idx = data.get("next_idx", 0)
    failed = data.get("failed", 0)
    return results, start_idx, failed


def _save_checkpoint(checkpoint_path, results, next_idx, failed):
    with open(checkpoint_path, "w") as f:
        json.dump({"results": results, "next_idx": next_idx, "failed": failed},
                  f, indent=2)


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def get_test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def classify_split(model, split, data_dir, output_dir, limit, device,
                   batch_size):
    json_path = data_dir / "data_json" / f"{split}.json"
    split_dir = data_dir / split

    with open(json_path) as f:
        entries = json.load(f)

    if limit is not None:
        entries = entries[:limit]

    checkpoint_path = output_dir / f".{split}_npr_checkpoint.json"
    results, start_idx, failed = _load_checkpoint_results(
        checkpoint_path, len(entries))

    if start_idx > 0:
        logger.info("Resuming from image %d/%d (%d already processed)",
                    start_idx, len(entries), start_idx)

    transform = get_test_transform()

    for batch_start in tqdm(range(start_idx, len(entries), batch_size),
                            desc=f"{split} (npr)",
                            initial=start_idx // batch_size,
                            total=(len(entries) + batch_size - 1) // batch_size):
        batch_end = min(batch_start + batch_size, len(entries))
        batch_entries = entries[batch_start:batch_end]

        tensors = []
        valid_indices = []
        for i, entry in enumerate(batch_entries):
            idx = batch_start + i
            img_path = split_dir / entry["image"]
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img)
                img.close()
                tensors.append(tensor)
                valid_indices.append(idx)
            except Exception as e:
                logger.warning("Failed to load %s: %s", entry["image"], e)
                results[idx] = {
                    "image": entry["image"],
                    "ground_truth": entry["label"],
                    "predicted_label": -1,
                    "confidence": None,
                    "model": "npr",
                    "category": entry.get("cate", "unknown"),
                    "error": str(e),
                }
                failed += 1

        if tensors:
            batch_tensor = torch.stack(tensors).to(device)
            with torch.no_grad():
                outputs = model(batch_tensor)
                scores = torch.sigmoid(outputs).flatten().cpu().tolist()
            del batch_tensor, outputs

            for j, idx in enumerate(valid_indices):
                entry = entries[idx]
                score = scores[j]
                pred = npr_to_fakeclue(score)
                confidence = score if score > 0.5 else 1.0 - score
                results[idx] = {
                    "image": entry["image"],
                    "ground_truth": entry["label"],
                    "predicted_label": pred,
                    "confidence": round(confidence, 6),
                    "model": "npr",
                    "category": entry.get("cate", "unknown"),
                    "error": None,
                }

        if batch_end % CHECKPOINT_EVERY < batch_size:
            _save_checkpoint(checkpoint_path, results, batch_end, failed)

    out_path = output_dir / f"{split}_frequency_npr.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print_summary(results, split, failed)
    logger.info("Results written to %s", out_path)
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results, split, failed):
    valid = [r for r in results if r is not None and r["error"] is None]
    total = len(results)
    correct = sum(1 for r in valid
                  if r["ground_truth"] == r["predicted_label"])
    accuracy = correct / len(valid) if valid else 0.0

    tp = sum(1 for r in valid
             if r["ground_truth"] == 0 and r["predicted_label"] == 0)
    tn = sum(1 for r in valid
             if r["ground_truth"] == 1 and r["predicted_label"] == 1)
    fp = sum(1 for r in valid
             if r["ground_truth"] == 1 and r["predicted_label"] == 0)
    fn = sum(1 for r in valid
             if r["ground_truth"] == 0 and r["predicted_label"] == 1)

    print(f"\n{'='*60}")
    print(f"Split: {split} | Model: npr")
    print(f"{'='*60}")
    print(f"Total: {total} | Processed: {len(valid)} | Failed: {failed}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(valid)})")
    print(f"\nConfusion matrix (FakeClue convention: 0=fake, 1=real):")
    print(f"  TP (fake->fake):  {tp}")
    print(f"  TN (real->real):  {tn}")
    print(f"  FP (real->fake):  {fp}")
    print(f"  FN (fake->real):  {fn}")

    categories = Counter()
    cat_correct = Counter()
    for r in valid:
        cat = r.get("category", "unknown")
        categories[cat] += 1
        if r["ground_truth"] == r["predicted_label"]:
            cat_correct[cat] += 1

    if categories:
        print(f"\nPer-category accuracy:")
        for cat in sorted(categories):
            acc = cat_correct[cat] / categories[cat]
            print(f"  {cat}: {acc:.4f} ({cat_correct[cat]}/{categories[cat]})")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir / ".." / ".." / ".." / ".."

    parser = argparse.ArgumentParser(
        description="Classify FakeClue images using NPR model.")
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to NPR .pth checkpoint file.")
    parser.add_argument(
        "--split", default="test", choices=("train", "test", "both"),
        help="FakeClue split to process (default: test).")
    parser.add_argument(
        "--data-dir", type=Path,
        default=project_root / "data" / "external" / "FakeClue",
        help="FakeClue dataset root directory.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: DATA_DIR/data_json/NPR).")
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference (default: 32).")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only this many images (for debugging).")
    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.output_dir is None:
        args.output_dir = args.data_dir / "data_json" / "NPR"
    else:
        args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Building NPR model...")
    model = resnet50(num_classes=1)

    logger.info("Loading checkpoint: %s on %s", args.checkpoint, device)
    state_dict = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=False)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v
                      for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        classify_split(model, split, args.data_dir, args.output_dir,
                       args.limit, device, args.batch_size)


if __name__ == "__main__":
    main()
