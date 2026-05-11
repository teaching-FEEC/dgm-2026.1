import argparse
import json
import logging
import pathlib
from collections import Counter
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from scipy import interpolate
from tqdm import tqdm

from spai_model import build_mf_vit

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FEATURE_EXTRACTION_BATCH = 400
CHECKPOINT_EVERY = 500


# ---------------------------------------------------------------------------
# Weight loading (from spai/utils.py)
# ---------------------------------------------------------------------------

def remap_pretrained_keys_vit(model, checkpoint_model):
    if (getattr(model, 'use_rel_pos_bias', False)
            and "rel_pos_bias.relative_position_bias_table"
            in checkpoint_model):
        num_layers = model.get_num_layers()
        rel_pos_bias = checkpoint_model[
            "rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_model[
                f"blocks.{i}.attn.relative_position_bias_table"
            ] = rel_pos_bias.clone()
        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = (dst_num_pos
                                - (dst_patch_shape[0] * 2 - 1)
                                * (dst_patch_shape[1] * 2 - 1))
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                logger.info("Position interpolate for %s from %dx%d to %dx%d",
                            key, src_size, src_size, dst_size, dst_size)
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]
                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                all_rel_pos_bias = []
                for i in range(num_attn_heads):
                    z = (rel_pos_bias[:, i]
                         .view(src_size, src_size).float().numpy())
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1)
                        .to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                new_rel_pos_bias = torch.cat(
                    (rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias

    return checkpoint_model


def load_pretrained(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=False)
    checkpoint_model = checkpoint['model']

    if any('encoder.' in k for k in checkpoint_model.keys()):
        checkpoint_model = {
            k.replace('encoder.', ''): v
            for k, v in checkpoint_model.items() if k.startswith('encoder.')
        }

    remap_pretrained_keys_vit(
        model.get_vision_transformer(), checkpoint_model)

    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info("Checkpoint loaded: %s", msg)

    del checkpoint
    torch.cuda.empty_cache()

    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image preprocessing (replicates SPAI test transform)
# ---------------------------------------------------------------------------

def get_test_transform():
    return A.Compose([
        A.PadIfNeeded(min_height=224, min_width=224),
        A.Normalize(mean=0., std=1.),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def spai_to_fakeclue(score):
    """Map SPAI sigmoid score to FakeClue label.

    SPAI: high score = class 1 (fake)
    FakeClue: 0 = fake, 1 = real
    """
    if score > 0.5:
        return 0
    return 1


# ---------------------------------------------------------------------------
# Core classification
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


def classify_split(model, split, data_dir, output_dir, limit, device):
    json_path = data_dir / "data_json" / f"{split}.json"
    split_dir = data_dir / split

    with open(json_path) as f:
        entries = json.load(f)

    if limit is not None:
        entries = entries[:limit]

    checkpoint_path = output_dir / f".{split}_spai_checkpoint.json"
    results, start_idx, failed = _load_checkpoint_results(
        checkpoint_path, len(entries))

    if start_idx > 0:
        logger.info("Resuming from image %d/%d (%d already processed)",
                     start_idx, len(entries), start_idx)

    transform = get_test_transform()

    for idx in tqdm(range(start_idx, len(entries)), desc=f"{split} (spai)",
                    initial=start_idx, total=len(entries)):
        entry = entries[idx]
        img_path = split_dir / entry["image"]
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            transformed = transform(image=img_np)["image"]
            img.close()
            del img, img_np

            images = [transformed.unsqueeze(0).to(device)]
            del transformed

            with torch.no_grad():
                output = model(images, FEATURE_EXTRACTION_BATCH)
                score = torch.sigmoid(output).item()
            del images, output

            pred = spai_to_fakeclue(score)
            confidence = score if score > 0.5 else 1.0 - score

            results[idx] = {
                "image": entry["image"],
                "ground_truth": entry["label"],
                "predicted_label": pred,
                "confidence": round(confidence, 6),
                "model": "spai",
                "category": entry.get("cate", "unknown"),
                "error": None,
            }
        except Exception as e:
            logger.warning("Failed to process %s: %s", entry["image"], e)
            results[idx] = {
                "image": entry["image"],
                "ground_truth": entry["label"],
                "predicted_label": -1,
                "confidence": None,
                "model": "spai",
                "category": entry.get("cate", "unknown"),
                "error": str(e),
            }
            failed += 1
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(checkpoint_path, results, idx + 1, failed)

    out_path = output_dir / f"{split}_frequency_spai.json"
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
    valid = [r for r in results if r["error"] is None]
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
    print(f"Split: {split} | Model: spai")
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
        description="Classify FakeClue images using SPAI model.")
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to spai.pth checkpoint file.")
    parser.add_argument(
        "--split", default="test", choices=("train", "test", "both"),
        help="FakeClue split to process (default: test).")
    parser.add_argument(
        "--data-dir", type=Path,
        default=project_root / "data" / "external" / "FakeClue",
        help="FakeClue dataset root directory.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: DATA_DIR/data_json/SPAI).")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only this many images (for debugging).")
    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.output_dir is None:
        args.output_dir = args.data_dir / "data_json" / "SPAI"
    else:
        args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Building SPAI model...")
    model = build_mf_vit()

    logger.info("Loading checkpoint: %s on %s", args.checkpoint, device)
    model = load_pretrained(model, args.checkpoint, device)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        classify_split(model, split, args.data_dir, args.output_dir,
                       args.limit, device)


if __name__ == "__main__":
    main()
