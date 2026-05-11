import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_TYPES = ("ridge_pixel", "ridge_dct", "lasso_dct")
TARGET_SIZE = (1024, 1024)


# ---------------------------------------------------------------------------
# Preprocessing layers (replicated from GANDCTAnalysis)
# ---------------------------------------------------------------------------

def dct2(array):
    """2D DCT transform. Expects batched input (B, H, W, C)."""
    array = tf.cast(array, tf.float32)
    # BRCV -> BVCR
    array = tf.transpose(array, perm=[0, 3, 2, 1])
    array = tf.signal.dct(array, type=2, norm="ortho")
    # BVCR -> BVRC
    array = tf.transpose(array, perm=[0, 1, 3, 2])
    array = tf.signal.dct(array, type=2, norm="ortho")
    # BVRC -> BRCV
    array = tf.transpose(array, perm=[0, 2, 3, 1])
    return array


class DCTLayer(tf.keras.layers.Layer):
    def __init__(self, mean, var):
        super().__init__()
        self.mean = mean
        self.var = var
        self.std = np.sqrt(var)

    def build(self, input_shape):
        self.mean_w = self.add_weight(
            "mean", shape=input_shape[1:],
            initializer=tf.keras.initializers.Constant(self.mean),
            trainable=False,
        )
        self.std_w = self.add_weight(
            "std", shape=input_shape[1:],
            initializer=tf.keras.initializers.Constant(self.std),
            trainable=False,
        )

    def call(self, inputs):
        x = dct2(inputs)
        x = tf.abs(x)
        x += 1e-13
        x = tf.math.log(x)
        x = x - self.mean_w
        x = x / self.std_w
        return x


class PixelLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return (inputs / 127.5) - 1.0


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_and_resize(path, target_size=TARGET_SIZE):
    img = Image.open(path)
    img = img.convert("RGB")
    if img.size != target_size:
        img = img.resize(target_size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_type, models_dir):
    model_path = str(models_dir / "ffhq" / model_type)
    raw_model = tf.keras.models.load_model(model_path)

    if model_type == "ridge_pixel":
        return tf.keras.Sequential([PixelLayer(), raw_model])

    mean_var_dir = models_dir / "mean_var" / "ffhq_mean_var"
    mean = np.load(str(mean_var_dir / "mean.npy"))
    var = np.load(str(mean_var_dir / "var.npy"))
    return tf.keras.Sequential([DCTLayer(mean, var), raw_model])


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def gandct_to_fakeclue(gandct_pred):
    """Map GANDCTAnalysis prediction to FakeClue label convention.

    GANDCTAnalysis: 0 = real, >= 1 = fake
    FakeClue:       0 = fake, 1 = real
    """
    return 1 if gandct_pred == 0 else 0


def interpret_predictions(predictions):
    """Return (class_indices, confidences) from raw model output."""
    if predictions.shape[-1] > 1:
        class_indices = tf.argmax(predictions, axis=1).numpy()
        confidences = tf.reduce_max(predictions, axis=1).numpy()
    else:
        raw = predictions.numpy().flatten()
        class_indices = np.round(raw).astype(int)
        confidences = np.where(class_indices == 1, raw, 1.0 - raw)
    return class_indices, confidences


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify_split(model, model_type, split, data_dir, output_dir, batch_size,
                   limit):
    json_path = data_dir / "data_json" / f"{split}.json"
    split_dir = data_dir / split

    with open(json_path) as f:
        entries = json.load(f)

    if limit is not None:
        entries = entries[:limit]

    results = []
    failed = 0

    for batch_start in tqdm(range(0, len(entries), batch_size),
                            desc=f"{split} ({model_type})"):
        batch_entries = entries[batch_start:batch_start + batch_size]
        batch_images = []
        batch_indices = []

        for i, entry in enumerate(batch_entries):
            img_path = split_dir / entry["image"]
            try:
                img = load_and_resize(img_path)
                batch_images.append(img)
                batch_indices.append(i)
            except Exception as e:
                logger.warning("Failed to load %s: %s", entry["image"], e)
                results.append({
                    "image": entry["image"],
                    "ground_truth": entry["label"],
                    "predicted_label": -1,
                    "confidence": None,
                    "model_type": model_type,
                    "category": entry.get("cate", "unknown"),
                    "error": str(e),
                })
                failed += 1

        if not batch_images:
            continue

        batch_array = np.stack(batch_images)
        predictions = model(batch_array, training=False)
        class_indices, confidences = interpret_predictions(predictions)

        for j, idx in enumerate(batch_indices):
            entry = batch_entries[idx]
            gandct_pred = int(class_indices[j])
            results.append({
                "image": entry["image"],
                "ground_truth": entry["label"],
                "predicted_label": gandct_to_fakeclue(gandct_pred),
                "confidence": round(float(confidences[j]), 6),
                "model_type": model_type,
                "category": entry.get("cate", "unknown"),
                "error": None,
            })

    out_path = output_dir / f"{split}_frequency_{model_type}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results, split, model_type, failed)
    logger.info("Results written to %s", out_path)
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results, split, model_type, failed):
    valid = [r for r in results if r["error"] is None]
    total = len(results)
    correct = sum(1 for r in valid if r["ground_truth"] == r["predicted_label"])
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
    print(f"Split: {split} | Model: {model_type}")
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

    parser = argparse.ArgumentParser(
        description="Classify FakeClue images using GANDCTAnalysis models.")
    parser.add_argument(
        "--model-type", required=True, choices=MODEL_TYPES,
        help="Pre-trained model variant to use.")
    parser.add_argument(
        "--split", default="test", choices=("train", "test", "both"),
        help="FakeClue split to process (default: test).")
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Images per batch (default: 16).")
    project_root = script_dir / ".." / ".." / ".." / ".."
    parser.add_argument(
        "--models-dir", type=Path, default=project_root / "models" / "GANDCTAnalysis",
        help="Directory containing ffhq/ and mean_var/ folders.")
    parser.add_argument(
        "--data-dir", type=Path,
        default=project_root / "data" / "external" / "FakeClue",
        help="FakeClue dataset root directory.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for JSON results (default: DATA_DIR/data_json).")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only this many images (for debugging).")
    args = parser.parse_args()

    args.models_dir = args.models_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    if args.output_dir is None:
        args.output_dir = args.data_dir / "data_json" / "GANDCTAnalysis"
    else:
        args.output_dir = args.output_dir.resolve()

    return args


def main():
    args = parse_args()

    logger.info("Loading model: %s", args.model_type)
    model = load_model(args.model_type, args.models_dir)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        classify_split(model, args.model_type, split, args.data_dir,
                       args.output_dir, args.batch_size, args.limit)


if __name__ == "__main__":
    main()
