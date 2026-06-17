import argparse
import json
import logging
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

MODEL_NAMES = ("rn50", "rn50_mod", "clip")


# ---------------------------------------------------------------------------
# CLIP wrapper
# ---------------------------------------------------------------------------

class CLIPClassifier(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        import clip as clip_module
        self.encoder, _ = clip_module.load("ViT-L/14", device=device)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder.encode_image(x)
        return self.fc(features.float())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(model_name, device="cpu"):
    if model_name in ("rn50", "rn50_mod"):
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == "clip":
        model = CLIPClassifier(device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def strip_module_prefix(state_dict):
    """Remove 'module.' prefix from DistributedDataParallel checkpoints."""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k.removeprefix("module.")] = v
    return new_sd


def load_model(checkpoint_path, model_name, device):
    model = build_model(model_name, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    state_dict = strip_module_prefix(checkpoint["model_state_dict"])

    if model_name == "clip":
        fc_state = OrderedDict()
        for k, v in state_dict.items():
            fc_state[k.removeprefix("fc.")] = v
        model.fc.load_state_dict(fc_state)
    else:
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image preprocessing (replicates test_augment from FakeImageDetection)
# ---------------------------------------------------------------------------

def get_test_transform(model_name):
    mean = CLIP_MEAN if model_name == "clip" else IMAGENET_MEAN
    std = CLIP_STD if model_name == "clip" else IMAGENET_STD
    return transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_and_transform(path, transform):
    img = Image.open(path).convert("RGB")
    return transform(img)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FakeClueDataset(Dataset):
    def __init__(self, entries, split_dir, transform):
        self.entries = entries
        self.split_dir = split_dir
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = self.split_dir / entry["image"]
        try:
            tensor = load_and_transform(img_path, self.transform)
            return idx, tensor, ""
        except Exception as e:
            return idx, torch.zeros(3, 224, 224), str(e)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def fid_to_fakeclue(fid_pred):
    """Map FakeImageDetection prediction to FakeClue label convention.

    FakeImageDetection: 0 = real, 1 = fake
    FakeClue:           0 = fake, 1 = real
    """
    return 1 if fid_pred == 0 else 0


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify_split(model, model_name, checkpoint_name, split, data_dir,
                   output_dir, batch_size, limit, device, num_workers):
    json_path = data_dir / "data_json" / f"{split}.json"
    split_dir = data_dir / split

    with open(json_path) as f:
        entries = json.load(f)

    if limit is not None:
        entries = entries[:limit]

    transform = get_test_transform(model_name)
    dataset = FakeClueDataset(entries, split_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")

    results = [None] * len(entries)
    failed = 0

    for batch_indices, batch_tensors, batch_errors in tqdm(
            loader, desc=f"{split} ({checkpoint_name})"):
        valid_mask = [err == "" for err in batch_errors]
        indices = batch_indices.tolist()

        for i, idx in enumerate(indices):
            if not valid_mask[i]:
                entry = entries[idx]
                logger.warning("Failed to load %s: %s", entry["image"],
                               batch_errors[i])
                results[idx] = {
                    "image": entry["image"],
                    "ground_truth": entry["label"],
                    "predicted_label": -1,
                    "confidence": None,
                    "model": checkpoint_name,
                    "category": entry.get("cate", "unknown"),
                    "error": batch_errors[i],
                }
                failed += 1

        valid_tensors = batch_tensors[torch.tensor(valid_mask)]
        if len(valid_tensors) == 0:
            continue

        valid_tensors = valid_tensors.to(device)
        with torch.no_grad():
            logits = model(valid_tensors).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

        preds = (probs > 0.5).astype(int)
        confidences = np.where(preds == 1, probs, 1.0 - probs)

        j = 0
        for i, idx in enumerate(indices):
            if not valid_mask[i]:
                continue
            entry = entries[idx]
            results[idx] = {
                "image": entry["image"],
                "ground_truth": entry["label"],
                "predicted_label": fid_to_fakeclue(int(preds[j])),
                "confidence": round(float(confidences[j]), 6),
                "model": checkpoint_name,
                "category": entry.get("cate", "unknown"),
                "error": None,
            }
            j += 1

    out_path = output_dir / f"{split}_frequency_{checkpoint_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results, split, checkpoint_name, failed)
    logger.info("Results written to %s", out_path)
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results, split, checkpoint_name, failed):
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
    print(f"Split: {split} | Model: {checkpoint_name}")
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
        description="Classify FakeClue images using FakeImageDetection models.")
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to .pth checkpoint file.")
    parser.add_argument(
        "--model-name", default="rn50", choices=MODEL_NAMES,
        help="Model architecture (default: rn50).")
    parser.add_argument(
        "--split", default="test", choices=("train", "test", "both"),
        help="FakeClue split to process (default: test).")
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Images per batch (default: 64).")
    parser.add_argument(
        "--data-dir", type=Path,
        default=project_root / "data" / "external" / "FakeClue",
        help="FakeClue dataset root directory.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: DATA_DIR/data_json/FakeImageDetection).")
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes for parallel image loading (default: 4).")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only this many images (for debugging).")
    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.output_dir is None:
        args.output_dir = args.data_dir / "data_json" / "FakeImageDetection"
    else:
        args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_name = args.checkpoint.stem

    logger.info("Loading model: %s (%s) on %s", checkpoint_name,
                args.model_name, device)
    model = load_model(args.checkpoint, args.model_name, device)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        classify_split(model, args.model_name, checkpoint_name, split,
                       args.data_dir, args.output_dir, args.batch_size,
                       args.limit, device, args.num_workers)


if __name__ == "__main__":
    main()
