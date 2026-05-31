import argparse
import copy
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_tp_set(results_path, category):
    """Load true positive image paths from a classifier results file.

    Returns the set of image paths where ground_truth == 0 (fake)
    and predicted_label == 0 (correctly classified as fake).
    """
    with open(results_path) as f:
        results = json.load(f)

    tp_images = set()
    for r in results:
        if r is None:
            continue
        if (r.get("error") is None
                and r.get("ground_truth") == 0
                and r.get("predicted_label") == 0
                and r.get("category") == category):
            tp_images.add(r["image"])

    return tp_images


def append_text(conversations, augment_text):
    """Append augment_text to the last GPT response."""
    value = conversations[-1]["value"].rstrip()
    if not value.endswith((".", "!", "?", '"')):
        value += "."
    value += " " + augment_text
    conversations[-1]["value"] = value


def augment_split(config, split, data_dir):
    original_path = data_dir / "data_json" / f"{split}.json"
    with open(original_path) as f:
        entries = json.load(f)

    logger.info("Loaded %d entries from %s", len(entries), original_path)

    augment_text = config["text"]

    tp_sets = {}
    for cat_config in config["categories"]:
        category = cat_config["category"]
        results_file = cat_config["results_file"].replace("{split}", split)
        results_path = (data_dir / "data_json" / cat_config["results_dir"]
                        / results_file)

        if not results_path.exists():
            logger.warning("Results file not found, skipping category '%s': %s",
                           category, results_path)
            continue

        tp_set = load_tp_set(results_path, category)
        tp_sets[category] = tp_set
        logger.info("  %s: %d TPs loaded from %s",
                     category, len(tp_set), results_path.name)

    augmented = copy.deepcopy(entries)
    augmented_counts = Counter()
    fake_counts = Counter()

    for entry in augmented:
        category = entry.get("cate", "unknown")

        if entry["label"] == 0:
            fake_counts[category] += 1
            if category in tp_sets and entry["image"] in tp_sets[category]:
                append_text(entry["conversations"], augment_text)
                augmented_counts[category] += 1

    out_path = data_dir / "data_json" / f"{split}_frequency.json"
    with open(out_path, "w") as f:
        json.dump(augmented, f, indent=4)

    print_summary(entries, augmented_counts, fake_counts, split, out_path)


def print_summary(entries, augmented_counts, fake_counts, split, out_path):
    total = len(entries)
    total_fake = sum(fake_counts.values())
    total_augmented = sum(augmented_counts.values())

    print(f"\n{'='*60}")
    print(f"Split: {split} | Frequency label augmentation")
    print(f"{'='*60}")
    print(f"Total entries: {total}")
    print(f"Total fake: {total_fake}")
    print(f"Total augmented: {total_augmented} ({total_augmented/total_fake:.1%} "
          f"of fake images)")
    print(f"\nPer-category:")
    for cat in sorted(set(list(fake_counts.keys()) + list(augmented_counts.keys()))):
        aug = augmented_counts.get(cat, 0)
        fake = fake_counts.get(cat, 0)
        pct = f"{aug/fake:.1%}" if fake > 0 else "N/A"
        print(f"  {cat:<15} {aug:>6}/{fake:<6} ({pct})")
    print(f"\nOutput: {out_path}")
    print(f"{'='*60}\n")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir / ".." / ".." / ".."

    parser = argparse.ArgumentParser(
        description="Augment FakeClue labels with frequency-domain artifact "
                    "sentences for true positive fake images.")
    parser.add_argument(
        "--split", required=True, choices=("train", "test"),
        help="FakeClue split to process.")
    parser.add_argument(
        "--config", type=Path, default=script_dir / "augment_config.json",
        help="Path to config JSON (default: augment_config.json).")
    parser.add_argument(
        "--data-dir", type=Path,
        default=project_root / "data" / "external" / "FakeClue",
        help="FakeClue dataset root directory.")
    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.config = args.config.resolve()
    return args


def main():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    logger.info("Config: %s", args.config)
    logger.info("Text: \"%s\"", config["text"])
    logger.info("Categories: %d", len(config["categories"]))

    augment_split(config, args.split, args.data_dir)


if __name__ == "__main__":
    main()
