"""
Smoke test for the Benchmarker class using temporary JSON files.
Run from the ed-xai/ed-xai/ directory:

    python -m benchmarking.test_smoke
"""
import json
import random
import tempfile
from pathlib import Path

from benchmarking import Benchmarker
from benchmarking.metrics import accuracy, auc, avg_precision, css, f1, rouge_l


_FAKE_RESPONSES = [
    "This image is clearly fake. It shows deepfake artifacts and manipulated facial features.",
    "The image appears synthetic and artificially generated.",
    "This looks like a forged image with altered regions.",
    "Deepfake artifacts are visible in this manipulated image.",
]
_REAL_RESPONSES = [
    "This image looks real and authentic with no signs of manipulation.",
    "The image appears genuine and original.",
    "This is an authentic photograph with no detected artifacts.",
    "The image is real with natural lighting and textures.",
]


def make_dataset_json(path: Path, n: int = 100, seed: int = 0, balance: float = 0.5):
    """Write a fake FakeClue-format dataset JSON."""
    rng = random.Random(seed)
    entries = []
    for i in range(n):
        label = 0 if rng.random() < balance else 1
        ref = rng.choice(_FAKE_RESPONSES if label == 0 else _REAL_RESPONSES)
        entries.append({
            "image": f"test/img_{i:05d}.jpg",
            "label": label,
            "conversations": [
                {"from": "human", "value": "Is this image real or fake?"},
                {"from": "gpt", "value": ref},
            ],
        })
    path.write_text(json.dumps(entries, indent=2))
    return entries


def make_results_json(path: Path, dataset_entries: list, accuracy: float = 0.8, seed: int = 1):
    """Write a fake eval.py-format results JSON with the given accuracy."""
    rng = random.Random(seed)
    results = []
    for entry in dataset_entries:
        true_label = entry["label"]
        correct = rng.random() < accuracy
        predicted_label = true_label if correct else (1 - true_label)
        response = rng.choice(_FAKE_RESPONSES if predicted_label == 0 else _REAL_RESPONSES)
        results.append({
            "image": entry["image"],
            "ground_truth_label": true_label,
            "question": "Is this image real or fake?",
            "response": response,
        })
    path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Create dataset JSONs
        test_entries  = make_dataset_json(tmp / "test.json",  n=200, seed=10)
        train_entries = make_dataset_json(tmp / "train.json", n=500, seed=20)

        # Create model result JSONs
        make_results_json(tmp / "weak_test.json",   test_entries,  accuracy=0.65, seed=1)
        make_results_json(tmp / "strong_test.json",  test_entries,  accuracy=0.88, seed=2)
        make_results_json(tmp / "weak_train.json",  train_entries, accuracy=0.65, seed=3)
        make_results_json(tmp / "strong_train.json", train_entries, accuracy=0.88, seed=4)

        benchmarker = Benchmarker(
            models=[
                ("FakeVLM-Base",     tmp / "weak_test.json"),
                ("FakeVLM-Extended", tmp / "strong_test.json"),
            ],
            datasets=[
                ("FakeClue-Test",  tmp / "test.json"),
                ("FakeClue-Train", tmp / "train.json"),
            ],
            metrics=[accuracy, auc, f1, avg_precision, css, rouge_l],
            output_dir="./benchmark_output",
        )

        print("Running benchmark...")
        benchmarker.run()

        print("\n=== Results ===")
        print(benchmarker.results.to_string())

        print("\nExporting...")
        benchmarker.export_xlsx().export_png()

        print("\nDone. Check ./benchmark_output/")
