import re
import json
from pathlib import Path
from collections import defaultdict

EVALUATION_DIR = Path(__file__).parent
OUTPUT_JSON = EVALUATION_DIR / "errors_summary.json"
OUTPUT_RECURRENT_JSON = EVALUATION_DIR / "recurrent_errors.json"

LABEL_MAP = {
        0: "fake", 
        1: "real"
    }


def extract_prediction(response: str) -> str | None:
    match = re.search(r"ASSISTANT:\s*(This is a (fake|real) image)", response, re.IGNORECASE)
    if match:
        return match.group(2).lower()
    return None


def check_folder(folder: Path) -> dict:
    json_path = folder / "eval_results.json"
    with open(json_path) as f:
        data = json.load(f)

    errors = {
            "false_positive": [], 
            "false_negative": [], 
            "unrecognized": []
        }

    for entry in data:
        image = entry["image"]
        ground_truth = LABEL_MAP[entry["ground_truth_label"]]
        prediction = extract_prediction(entry["response"])

        if prediction is None:
            errors["unrecognized"].append(image)
        elif prediction != ground_truth:
            key = "false_positive" if ground_truth == "real" else "false_negative"
            errors[key].append(image)

    total = len(data)
    total_errors = sum(len(v) for v in errors.values())
    correct = total - total_errors

    return {
        "total": total,
        "correct": correct,
        "errors": total_errors,
        "accuracy": round(correct / total, 4),
        "evidence": errors,
    }


def build_recurrent_errors(results: dict) -> dict:
    categories = ("false_positive", "false_negative", "unrecognized")
    counts: dict[str, dict[str, list[str]]] = {c: defaultdict(list) for c in categories}

    for folder_name, data in results.items():
        for category in categories:
            for image in data["evidence"][category]:
                counts[category][image].append(folder_name)

    recurrent = {}
    for category in categories:
        recurrent[category] = sorted(
            [{"image": img, "count": len(folders), "folders": folders}
             for img, folders in counts[category].items()],
            key=lambda x: x["count"],
            reverse=True,
        )

    return recurrent


def main():
    folders = sorted(
        p for p in EVALUATION_DIR.iterdir()
        if p.is_dir() and (p / "eval_results.json").exists()
    )

    results = {}
    for folder in folders:
        name = folder.name
        print(f"Processing '{name}'...")
        results[name] = check_folder(folder)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    recurrent = build_recurrent_errors(results)
    with open(OUTPUT_RECURRENT_JSON, "w") as f:
        json.dump(recurrent, f, indent=2)

    print(f"\nSaved to {OUTPUT_JSON}")
    print(f"Saved to {OUTPUT_RECURRENT_JSON}\n")

    for name, data in results.items():
        print(f"[{name}] total={data['total']}  correct={data['correct']}  errors={data['errors']}  accuracy={data['accuracy']:.1%}")


if __name__ == "__main__":
    main()
