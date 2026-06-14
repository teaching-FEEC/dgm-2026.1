"""
Per-category error analysis across model configurations.

Run from ed-xai/ed-xai/:

    python -m benchmarking.error_analysis \
        --dataset ../data/external/FakeClue/data_json/test.json \
        --model "Baseline"      ../reports/evaluation/baseline/eval_results.json \
        --model "Magnitude"     ../reports/evaluation/fft/eval_results.json \
        --model "Phase"         ../reports/evaluation/fft_phase/eval_results.json \
        --model "LoRA Vicuna"   ../reports/evaluation/lora_vicuna/eval_results.json \
        --model "LoRA Vic+Proj" ../reports/evaluation/lora_vicuna_projector/eval_results.json
"""
import argparse
import json
from collections import Counter


def parse_prediction(response: str) -> int:
    """Match the benchmarker logic: check first sentence for real/fake,
    return -1 when undecided so unrecognized responses count as errors."""
    idx = response.find("ASSISTANT:")
    text = response[idx:] if idx >= 0 else response
    first_sentence = text.split(".")[0].lower()
    if "real" in first_sentence:
        return 1
    if "fake" in first_sentence:
        return 0
    return -1


def main():
    parser = argparse.ArgumentParser(description="Per-category error analysis.")
    parser.add_argument(
        "--dataset", required=True,
        help="Path to FakeClue test JSON (with 'cate' field).",
    )
    parser.add_argument(
        "--model", nargs=2, metavar=("NAME", "RESULTS_JSON"),
        action="append", required=True,
        help="Model name and eval_results.json path. Repeatable.",
    )
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)
    img_to_cat = {entry["image"]: entry["cate"] for entry in dataset}

    categories = sorted(set(img_to_cat.values()))
    cat_totals = Counter(img_to_cat.values())

    model_names = []
    model_errors = []

    for name, path in args.model:
        with open(path) as f:
            results = json.load(f)

        errors = Counter()
        for entry in results:
            pred = parse_prediction(entry["response"])
            if pred != entry["ground_truth_label"]:
                cat = img_to_cat.get(entry["image"], "unknown")
                errors[cat] += 1

        model_names.append(name)
        model_errors.append(errors)

    header = ["Category", "Images"] + model_names
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))

    totals = [0] * len(model_names)
    for cat in categories:
        row = [cat, str(cat_totals[cat])]
        for i, errors in enumerate(model_errors):
            count = errors.get(cat, 0)
            totals[i] += count
            row.append(str(count))
        print(" | ".join(row))

    total_images = sum(cat_totals.values())
    row = ["**Total**", f"**{total_images}**"]
    for t in totals:
        row.append(f"**{t}**")
    print(" | ".join(row))


if __name__ == "__main__":
    main()
