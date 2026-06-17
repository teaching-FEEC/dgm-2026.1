"""
CLI entry point for the Benchmarker.

Run from ed-xai/ed-xai/:

    python -m benchmarking.run \
        --model "FakeVLM-Baseline" path/to/baseline/eval_results.json \
        --model "FakeVLM-FFT"      path/to/fft/eval_results.json \
        --dataset "FakeClue-Test"      ../data/external/FakeClue/data_json/test.json \
        --dataset "FakeClue-Frequency" ../data/external/FakeClue/data_json/test_frequency.json \
        --eval "FakeVLM-Baseline" "FakeClue-Test" \
        --eval "FakeVLM-FFT"      "FakeClue-Frequency" \
        --output-dir ./benchmark_output

If --eval is omitted, all model x dataset combinations are evaluated.
"""
import argparse

from benchmarking import Benchmarker
from benchmarking.metrics import accuracy, f1, rouge_l, css


def main():
    parser = argparse.ArgumentParser(description="Run benchmarking evaluation.")
    parser.add_argument(
        "--model", nargs=2, metavar=("NAME", "RESULTS_JSON"),
        action="append", required=True,
        help="Model name and path to its eval.py output JSON. Repeatable.",
    )
    parser.add_argument(
        "--dataset", nargs=2, metavar=("NAME", "DATASET_JSON"),
        action="append", required=True,
        help="Dataset name and path to its FakeClue JSON. Repeatable.",
    )
    parser.add_argument(
        "--eval", nargs=2, metavar=("MODEL_NAME", "DATASET_NAME"),
        action="append", default=None,
        help="Explicit (model, dataset) pair to evaluate. Repeatable. "
             "If omitted, all combinations are run.",
    )
    parser.add_argument("--output-dir", default="./benchmark_output")
    args = parser.parse_args()

    benchmarker = Benchmarker(
        models=[(name, path) for name, path in args.model],
        datasets=[(name, path) for name, path in args.dataset],
        metrics=[accuracy, f1, rouge_l, css],
        eval_pairs=[(m, d) for m, d in args.eval] if args.eval else None,
        output_dir=args.output_dir,
    )

    print("Running benchmark...")
    benchmarker.run()

    print("\n=== Results ===")
    print(benchmarker.results.to_string())

    print("\nExporting...")
    benchmarker.export()

    print(f"\nDone. Check {args.output_dir}/")


if __name__ == "__main__":
    main()
