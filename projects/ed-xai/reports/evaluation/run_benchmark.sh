#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ED_XAI="$SCRIPT_DIR/../../ed-xai"
DATA="$SCRIPT_DIR/../../data/external/FakeClue/data_json"
OUTPUT="$SCRIPT_DIR/benchmark_output"

cd "$ED_XAI"

python -m benchmarking.run \
    --model "FakeVLM-Baseline" "$SCRIPT_DIR/baseline/eval_results.json" \
    --model "FakeVLM-FFT"      "$SCRIPT_DIR/fft/eval_results.json" \
    --dataset "FakeClue-Test"      "$DATA/test.json" \
    --dataset "FakeClue-Frequency" "$DATA/test_frequency.json" \
    --eval "FakeVLM-Baseline" "FakeClue-Test" \
    --eval "FakeVLM-FFT"      "FakeClue-Frequency" \
    --output-dir "$OUTPUT"
