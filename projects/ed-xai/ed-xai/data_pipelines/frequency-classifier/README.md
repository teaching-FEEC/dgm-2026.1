# Frequency-Domain Deepfake Detection for FakeClue

Evaluates frequency-domain deepfake detection models on the FakeClue dataset
and augments FakeClue explainability labels with frequency artifact sentences
for true positive detections.

## Workflow

1. **Classify** — run each frequency-domain classifier on FakeClue to produce
   per-image predictions (JSON files under `data_json/{ModelName}/`).
2. **Augment** — use `augment_labels.py` to append a frequency sentence to
   FakeClue labels for fake images that were correctly identified by the best
   classifier for each category.

## Best Frequency-Domain Model per Category

| Category | Model | Technique | TPs (train) | Coverage |
|---|---|---|---|---|
| deepfake | ridge_dct | DCT | 19,066 | 99.5% |
| satellite | spai | FFT | 8,397 | 87.9% |
| object | spai | FFT | 8,304 | 75.6% |
| animal | spai | FFT | 5,983 | 75.7% |
| human | spai | FFT | 4,577 | 69.2% |
| scene | spai | FFT | 2,892 | 61.6% |
| doc | rn50_modft_spectralmask | spectral masking | 1,785 | 18.9% |
| **Total** | | | **51,004** | **74.6%** |

## Label Augmentation

### Usage

```bash
python augment_labels.py --split train
python augment_labels.py --split test
python augment_labels.py --split train --config custom_config.json
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--split` | *required* | `train` or `test` |
| `--config` | `augment_config.json` | Path to config file |
| `--data-dir` | `../../../../data/external/FakeClue` | FakeClue dataset root |

### Config Format

`augment_config.json` maps each FakeClue category to the classifier whose
true positive results should be used:

```json
{
  "text": "The image also presents artifacts in the frequency domain.",
  "categories": [
    {"category": "deepfake",  "results_dir": "GANDCTAnalysis", "results_file": "{split}_frequency_ridge_dct.json"},
    {"category": "animal",    "results_dir": "SPAI",           "results_file": "{split}_frequency_spai.json"}
  ]
}
```

- `text`: sentence appended to the GPT response for true positive fake images
- `category`: FakeClue category name
- `results_dir`: subdirectory under `data_json/` containing classifier results
- `results_file`: classifier output filename (`{split}` is replaced at runtime)

### Output

A single file per split: `data_json/{split}_frequency.json`

This is a copy of the original FakeClue `{split}.json` where fake images
identified as true positives by the configured classifier have the frequency
sentence appended to their GPT response.

### Prerequisites

Classifier results must exist before running the augmentation script. If
results for a category are missing, that category is skipped with a warning.

## Running All Classifiers

Two convenience scripts run all 17 classifiers and then augment the labels:

```bash
./run_test_split.sh    # test split (5K images, ~15 min)
./run_train_split.sh   # train split (104K images, several hours — SPAI is slow)
```

Each script runs GANDCTAnalysis (3 models), FakeImageDetection (12 models),
SPAI (1 model), and NPR (1 model) in sequence, then calls `augment_labels.py`
to produce the final `{split}_frequency.json`. The scripts use `set -e` and
will stop on the first error.

## Classifiers

Each subdirectory contains a self-contained classifier with its own README:

| Directory | Model | Paper |
|---|---|---|
| `GANDCTAnalysis/` | ridge_dct, ridge_pixel, lasso_dct | Frank et al., 2020 |
| `FakeImageDetection/` | ResNet-50, CLIP ViT-L/14 (various masks) | Doloriel and Cheung, 2024 |
| `SPAI/` | PatchBasedMFViT (FFT + spectral attention) | Karageorgiou et al., CVPR 2025 |
| `NPR/` | ResNet-50 + NPR features | Tan et al., CVPR 2024 |
