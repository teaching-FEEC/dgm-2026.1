# Frequency-Domain Classifier for FakeClue Dataset

Classifies FakeClue dataset images as fake or real using pre-trained
GANDCTAnalysis models from [Frank et al., "Leveraging Frequency Analysis
for Deep Fake Image Recognition" (ICML 2020)](https://arxiv.org/abs/2003.08685).

## Requirements

- Python 3.12 (recommended; the pre-trained models require the legacy Keras 2
  API via `tf-keras`, which is compatible with TF 2.16+)
- Dependencies: see `requirements.txt`

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic (test split with ridge_dct model)

```bash
python classify.py --model-type ridge_dct --split test
```

### All models on both splits

```bash
for model in ridge_pixel ridge_dct lasso_dct; do
    python classify.py --model-type $model --split both
done
```

### Custom paths

```bash
python classify.py \
    --model-type ridge_pixel \
    --split test \
    --models-dir /path/to/models \
    --data-dir /path/to/FakeClue \
    --output-dir /path/to/output
```

### Quick test run

```bash
python classify.py --model-type ridge_pixel --split test --limit 100
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model-type` | *required* | Model variant: `ridge_pixel`, `ridge_dct`, or `lasso_dct` |
| `--split` | `test` | FakeClue split: `train`, `test`, or `both` |
| `--batch-size` | `16` | Images per batch |
| `--models-dir` | `../../../../models/GANDCTAnalysis` | Directory containing `ffhq/` and `mean_var/` |
| `--data-dir` | `../../../../data/external/FakeClue` | FakeClue dataset root |
| `--output-dir` | `DATA_DIR/data_json/GANDCTAnalysis` | Output directory for JSON results |
| `--limit` | None | Process only N images (for debugging) |

Default paths are relative to the script location and resolve correctly when
the project follows the expected directory layout.

## Output Format

Results are written to `{split}_frequency_{model_type}.json`:

```json
[
  {
    "image": "ff++/fake/Deepfakes/c23/frames/856_881/1066.png",
    "ground_truth": 0,
    "predicted_label": 0,
    "confidence": 0.9873,
    "model_type": "ridge_dct",
    "category": "deepfake",
    "error": null
  }
]
```

- `ground_truth`: label from the FakeClue dataset (0 = fake, 1 = real)
- `predicted_label`: model prediction mapped to FakeClue convention (0 = fake, 1 = real)
- `confidence`: model confidence score in [0, 1]
- `error`: error message if the image could not be processed, otherwise null

A summary with accuracy, confusion matrix, and per-category breakdown is
printed to stdout after processing.

## Label Conventions

The two projects use opposite conventions:

- **FakeClue**: 0 = fake, 1 = real
- **GANDCTAnalysis**: 0 = real, >= 1 = fake

This script maps model predictions to the FakeClue convention automatically.

## Models

Pre-trained models from the [GANDCTAnalysis Google Drive](https://drive.google.com/drive/folders/1QjQnqMQnQOoIPwgzdJVJGwYzdReKqc0N),
trained on FFHQ 1024x1024 images:

- **ridge_pixel**: Ridge regression on pixel values. No mean/var files needed.
- **ridge_dct**: Ridge regression on DCT coefficients (log-scaled, normalized).
  Requires `mean_var/ffhq_mean_var/`.
- **lasso_dct**: Lasso regression on DCT coefficients. Same mean/var requirement.

## Notes

- Images are resized to 1024x1024 to match the model input size. This may
  introduce artifacts for very small images.
- The models were trained on FFHQ (faces). Performance on non-face categories
  (documents, satellites) may be limited due to domain mismatch.
- Processing the full train split (~104K images) can take several hours.

## References

- Frank et al., "Leveraging Frequency Analysis for Deep Fake Image
  Recognition," ICML 2020. https://arxiv.org/abs/2003.08685
- Wen et al., "Spot the Fake: Large Multimodal Model-Based Synthetic Image
  Detection with Artifact Explanation," 2025. https://arxiv.org/abs/2503.14905
