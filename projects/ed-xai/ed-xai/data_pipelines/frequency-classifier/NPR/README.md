# NPR Classifier for FakeClue Dataset

Classifies FakeClue dataset images as fake or real using the pre-trained model
from [Tan et al., "Rethinking the Up-Sampling Operations in CNN-based Generative
Network for Generalizable Deepfake Detection" (CVPR 2024)](https://arxiv.org/abs/2312.10461).

## Requirements

- Python 3.11+
- CUDA-capable GPU (recommended)
- Dependencies: see `requirements.txt`

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Model Checkpoint

Copy or symlink the pre-trained NPR weights from the cloned
[NPR repo](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
to `../../../../models/NPR/NPR.pth`:

```bash
mkdir -p ../../../../models/NPR
cp /path/to/NPR-DeepfakeDetection/NPR.pth ../../../../models/NPR/
```

Two checkpoints are available in the repo: `NPR.pth` and
`model_epoch_last_3090.pth`.

## Usage

### Basic (test split)

```bash
python classify.py \
    --checkpoint ../../../../models/NPR/NPR.pth \
    --split test
```

### Quick test run

```bash
python classify.py \
    --checkpoint ../../../../models/NPR/NPR.pth \
    --split test \
    --limit 20
```

### Both splits

```bash
python classify.py \
    --checkpoint ../../../../models/NPR/NPR.pth \
    --split both
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | *required* | Path to NPR `.pth` checkpoint file |
| `--split` | `test` | FakeClue split: `train`, `test`, or `both` |
| `--data-dir` | `../../../../data/external/FakeClue` | FakeClue dataset root |
| `--output-dir` | `DATA_DIR/data_json/NPR` | Output directory for JSON results |
| `--batch-size` | `32` | Batch size for inference |
| `--limit` | None | Process only N images (for debugging) |

## Output Format

Results are written to `{split}_frequency_npr.json`:

```json
[
  {
    "image": "ff++/fake/Deepfakes/c23/frames/856_881/1066.png",
    "ground_truth": 0,
    "predicted_label": 0,
    "confidence": 0.9873,
    "model": "npr",
    "category": "deepfake",
    "error": null
  }
]
```

- `ground_truth`: label from the FakeClue dataset (0 = fake, 1 = real)
- `predicted_label`: model prediction mapped to FakeClue convention (0 = fake, 1 = real)
- `confidence`: model confidence score in [0.5, 1]
- `error`: error message if the image could not be processed, otherwise null

A summary with accuracy, confusion matrix, and per-category breakdown is
printed to stdout after processing.

## Resume Support

Processing state is checkpointed every 500 images to
`.{split}_npr_checkpoint.json`. If the process is interrupted, re-running the
same command will resume from the last checkpoint.

## Label Conventions

The two projects use opposite conventions:

- **FakeClue**: 0 = fake, 1 = real
- **NPR**: class 0 = real, class 1 = fake (high sigmoid score = fake)

This script maps model predictions to the FakeClue convention automatically.

## Architecture

NPR uses Neighboring Pixel Relationships for deepfake detection:

1. **NPR feature**: The input image is downsampled by 2x (nearest neighbor) then
   upsampled back. The difference between original and reconstructed captures
   artifacts from up-sampling operations in generative networks.
2. **Truncated ResNet-50**: Only `layer1` and `layer2` are used (no `layer3`/`layer4`),
   feeding into adaptive average pooling and a linear classifier.
3. **Output**: Single logit passed through sigmoid for binary prediction.

## References

- Tan et al., "Rethinking the Up-Sampling Operations in CNN-based Generative
  Network for Generalizable Deepfake Detection," CVPR 2024.
  https://arxiv.org/abs/2312.10461
