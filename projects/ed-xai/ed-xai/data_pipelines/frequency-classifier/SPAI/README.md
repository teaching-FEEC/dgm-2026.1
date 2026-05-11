# SPAI Classifier for FakeClue Dataset

Classifies FakeClue dataset images as fake or real using the pre-trained model
from [Karageorgiou et al., "Any-Resolution AI-Generated Image Detection by
Spectral Learning" (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html).

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

Download the pre-trained SPAI weights from
[Google Drive](https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view?usp=sharing)
and place at `../../../../models/SPAI/spai.pth`.

## Usage

### Basic (test split)

```bash
python classify.py \
    --checkpoint ../../../../models/SPAI/spai.pth \
    --split test
```

### Quick test run

```bash
python classify.py \
    --checkpoint ../../../../models/SPAI/spai.pth \
    --split test \
    --limit 20
```

### Both splits

```bash
python classify.py \
    --checkpoint ../../../../models/SPAI/spai.pth \
    --split both
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | *required* | Path to `spai.pth` checkpoint file |
| `--split` | `test` | FakeClue split: `train`, `test`, or `both` |
| `--data-dir` | `../../../../data/external/FakeClue` | FakeClue dataset root |
| `--output-dir` | `DATA_DIR/data_json/SPAI` | Output directory for JSON results |
| `--limit` | None | Process only N images (for debugging) |

## Output Format

Results are written to `{split}_frequency_spai.json`:

```json
[
  {
    "image": "ff++/fake/Deepfakes/c23/frames/856_881/1066.png",
    "ground_truth": 0,
    "predicted_label": 0,
    "confidence": 0.9873,
    "model": "spai",
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

## Label Conventions

The two projects use opposite conventions:

- **FakeClue**: 0 = fake, 1 = real
- **SPAI**: class 0 = real, class 1 = fake (high sigmoid score = fake)

This script maps model predictions to the FakeClue convention automatically.

## Architecture

SPAI uses spectral learning for deepfake detection:

1. **Frequency decomposition**: Images are split into low-frequency and
   high-frequency components via FFT with a circular mask (radius 16).
2. **ViT-B/16 backbone**: All three components (original, low-freq, high-freq)
   are processed by a Vision Transformer to extract features from all 12
   intermediate layers.
3. **Spectral reconstruction similarity**: Cosine similarity between the
   original and frequency component features quantifies how well the image's
   spectral distribution matches the learned model of real images.
4. **Spectral context attention**: For arbitrary-resolution images, 224x224
   patches are extracted and aggregated via cross-attention.
5. **Classification head**: MLP outputs a binary fake/real prediction.

## Notes

- Images are processed at their **original resolution** (no resize). They are
  padded to a minimum of 224x224 and split into non-overlapping 224x224 patches.
- Each image is processed individually (batch size 1) due to varying resolutions.
- Each patch requires 3 ViT forward passes (original + low-freq + high-freq),
  so large images with many patches will be slower.
- GPU memory usage should stay under 8GB for typical image sizes.
- The bundled `spai_model/` package contains trimmed source from the
  [SPAI repository](https://github.com/mever-team/spai), keeping only
  inference-related code.

## References

- Karageorgiou et al., "Any-Resolution AI-Generated Image Detection by
  Spectral Learning," CVPR 2025.
  https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html
- Wen et al., "Spot the Fake: Large Multimodal Model-Based Synthetic Image
  Detection with Artifact Explanation," 2025. https://arxiv.org/abs/2503.14905
