import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


REQUIRED_FILES = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "freq_projector.pt",
]

COPY_FILES = [
    "adapter_model.safetensors",
    "freq_projector.pt",
    "tokenizer.json",
    "tokenizer_config.json",
]

CODE_FILES = [
    "__init__.py",
    "model.py",
    "frequency_projector.py",
    "loader.py",
    "extractors/__init__.py",
    "extractors/base.py",
    "extractors/fft.py",
]

MODEL_CARD = """\
---
library_name: peft
base_model: lingcco/fakeVLM
tags:
  - deepfake-detection
  - llava
  - lora
  - frequency-domain
---

# FakeVLM-Extended

FakeVLM augmented with frequency-domain features for improved deepfake
detection and explainability.

## Architecture

Extends [FakeVLM](https://huggingface.co/lingcco/fakeVLM) (LLaVA 1.5 7B
fine-tuned on [FakeClue](https://github.com/keithhans/FakeClue)) with a
parallel frequency-domain feature branch:

```
Image → CLIP-ViT → 576 × 4096 ──┐
                                  ├─ concat → 577 × 4096 → Vicuna 7B
Image → FFT → 3072 ─────────────┘
              ↓
        FreqProjector → 1 × 4096
```

A log-magnitude FFT spectrum is extracted from each image and projected into
the LLM embedding space via a 2-layer MLP (~22M params). The resulting token
is concatenated with the 576 CLIP visual tokens before being passed to
Vicuna 7B.

## Weights

- `adapter_model.safetensors` — LoRA adapter (r=8, α=16) for Vicuna 7B
- `freq_projector.pt` — Frequency projector MLP weights

## Usage

```python
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel

# Clone or download the FakeVLM_extended code from this repo
from FakeVLM_extended.extractors import get_extractor
from FakeVLM_extended.frequency_projector import FrequencyProjector
from FakeVLM_extended.model import extend_model

# Load base FakeVLM model
model = LlavaForConditionalGeneration.from_pretrained(
    "lingcco/fakeVLM", torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained("lingcco/fakeVLM")
config = model.config

# Create and load frequency components
freq_extractor = get_extractor("fft", input_size=224, pool_size=32)
freq_projector = FrequencyProjector(
    input_dim=freq_extractor.output_dim,
    output_dim=config.text_config.hidden_size,
    num_tokens=1,
)
freq_projector.load_state_dict(
    torch.load("freq_projector.pt", map_location="cpu")
)

# Extend model with frequency branch
model = extend_model(model, freq_extractor, freq_projector, num_freq_tokens=1)

# Load and merge LoRA adapter
model = PeftModel.from_pretrained(model, "<this-repo-id>")
model = model.merge_and_unload()
model.eval()

# Inference
image = Image.open("test_image.jpg").convert("RGB")
prompt = "USER: <image>\\nIs this image real or fake? Explain.\\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Expand image tokens for frequency token (576 → 577)
image_token_id = config.image_token_index
input_ids = inputs["input_ids"]
mask = input_ids[0] == image_token_id
last_img_pos = mask.nonzero()[-1].item()
extra = torch.full(
    (1, 1), image_token_id,
    dtype=input_ids.dtype, device=input_ids.device,
)
inputs["input_ids"] = torch.cat(
    [input_ids[:, :last_img_pos + 1], extra, input_ids[:, last_img_pos + 1:]],
    dim=1,
)

freq_pixel_values = freq_extractor.preprocess([image]).to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        freq_pixel_values=freq_pixel_values,
        max_new_tokens=256,
    )

print(processor.decode(output[0], skip_special_tokens=True))
```

## Training

Two-stage training on FakeClue (104K images, 7 categories):

**Stage 1** — Frequency projector only (~22M params), 1 epoch.
lr=1e-3, batch=8, grad_accum=2, cosine schedule, bf16.

**Stage 2** — LoRA (r=8, α=16) on all Vicuna linear layers + frequency
projector, 3 epochs. lr=2e-5, batch=4, grad_accum=4, cosine schedule, bf16.

Training data used frequency-augmented labels: for fake images correctly
identified by frequency-domain classifiers, the sentence "The image also
presents artifacts in the frequency domain." was appended to the ground-truth
response.

## Base Model

[FakeVLM](https://huggingface.co/lingcco/fakeVLM) — LLaVA 1.5 7B fine-tuned
on FakeClue for deepfake detection and explainability.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Push FakeVLM-Extended LoRA adapter to HuggingFace Hub"
    )
    parser.add_argument("--stage2-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    stage2 = Path(args.stage2_dir)
    for name in REQUIRED_FILES:
        if not (stage2 / name).exists():
            raise FileNotFoundError(f"Missing {name} in {stage2}")

    code_dir = Path(__file__).parent

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Copy model artifacts
        for name in COPY_FILES:
            src = stage2 / name
            if src.exists():
                shutil.copy2(src, tmp / name)

        # Fix adapter_config.json base_model path
        with open(stage2 / "adapter_config.json") as f:
            adapter_config = json.load(f)
        adapter_config["base_model_name_or_path"] = "lingcco/fakeVLM"
        with open(tmp / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)

        # Write model card
        readme = MODEL_CARD.replace("<this-repo-id>", args.repo_id)
        with open(tmp / "README.md", "w") as f:
            f.write(readme)

        # Copy inference code
        dest_code = tmp / "FakeVLM_extended"
        dest_extractors = dest_code / "extractors"
        dest_extractors.mkdir(parents=True)
        for rel in CODE_FILES:
            shutil.copy2(code_dir / rel, dest_code / rel)

        # Upload
        api = HfApi()
        api.create_repo(args.repo_id, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(tmp), repo_id=args.repo_id)

    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
