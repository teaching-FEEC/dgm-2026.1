#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== GANDCTAnalysis (3 models) ==="
cd "$SCRIPT_DIR/GANDCTAnalysis"
source .venv/bin/activate
python classify.py --model-type ridge_pixel --split test
python classify.py --model-type ridge_dct --split test
python classify.py --model-type lasso_dct --split test
deactivate

echo "=== FakeImageDetection (12 models) ==="
cd "$SCRIPT_DIR/FakeImageDetection"
source .venv/bin/activate
MODELS=../../../../models/FakeImageDetection
python classify.py --checkpoint $MODELS/mask_0/rn50ft.pth --model-name rn50 --split test
python classify.py --checkpoint $MODELS/mask_0/rn50_modft.pth --model-name rn50_mod --split test
python classify.py --checkpoint $MODELS/mask_15/clipft.pth --model-name clip --split test
python classify.py --checkpoint $MODELS/mask_15/clipft_spectralmask.pth --model-name clip --split test
python classify.py --checkpoint $MODELS/mask_15/rn50ft_highspectralmask.pth --model-name rn50 --split test
python classify.py --checkpoint $MODELS/mask_15/rn50ft_lowspectralmask.pth --model-name rn50 --split test
python classify.py --checkpoint $MODELS/mask_15/rn50ft_midspectralmask.pth --model-name rn50 --split test
python classify.py --checkpoint $MODELS/mask_15/rn50ft_patchmask.pth --model-name rn50 --split test
python classify.py --checkpoint $MODELS/mask_15/rn50ft_pixelmask.pth --model-name rn50 --split test
python classify.py --checkpoint $MODELS/mask_15/rn50ft_spectralmask.pth --model-name rn50 --split test
python classify.py --checkpoint "$MODELS/mask_15/rn50ft_spectralmask(0.5).pth" --model-name rn50 --split test
python classify.py --checkpoint $MODELS/mask_15/rn50_modft_spectralmask.pth --model-name rn50_mod --split test
deactivate

echo "=== SPAI (1 model) ==="
cd "$SCRIPT_DIR/SPAI"
source .venv/bin/activate
python classify.py --checkpoint ../../../../models/SPAI/spai.pth --split test
deactivate

echo "=== NPR (1 model) ==="
cd "$SCRIPT_DIR/NPR"
source .venv/bin/activate
python classify.py --checkpoint ../../../../models/NPR/NPR.pth --split test
deactivate

echo "=== Augmenting test labels ==="
cd "$SCRIPT_DIR"
python3 augment_labels.py --split test

echo "=== Done ==="
