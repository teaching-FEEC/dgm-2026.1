#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== GANDCTAnalysis (3 models) ==="
cd "$SCRIPT_DIR/GANDCTAnalysis"
source .venv/bin/activate
python classify.py --model-type ridge_pixel --split train
python classify.py --model-type ridge_dct --split train
python classify.py --model-type lasso_dct --split train
deactivate

echo "=== FakeImageDetection (12 models) ==="
cd "$SCRIPT_DIR/FakeImageDetection"
source .venv/bin/activate
MODELS=../../../../models/FakeImageDetection
python classify.py --checkpoint $MODELS/mask_0/rn50ft.pth --model-name rn50 --split train
python classify.py --checkpoint $MODELS/mask_0/rn50_modft.pth --model-name rn50_mod --split train
python classify.py --checkpoint $MODELS/mask_15/clipft.pth --model-name clip --split train
python classify.py --checkpoint $MODELS/mask_15/clipft_spectralmask.pth --model-name clip --split train
python classify.py --checkpoint $MODELS/mask_15/rn50ft_highspectralmask.pth --model-name rn50 --split train
python classify.py --checkpoint $MODELS/mask_15/rn50ft_lowspectralmask.pth --model-name rn50 --split train
python classify.py --checkpoint $MODELS/mask_15/rn50ft_midspectralmask.pth --model-name rn50 --split train
python classify.py --checkpoint $MODELS/mask_15/rn50ft_patchmask.pth --model-name rn50 --split train
python classify.py --checkpoint $MODELS/mask_15/rn50ft_pixelmask.pth --model-name rn50 --split train
python classify.py --checkpoint $MODELS/mask_15/rn50ft_spectralmask.pth --model-name rn50 --split train
python classify.py --checkpoint "$MODELS/mask_15/rn50ft_spectralmask(0.5).pth" --model-name rn50 --split train
python classify.py --checkpoint $MODELS/mask_15/rn50_modft_spectralmask.pth --model-name rn50_mod --split train
deactivate

echo "=== SPAI (1 model) ==="
cd "$SCRIPT_DIR/SPAI"
source .venv/bin/activate
python classify.py --checkpoint ../../../../models/SPAI/spai.pth --split train
deactivate

echo "=== NPR (1 model) ==="
cd "$SCRIPT_DIR/NPR"
source .venv/bin/activate
python classify.py --checkpoint ../../../../models/NPR/NPR.pth --split train
deactivate

echo "=== Augmenting train labels ==="
cd "$SCRIPT_DIR"
python3 augment_labels.py --split train

echo "=== Done ==="
