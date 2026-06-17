#!/bin/bash

# ============================================================
# run_experiments.sh
# Trains STE-GAN for 4 combinations of NeuroRVQ and MTD weights
# Weights: 0, 1, 7, 15
# ============================================================

set -e  # Exit on error

EMG_ENC_CKPT="/workspace/ste-gan/exp/emg_encoder/EMGEncoderTransformer_voiced_only__seq_len__200__data_gaddy_complete/best_val_loss_model.pt"
CONFIG_DIR="/workspace/dgm-2026.1/projects/EMGAN/configs/generated"
BASE_TRAIN_SCRIPT="ste_gan/train.py"

mkdir -p "$CONFIG_DIR"

# Weights to sweep




WEIGHTS_NRVQ=(0, 1, 7, 15)

WEIGHTS_MTD=(0, 1, 7, 15)

for NEURORVQ_W in "${WEIGHTS_NRVQ[@]}"; do
  for MTD_W in "${WEIGHTS_MTD[@]}"; do

    CONFIG_PATH="${CONFIG_DIR}/ste_gan_neurorvq${NEURORVQ_W}_mtd${MTD_W}.yaml"

    echo "=========================================="
    echo "Generating config: NeuroRVQ weight=${NEURORVQ_W}, MTD weight=${MTD_W}"
    echo "=========================================="

    cat > "$CONFIG_PATH" <<YAML
# Auto-generated config — NeuroRVQ weight=${NEURORVQ_W}, MTD weight=${MTD_W}
# Where models should be saved
model_base_dir: exp/ste-gan_mtd${MTD_W}_neurorvq${NEURORVQ_W}

# The main model
model:
  type: "EMGGeneratorGanTTS"
  # The main speech features for speech to EMG
  # Defaults to speech units -> 50Hz
  # Alternative: MFCCs -> 100Hz
  speech_feature_type: "SPEECH_UNITS"
  discriminator_small: true

train:
  random_seed: 0
  debug: false

  # LOSS WEIGHTS
  loss_adversarial: "mse"

  mixed_precision: true

  loss_neuro_rvq_error: true
  loss_neuro_rvq_weight: ${NEURORVQ_W}.0
  neuro_rvq_config_path: "./NeuroRVQ/flags/NeuroRVQ_EMG_v1.yml"
  neuro_rvq_model_path:  "./ste_gan/pretrained_models/pretrained_models/tokenizers/NeuroRVQ_EMG_tokenizer_v1.pt"
  neuro_rvq_ch_names: ["c1","c2","c3","c4","c5","c6","c7","c8"]

  # EMG encoder losses (speech unit + phoneme loss)
  # Speech Unit Loss
  loss_speech_unit_error: true
  loss_speech_unit_weight: 1.0

  # Phoneme classification
  loss_phoneme_error: true
  loss_phoneme_weight: 1.0

  # Multi Time-Domain Loss
  loss_multi_td_error: true
  loss_multi_td_weight: ${MTD_W}.0

  # Feature matching
  loss_feat_match_error: true
  loss_feat_match_weight: 7.0

  # MSE loss
  loss_waveform_error: false
  loss_waveform_weight: 0.0

  # Optimization settings
  batch_size: 64

  # Number of EMG samples to output
  chunk_size: 2048

  # max steps
  max_steps: 25_000

  # Logging & Eval settings
  interval_log: 50
  interval_sample: 1_000
  interval_save: 10_000
  interval_valid: 500
  interval_waveform: 500
  interval_plot: 1_000
  num_test_samples: 10
YAML

    echo "Config written to: $CONFIG_PATH"
    echo ""
    echo "Launching training: NeuroRVQ=${NEURORVQ_W}, MTD=${MTD_W}"
    echo "------------------------------------------"

    python "$BASE_TRAIN_SCRIPT" \
      --emg_enc_ckpt "$EMG_ENC_CKPT" \
      --config "$CONFIG_PATH"

    echo ""
    echo "Finished run: NeuroRVQ=${NEURORVQ_W}, MTD=${MTD_W}"
    echo ""

  done
done

echo "=========================================="
echo "All experiments completed."
echo "=========================================="
