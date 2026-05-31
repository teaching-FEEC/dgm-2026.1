DEVICE = "cuda"

IMAGE_SIZE = 128

BATCH_SIZE = 4

NUM_WORKERS = 4

EPOCHS = 200

LR = 2e-4

LAMBDA_CYCLE = 10.0

LAMBDA_IDENTITY = 5.0

CHECKPOINT_DIR = "checkpoints"

TRAIN_HEALTHY_DIR = (
    "/home/gfreitas/ia376n/final_project/dgm-2026.1/projects/ecgpcx-ray/data/processed/train/healthy"
)

TRAIN_PNEUMONIA_DIR = (
    "/home/gfreitas/ia376n/final_project/dgm-2026.1/projects/ecgpcx-ray/data/processed/train/pneumonia"
)

VAL_HEALTHY_DIR = (
    "/home/gfreitas/ia376n/final_project/dgm-2026.1/projects/ecgpcx-ray/data/processed/val/healthy"
)

VAL_PNEUMONIA_DIR = (
    "/home/gfreitas/ia376n/final_project/dgm-2026.1/projects/ecgpcx-ray/data/processed/val/pneumonia"
)