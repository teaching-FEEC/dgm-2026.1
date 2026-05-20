"""Utilities for training models."""

import numpy as np
import torch
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import time
import json
from sklearn.metrics import roc_auc_score


def run_epoch(loader, model, criterion, optimizer=None, device="cpu"):
    """
    Run one epoch. If optimizer is None, runs in eval mode.

    Args:
        loader: DataLoader for the epoch.
        model: The model to train/evaluate.
        criterion: Loss function.
        optimizer: Optimizer for training. If None, runs in eval mode.
        device: Device to run on.
    Returns:
        avg_loss: Average loss over the epoch.
        all_labels: Ground truth labels for the epoch.
        all_scores: Predicted scores (after sigmoid) for the epoch.

    """
    training = optimizer is not None
    model = model.to(device)
    model.train() if training else model.eval()

    total_loss = 0.0
    all_labels, all_scores = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels, _ in loader:
            images = images.to(device)  # (B, 1, H, W)
            y = labels[:, 1].unsqueeze(1).float().to(device)  # pneumonia score

            logits = model(images)  # (B, 1)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_labels.extend(y.cpu().squeeze(1).tolist())
            all_scores.extend(torch.sigmoid(logits).cpu().squeeze(1).tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_scores)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    device="cpu",
    num_epochs=20,
    CHECKPOINT_PATH=None,
    early_stopping_patience=10,
):
    """
    Train the model and evaluate on validation set after each epoch.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler (optional).
        device: Device to run on.
        num_epochs: Number of epochs to train.
        CHECKPOINT_PATH: Path to save the best model checkpoint.
        early_stopping_patience: Number of epochs to wait for improvement before stopping early.
    Returns:
        best_val_auc: Best validation AUC achieved during training.
        history: Dictionary containing training and validation loss and AUC history.
    """
    best_val_auc = 0.0
    history = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}
    epochs_without_improvement = 0

    pbar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")
    for epoch in pbar:
        start_time = time.time()
        train_loss, train_y, train_sc = run_epoch(
            train_loader, model, criterion, optimizer, device=device
        )
        val_loss, val_y, val_sc = run_epoch(val_loader, model, criterion, device=device)

        # Compute AUC
        train_auc = roc_auc_score(train_y, train_sc)
        val_auc = roc_auc_score(val_y, val_sc)

        # Check if scheduler is provided and step with validation AUC
        if scheduler is not None:
            scheduler.step(val_auc)

        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            flag = " ← best"
            epochs_without_improvement = 0
            if CHECKPOINT_PATH is not None:
                torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1
            flag = ""

        elapsed_time = time.time() - start_time
        print(
            f"Epoch {epoch:02d}/{num_epochs} "
            f"| elapsed_time={elapsed_time:.2f}s"
            f"| train_loss={train_loss:.4f} "
            f"| train_auc={train_auc:.4f} "
            f"| val_loss={val_loss:.4f} "
            f"| val_auc={val_auc:.4f}{flag} "
        )
        
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"\nBest val AUC: {best_val_auc:.4f}")
    return best_val_auc, history


def training_curves(history, results_path=None):
    """Plot training and validation loss and AUC curves."""

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_auc"], label="Train AUC")
    plt.plot(epochs, history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Training and Validation AUC")
    plt.legend()

    plt.tight_layout()
    plt.show()

    if results_path is not None:
        plt.savefig(results_path)


def save_results(
    model="",
    img_size=(224, 224),
    epochs=20,
    batch_size=32,
    lr=1e-3,
    dropout=0.5,
    seed=42,
    pos_weight=1.0,
    augmentation=None,
    best_val_auc=0.0,
    test_accuracy=0.0,
    test_auc_roc=0.0,
    history=None,
    results_path=None,
):
    """Save training history to a .npz file."""

    results = {
        "model": model,
        "img_size": list(img_size),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "dropout": dropout,
        "seed": seed,
        "pos_weight": round(pos_weight.item(), 4) if isinstance(pos_weight, torch.Tensor) else round(float(pos_weight), 4),
        "augmentation": augmentation,
        "best_val_auc": round(best_val_auc, 4),
        "test_accuracy": round(test_accuracy, 4),
        "test_auc_roc": round(test_auc_roc, 4),
        "history": history,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")
    print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=2))
