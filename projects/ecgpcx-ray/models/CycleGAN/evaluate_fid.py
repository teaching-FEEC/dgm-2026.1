"""
FID evaluation script for CycleGAN counterfactual generation.

Computes Fréchet Inception Distance (FID) between:
  - Real pneumonia (test) vs. Fake pneumonia generated from healthy (H→P)
  - Real healthy (test)  vs. Fake healthy generated from pneumonia (P→H)

Usage:
    python evaluate_fid.py \
        --checkpoint checkpoints/epoch_199.pt \
        --healthy_dir  ../../data/processed/test/healthy \
        --pneumonia_dir ../../data/processed/test/pneumonia \
        [--batch_size 16] [--image_size 128] [--device cuda]
"""

import argparse
import json
import datetime
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Allow imports from the CycleGAN package when running from any directory
sys.path.insert(0, str(Path(__file__).parent))
from model_utils.cyclegan import CycleGAN


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    """Loads all images from a directory as grayscale PIL images."""

    EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(self, directory: str | Path, transform=None):
        self.paths = sorted(
            p for p in Path(directory).iterdir()
            if p.suffix.lower() in self.EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {directory}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img


# ---------------------------------------------------------------------------
# Inception feature extractor
# ---------------------------------------------------------------------------

class InceptionFeatureExtractor(torch.nn.Module):
    """InceptionV3 truncated at the pool3 (2048-d) layer."""

    def __init__(self, device: torch.device):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.eval()

        # Keep only up to the adaptive avg pool (pool3 = 2048-d embedding)
        self.layers = torch.nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W) in [0, 1] with ImageNet normalisation applied
        return self.layers(x).squeeze(-1).squeeze(-1)  # (N, 2048)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def cyclegan_to_inception(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Convert a CycleGAN output tensor to InceptionV3 input.

    CycleGAN outputs: (N, 1, H, W) normalised to [-1, 1].
    InceptionV3 expects: (N, 3, 299, 299) normalised with ImageNet stats.
    """
    # [-1, 1] → [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0.0, 1.0)

    # (N, 1, H, W) → (N, 3, H, W)  (repeat grayscale across RGB)
    tensor = tensor.repeat(1, 3, 1, 1)

    # Resize to 299x299 (InceptionV3 minimum)
    tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)

    # ImageNet normalisation
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


# ---------------------------------------------------------------------------
# Feature collection
# ---------------------------------------------------------------------------

def collect_features(
    loader: DataLoader,
    extractor: InceptionFeatureExtractor,
    device: torch.device,
    generator: torch.nn.Module | None = None,
    desc: str = "",
) -> np.ndarray:
    """
    Iterate over a DataLoader and collect 2048-d Inception features.

    If `generator` is provided the images are first passed through it
    (counterfactual generation); otherwise the raw images are used.
    """
    all_features = []
    n = len(loader.dataset)
    print(f"  Extracting features — {desc} ({n} images)...")

    for batch in loader:
        batch = batch.to(device)

        if generator is not None:
            with torch.no_grad():
                batch = generator(batch)

        inception_input = cyclegan_to_inception(batch, device)

        features = extractor(inception_input)          # (B, 2048)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)        # (N, 2048)


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_statistics(features: np.ndarray):
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """
    Fréchet distance between two Gaussians.

    FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ − 2·sqrt(Σ₁·Σ₂))
    """
    diff = mu1 - mu2

    # Product of covariances; add small diagonal to avoid singular matrices
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical clean-up: sqrtm can return tiny imaginary parts
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("sqrtm produced large imaginary component")
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID for CycleGAN counterfactual generation on the test set."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the .pt checkpoint file (e.g. checkpoints/epoch_199.pt)."
    )
    parser.add_argument(
        "--healthy_dir", required=True,
        help="Directory containing test healthy X-ray images."
    )
    parser.add_argument(
        "--pneumonia_dir", required=True,
        help="Directory containing test pneumonia X-ray images."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for feature extraction (default: 16)."
    )
    parser.add_argument(
        "--image_size", type=int, default=128,
        help="Spatial size to resize images before the generator (default: 128)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker count (default: 4)."
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use — 'cuda' or 'cpu' (default: cuda if available)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save results as JSON. "
             "Defaults to <checkpoint_stem>_fid.json next to the checkpoint."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load generators from checkpoint
    # ------------------------------------------------------------------
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = CycleGAN(image_channels=1)
    model.G_H2P.load_state_dict(ckpt["G_H2P"])
    model.G_P2H.load_state_dict(ckpt["G_P2H"])

    G_H2P = model.G_H2P.to(device).eval()
    G_P2H = model.G_P2H.to(device).eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')} checkpoint.")

    # ------------------------------------------------------------------
    # Prepare data transforms (no augmentation — deterministic eval)
    # ------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    healthy_ds   = ImageFolderDataset(args.healthy_dir,   transform=transform)
    pneumonia_ds = ImageFolderDataset(args.pneumonia_dir, transform=transform)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    healthy_loader   = DataLoader(healthy_ds,   **loader_kwargs)
    pneumonia_loader = DataLoader(pneumonia_ds, **loader_kwargs)

    print(f"\nTest set — healthy: {len(healthy_ds)} | pneumonia: {len(pneumonia_ds)}")

    # ------------------------------------------------------------------
    # InceptionV3 feature extractor
    # ------------------------------------------------------------------
    print("\nLoading InceptionV3 feature extractor...")
    extractor = InceptionFeatureExtractor(device)

    # ------------------------------------------------------------------
    # Collect features
    # ------------------------------------------------------------------
    print("\n--- Collecting features ---")

    # Real images (no generator)
    feats_real_healthy   = collect_features(healthy_loader,   extractor, device,
                                            desc="real healthy")
    feats_real_pneumonia = collect_features(pneumonia_loader, extractor, device,
                                            desc="real pneumonia")

    # Counterfactuals
    feats_fake_pneumonia = collect_features(healthy_loader,   extractor, device,
                                            generator=G_H2P,
                                            desc="fake pneumonia (H→P via G_H2P)")
    feats_fake_healthy   = collect_features(pneumonia_loader, extractor, device,
                                            generator=G_P2H,
                                            desc="fake healthy   (P→H via G_P2H)")

    # ------------------------------------------------------------------
    # Compute FID scores
    # ------------------------------------------------------------------
    print("\n--- Computing FID ---")

    mu_rP, sig_rP = compute_statistics(feats_real_pneumonia)
    mu_fP, sig_fP = compute_statistics(feats_fake_pneumonia)
    fid_H2P = frechet_distance(mu_rP, sig_rP, mu_fP, sig_fP)

    mu_rH, sig_rH = compute_statistics(feats_real_healthy)
    mu_fH, sig_fH = compute_statistics(feats_fake_healthy)
    fid_P2H = frechet_distance(mu_rH, sig_rH, mu_fH, sig_fH)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    mean_fid = (fid_H2P + fid_P2H) / 2

    print("\n" + "=" * 50)
    print("FID Results")
    print("=" * 50)
    print(f"  H→P  (real pneumonia  vs fake pneumonia) : {fid_H2P:.4f}")
    print(f"  P→H  (real healthy    vs fake healthy)   : {fid_P2H:.4f}")
    print(f"  Mean FID                                 : {mean_fid:.4f}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # Save results to JSON
    # ------------------------------------------------------------------
    output_path = Path(args.output) if args.output else (
        Path(args.checkpoint).with_name(Path(args.checkpoint).stem + "_fid.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "fid_H2P": round(fid_H2P, 6),
        "fid_P2H": round(fid_P2H, 6),
        "mean_fid": round(mean_fid, 6),
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "healthy_dir": str(args.healthy_dir),
        "pneumonia_dir": str(args.pneumonia_dir),
        "n_healthy": len(healthy_ds),
        "n_pneumonia": len(pneumonia_ds),
        "image_size": args.image_size,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return {"fid_H2P": fid_H2P, "fid_P2H": fid_P2H}


if __name__ == "__main__":
    main()
