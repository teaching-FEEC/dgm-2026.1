import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pathlib import Path
import os
import sys
from tqdm import tqdm

# Add parent directory to path so utils can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import preprocessing

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed at module level
set_seed(42)

class CVAE(nn.Module):
    def __init__(self, img_channels=1, img_size=128, latent_dim=64, num_classes=2, metadata_dim=2, gender_embedding_dim=4):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.metadata_dim = metadata_dim
        self.gender_embedding_dim = gender_embedding_dim

        input_dim = img_channels * img_size * img_size
        cond_dim = num_classes + (metadata_dim - 1) + gender_embedding_dim

        #Encoder
        self.fc1 = nn.Linear(input_dim + cond_dim, 1024)
        self.fc21 = nn.Linear(1024, latent_dim)
        self.fc22 = nn.Linear(1024, latent_dim)

        #Decoder
        self.fc3 = nn.Linear(latent_dim + cond_dim, 1024)
        self.fc4 = nn.Linear(1024, input_dim)

        self.gender_embedding = nn.Embedding(2, gender_embedding_dim)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def _prepare_metadata(self, m):
        if m is None:
            return None
        age = m[:, [0]]
        gender_idx = m[:, 1].long().clamp(0, self.gender_embedding.num_embeddings - 1)
        gender_emb = self.gender_embedding(gender_idx)
        return torch.cat([age, gender_emb], dim=1)

    def encode(self, x, y, m=None):
        inputs = torch.cat([x, y], dim=1)
        if m is not None:
            m = self._prepare_metadata(m)
            inputs = torch.cat([inputs, m], dim=1)
        h1 = self.elu(self.fc1(inputs))
        mu = self.fc21(h1)  
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y, m=None):
        cond = [y]
        if m is not None:
            m = self._prepare_metadata(m)
            cond.append(m)

        cond = torch.cat(cond, dim=1)
        inputs = torch.cat([z, cond], dim=1)

        h3 = self.elu(self.fc3(inputs))
        x_hat = self.sigmoid(self.fc4(h3))
        return x_hat

    def forward(self, x, y, m=None):
        x_flat = x.view(x.size(0), -1)
        mu, logvar = self.encode(x_flat, y, m)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y, m)
        return x_hat, mu, logvar


def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    x_flat = x.view(x.size(0), -1)
    x_hat_flat = x_hat.view(x.size(0), -1)
    reconstruction_loss = 0.5 * F.mse_loss(x_hat_flat, x_flat, reduction='mean') \
                    + 0.5 * F.l1_loss(x_hat_flat, x_flat, reduction='mean')
    # Normalize KLD by the batch size so it is on the same scale as the reconstruction loss.
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / x_flat.size(0)
    weighted_kld = beta * KLD
    return reconstruction_loss + weighted_kld, reconstruction_loss, KLD, weighted_kld, beta


def evaluate(model, evaluate_loader, device, epoch, beta=1.0, results_dir="../training-results/cvae/results"):
    model.eval()
    total_loss = 0
    total_rec_loss = 0
    total_kld = 0
    total_weighted_kld = 0
    with torch.no_grad():
        for i, batch in enumerate(evaluate_loader):
            x, y, m = batch
            x = x.to(device)
            y = y.to(device)
            m = m.to(device) if m is not None else None
            
            x_hat, mu, logvar = model(x, y, m)
            loss, rec_loss, kld, weighted_kld, beta = vae_loss(x, x_hat, mu, logvar, beta=beta)
            total_loss += loss.item() * x.size(0)
            total_rec_loss += rec_loss.item() * x.size(0)
            total_kld += kld.item() * x.size(0)
            total_weighted_kld += weighted_kld.item() * x.size(0)

            if i == 0:
                n = min(x.size(0), 8)
                comparison = torch.cat([x[:n], x_hat[:n].view(-1, 1, 128, 128)], dim=0)
                save_image(comparison.cpu(), f'{results_dir}/reconstruction_{epoch}.png', nrow=n)

    avg_loss = total_loss / len(evaluate_loader.dataset)
    avg_rec_loss = total_rec_loss / len(evaluate_loader.dataset)
    avg_kld = total_kld / len(evaluate_loader.dataset)
    avg_weighted_kld = total_weighted_kld / len(evaluate_loader.dataset)
    tqdm.write('Validation set loss (epoch {:03d}): total={:.3f}, rec={:.3f}, kld={:.3f}, weighted_kld={:.3f}'.format(epoch, avg_loss, avg_rec_loss, avg_kld, avg_weighted_kld))
    return avg_loss


def train(model, train_loader, val_loader, optimizer, device, epoch, beta=1.0):
    """Train the CVAE model for a single epoch."""
    model.train()
    total_loss = 0
    total_rec_loss = 0
    total_kld = 0
    total_weighted_kld = 0

    # Initialize tqdm (shows the progress during training)
    progress_bar = tqdm(train_loader, desc='Epoch {:03d}'.format(epoch), leave=False, disable=False)

    for x, y, m in progress_bar:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device) if m is not None else None
        
        # Forward pass
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x, y, m)
        
        # Compute loss
        loss, rec_loss, kld, weighted_kld, beta = vae_loss(x, x_hat, mu, logvar, beta=beta)
        
        # Backward pass
        loss.backward()
        
        # Track loss
        total_loss += loss.item() * x.size(0)
        total_rec_loss += rec_loss.item() * x.size(0)
        total_kld += kld.item() * x.size(0)
        total_weighted_kld += weighted_kld.item() * x.size(0)
        
        #Update model parameters
        optimizer.step()
        
        # Update the progress bar with the current batch's loss
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()), 'beta': '{:.3f}'.format(beta)})
    
    avg_loss = total_loss / len(train_loader.dataset)
    avg_rec_loss = total_rec_loss / len(train_loader.dataset)
    avg_kld = total_kld / len(train_loader.dataset)
    avg_weighted_kld = total_weighted_kld / len(train_loader.dataset)
    tqdm.write('Training set loss (epoch {:03d}): total={:.3f}, rec={:.3f}, kld={:.3f}, weighted_kld={:.3f}'.format(epoch, avg_loss, avg_rec_loss, avg_kld, avg_weighted_kld))

    #Evaluate on validation set after each epoch
    val_loss = evaluate(model, val_loader, device, epoch, beta=beta)
    return avg_loss, val_loss


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir="../training-results/cvae/models"):
    """Save model checkpoint."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}: {checkpoint_path}")
    return epoch

def get_latest_checkpoint(checkpoint_dir="../training-results/cvae/models"):
    """Get the latest checkpoint path."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
    if not checkpoints:
        return None
    
    checkpoints.sort()
    latest = checkpoints[-1]
    return os.path.join(checkpoint_dir, latest)

def beta_schedule(epoch, total_epochs, max_beta=1.0):
    # Ramp beta from a small positive value up to max_beta over the first half of training.
    return max_beta * min(1.0, (epoch + 1) / (total_epochs * 0.5))

def train_loop(model, train_loader, val_loader, optimizer, device, epochs=10, beta=1.0, checkpoint_dir="../training-results/cvae/models", resume=True):
    """Train the CVAE model for multiple epochs."""
    model.to(device)
    train_losses, val_losses = [], []
    start_epoch = 0
    
    # Load latest checkpoint if resume is True
    if resume:
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            start_epoch = load_checkpoint(model, optimizer, latest_checkpoint) + 1
    
    for epoch in range(start_epoch, epochs):
        # beta = beta_schedule(epoch, epochs, max_beta=beta)
        avg_loss, val_loss = train(model, train_loader, val_loader, optimizer, device, epoch, beta=beta)
        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        # current_lr = optimizer.param_groups[0]['lr']
        # tqdm.write(f'Learning rate after epoch {epoch}: {current_lr:.6f}')
        
        # Save checkpoint after each epoch
        if epoch % 10 == 0 or epoch == epochs - 1:  # Save every 10 epochs and the last epoch
            save_checkpoint(model, optimizer, epoch, avg_loss, val_loss, checkpoint_dir)
    
    # Save the recorded training losses to a txt file
    np.savetxt(f'{checkpoint_dir}/training_losses.txt', np.array(train_losses), delimiter='\n')
    # Save the recorded validation losses to a txt file
    np.savetxt(f'{checkpoint_dir}/validation_losses.txt', np.array(val_losses), delimiter='\n')
    
    return train_losses, val_losses
    

# def generate_counterfactual(model, x, y_source, y_target, m=None):
#     """Generate counterfactual examples by optimizing in the latent space."""
#     model.eval()
#     with torch.no_grad():
#         x_flat = x.view(x.size(0), -1)
#         mu, logvar = model.encode(x_flat, y_source, m)
#         z = model.reparameterize(mu, logvar)

#         # Decode with target label
#         x_cf = model.decode(z, y_target, m)
#         return x_cf.view(x.size())  # Reshape to original image dimensions

def generate_counterfactual(model, x, y_source, y_target, m=None,
                            num_steps=50, lr=1e-2, lambda_sim=10.0, lambda_z=0.1):
    model.eval()

    with torch.no_grad():
        x_flat = x.view(x.size(0), -1)
        mu, logvar = model.encode(x_flat, y_source, m)
        z_init = model.reparameterize(mu, logvar)

    z = z_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    # Flatten target image to match decoder output shape ([B, input_dim])
    x_target = x.view(x.size(0), -1)

    for _ in range(num_steps):
        x_cf = model.decode(z, y_target, m)
        # Use per-sample mean losses for stable gradients during latent optimization
        recon_loss = F.mse_loss(x_cf, x_target, reduction='mean')
        z_reg = F.mse_loss(z, mu, reduction='mean')
        # Keeps the counterfactual close to the original image in latent space, preventing unrealistic changes
        loss = lambda_sim * recon_loss + lambda_z * z_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # After optimization, decode the final counterfactual
    x_cf = model.decode(z, y_target, m)
    return x_cf.view(x.size())

def save_counterfactuals_individual(x_original, x_counterfactual, start_idx=0, 
                                    original_dir="../training-results/cvae/results/original/",
                                    counterfactual_dir="../training-results/cvae/results/counterfactuals/"):
    """Save original and counterfactual images individually and as pairs.
    
    Args:
        x_original: Original images (batch_size, 1, 128, 128)
        x_counterfactual: Counterfactual images (batch_size, 1, 128, 128)
        start_idx: Starting index for image naming
        original_dir: Directory to save original images
        counterfactual_dir: Directory to save counterfactual images
    """
    Path(original_dir).mkdir(parents=True, exist_ok=True)
    Path(counterfactual_dir).mkdir(parents=True, exist_ok=True)
    
    # Ensure tensors are in image format (batch_size, 1, 128, 128)
    if x_original.dim() == 2:
        x_original = x_original.view(-1, 1, 128, 128)
    if x_counterfactual.dim() == 2:
        x_counterfactual = x_counterfactual.view(-1, 1, 128, 128)
    
    batch_size = x_original.size(0)
    
    # Save each image individually
    for i in range(batch_size):
        img_idx = start_idx + i
        
        # Save original image
        orig_img = x_original[i:i+1]
        orig_path = os.path.join(original_dir, f"img_{img_idx:06d}_original.png")
        save_image(orig_img, orig_path)
        
        # Save counterfactual image
        cf_img = x_counterfactual[i:i+1]
        cf_path = os.path.join(counterfactual_dir, f"img_{img_idx:06d}_counterfactual.png")
        save_image(cf_img, cf_path)
        
        # Save pair (original and counterfactual side by side)
        pair = torch.cat([orig_img, cf_img], dim=0)
        pair_path = os.path.join(counterfactual_dir, f"img_{img_idx:06d}_pair.png")
        save_image(pair, pair_path, nrow=2)
    
    print(f"Saved {batch_size} original images to {original_dir}")
    print(f"Saved {batch_size} counterfactual images and pairs to {counterfactual_dir}")

def test(test_loader, model, device):
    image_idx = 0
    for batch_idx, (x, y, m) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        # Generate counterfactuals (flip class: 0->1, 1->0)
        y_target = 1 - y  # Flip labels: healthy (0) -> pneumonia (1), or vice versa

        x_cf = generate_counterfactual(model, x, y, y_target, m)

        # Save original and counterfactual images individually and as pairs
        save_counterfactuals_individual(
            x.cpu(), 
            x_cf.cpu(), 
            start_idx=image_idx,
            original_dir="../training-results/cvae/results/original/",
            counterfactual_dir="../training-results/cvae/results/counterfactuals/"
        )
        
        image_idx += x.size(0)
        print(f"Batch {batch_idx + 1} completed. Total images processed: {image_idx}")
    
    print(f"\nCounterfactual generation complete! Processed {image_idx} test images.")

# =========================
# 🔹 EXAMPLE USAGE
# =========================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    preprocessing = preprocessing.Preprocessing(label="Pneumonia")

    model = CVAE(
        img_channels=1,
        img_size=128,
        latent_dim=64,
        num_classes=2,
        metadata_dim=2  # age + gender
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create CVAE-compatible dataset
    train_dataset, test_dataset, val_dataset = preprocessing.create_cvae_dataset(
        img_size=(128, 128),
        verbose=True, 
        method='standard'
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    train_loop(model, train_loader, val_loader, optimizer, device, epochs=200, beta=0.01)

    # Generate counterfactuals for all test samples
    print("\nGenerating counterfactuals for all test samples...")
    
    test(test_loader, model, device)