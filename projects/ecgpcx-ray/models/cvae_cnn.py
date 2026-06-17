import numpy as np
import torch
import torch.nn as nn


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
        self.img_channels = img_channels
        self.img_size = img_size

        cond_dim = num_classes + (metadata_dim - 1) + gender_embedding_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc21 = nn.Linear(256 * (img_size // 16) * (img_size // 16) + cond_dim, latent_dim)
        self.fc22 = nn.Linear(256 * (img_size // 16) * (img_size // 16) + cond_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + cond_dim, 256 * (img_size // 16) * (img_size // 16))

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, img_size // 16, img_size // 16)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),  
            nn.Sigmoid()
        )

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
        x = x.view(x.size(0), self.img_channels, self.img_size, self.img_size)
        h1 = self.encoder(x)

        inputs = [h1, y]
        if m is not None:
            m = self._prepare_metadata(m)
            inputs.append(m)

        inputs = torch.cat(inputs, dim=1)
        # h1 = self.elu(self.fc1(inputs))
        mu = self.fc21(inputs)
        logvar = self.fc22(inputs)
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

        # h3 = self.elu(self.fc3(inputs))
        h2 = self.decoder_input(inputs)
        x_hat = self.decoder(h2)
        # x_hat = self.sigmoid(self.fc4(h3))
        return x_hat

    def forward(self, x, y, m=None):
        mu, logvar = self.encode(x, y, m)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y, m)
        return x_hat, mu, logvar
