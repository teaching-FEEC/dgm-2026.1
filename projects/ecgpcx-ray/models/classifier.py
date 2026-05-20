"""Classes for baselines classifier for binary chest X-ray classification (Pneumonia vs Healthy)."""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d → BatchNorm2d → ReLU → MaxPool2d(2)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    """4-block CNN for binary chest X-ray classification.

    Input:  (B, 1, H, W) grayscale image, normalized to [0, 1]
    Output: (B, 1) raw logit for the pneumonia class
    """

    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),  # → (B, 32,  H/2,  W/2)
            ConvBlock(32, 64),  # → (B, 64,  H/4,  W/4)
            ConvBlock(64, 128),  # → (B, 128, H/8,  W/8)
            ConvBlock(128, 256), # → (B, 256, H/16, W/16)
            ConvBlock(256, 512), # → (B, 512, H/32, W/32)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class FrozenResNet18(nn.Module):
    """ResNet-18 com backbone ImageNet congelado e head binário.

    Aceita imagens grayscale (B, 1, H, W): repete o canal 3x internamente
    e aplica a normalização ImageNet antes de passar pelo backbone.
    Somente a camada fc é treinável.
    """

    # from ImageNet (used to normalize inputs)
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze the entire backbone
        for param in backbone.parameters():
            param.requires_grad = False

        # Replace the final classifier with a binary output
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)

        self.model = backbone

        mean = torch.tensor(self._MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self._STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # (B,1,H,W) → (B,3,H,W)
        x = (x - self.mean) / self.std  # ImageNet normalization
        return self.model(x)  # (B, 1)


class FrozenDenseNet121(nn.Module):
    """DenseNet-121 com backbone ImageNet e head binário.

    Arquitetura padrão para classificação de chest X-rays (CheXNet,
    Rajpurkar et al. 2017). As "dense connections" preservam features
    de baixo nível ao longo das camadas, úteis para detectar padrões
    sutis como consolidações e infiltrados.

    Aceita imagens grayscale (B, 1, H, W); repete o canal 3x e aplica
    normalização ImageNet antes do backbone.

    Args:
        unfreeze_from: controla quais blocos ficam treináveis.
            - None (default): só a head `classifier` (comportamento original).
            - "denseblock4" / "denseblock3" / "denseblock2" / "denseblock1":
              descongela esse bloco e tudo que vem depois (até a head).
              "denseblock4" é o típico fine-tuning leve do CheXNet.
            - "all": fine-tuning completo (todos os parâmetros treináveis).
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    # Ordem dos blocos em features._modules — usada para decidir o ponto de corte.
    _FEATURE_BLOCKS = [
        "conv0", "norm0", "relu0", "pool0",
        "denseblock1", "transition1",
        "denseblock2", "transition2",
        "denseblock3", "transition3",
        "denseblock4", "norm5",
    ]

    def __init__(self, unfreeze_from=None):
        super().__init__()
        backbone = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1)
        backbone.classifier = nn.Linear(backbone.classifier.in_features, 1)  # 1024 → 1
        self.model = backbone
        self.unfreeze_from = unfreeze_from
        self._apply_freeze(unfreeze_from)

        mean = torch.tensor(self._MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self._STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _apply_freeze(self, unfreeze_from):
        # Default: tudo congelado, apenas a head treinável.
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.classifier.parameters():
            p.requires_grad = True

        if unfreeze_from is None:
            return
        if unfreeze_from == "all":
            for p in self.model.parameters():
                p.requires_grad = True
            return
        if unfreeze_from not in self._FEATURE_BLOCKS:
            raise ValueError(
                f"unfreeze_from={unfreeze_from!r} inválido. "
                f"Use None, 'all', ou um de {self._FEATURE_BLOCKS}."
            )
        cut = self._FEATURE_BLOCKS.index(unfreeze_from)
        for name in self._FEATURE_BLOCKS[cut:]:
            for p in getattr(self.model.features, name).parameters():
                p.requires_grad = True

    def head_parameters(self):
        return self.model.classifier.parameters()

    def backbone_parameters(self):
        # Tudo que NÃO é a head e está treinável (depende do unfreeze_from).
        head_ids = {id(p) for p in self.model.classifier.parameters()}
        return (
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in head_ids
        )

    def param_groups(self, head_lr, backbone_lr=None):
        """Grupos para `torch.optim.Adam(model.param_groups(...))`.

        Use `backbone_lr=None` (default) para treinar só a head (apenas um grupo).
        Com `unfreeze_from != None`, passe um `backbone_lr` menor que `head_lr`
        (típico: head 1e-3, backbone 1e-5).
        """
        groups = [{"params": list(self.head_parameters()), "lr": head_lr}]
        if backbone_lr is not None:
            backbone = list(self.backbone_parameters())
            if backbone:
                groups.append({"params": backbone, "lr": backbone_lr})
        return groups

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return self.model(x)  # (B, 1)
