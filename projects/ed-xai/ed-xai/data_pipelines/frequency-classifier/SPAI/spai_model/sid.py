# From: https://github.com/mever-team/spai
# SPDX-License-Identifier: Apache-2.0
# Trimmed for inference only (removed ONNX export, CLIP/DINOv2 branches,
# unused feature processors).

from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import five_crop
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from einops import rearrange

from . import vision_transformer
from . import filters
from . import model_utils


class PatchBasedMFViT(nn.Module):
    def __init__(
        self,
        vit: vision_transformer.VisionTransformer,
        features_processor: 'FrequencyRestorationEstimator',
        cls_head: Optional[nn.Module],
        masking_radius: int,
        img_patch_size: int,
        img_patch_stride: int,
        cls_vector_dim: int,
        num_heads: int,
        attn_embed_dim: int,
        dropout: float = .0,
        frozen_backbone: bool = True,
        minimum_patches: int = 0,
        initialization_scope: str = "all"
    ) -> None:
        super().__init__()

        self.mfvit = MFViT(
            vit,
            features_processor,
            None,
            masking_radius,
            img_patch_size,
            frozen_backbone=frozen_backbone,
            initialization_scope=initialization_scope
        )

        self.img_patch_size = img_patch_size
        self.img_patch_stride = img_patch_stride
        self.minimum_patches = minimum_patches
        self.cls_vector_dim = cls_vector_dim

        dim_head = attn_embed_dim // num_heads
        self.heads = num_heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_kv = nn.Linear(cls_vector_dim, attn_embed_dim * 2, bias=False)
        self.patch_aggregator = nn.Parameter(
            torch.zeros((num_heads, 1, attn_embed_dim // num_heads)))
        nn.init.trunc_normal_(self.patch_aggregator, std=.02)
        self.to_out = nn.Sequential(
            nn.Linear(attn_embed_dim, cls_vector_dim, bias=False),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(cls_vector_dim)
        self.cls_head = cls_head

        if initialization_scope == "all":
            self.apply(_init_weights)
        elif initialization_scope == "local":
            for m_name, m in self._modules.items():
                if m_name != "mfvit":
                    m.apply(_init_weights)

    def forward(
        self,
        x: Union[torch.Tensor, list[torch.Tensor]],
        feature_extraction_batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return self.forward_batch(x)
        elif isinstance(x, list):
            if feature_extraction_batch_size is None:
                feature_extraction_batch_size = len(x)
            return self.forward_arbitrary_resolution_batch(
                x, feature_extraction_batch_size)
        else:
            raise TypeError('x must be a tensor or a list of tensors')

    def patches_attention(self, x: torch.Tensor) -> torch.Tensor:
        aggregator = self.patch_aggregator.expand(x.size(0), -1, -1, -1)
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        dots = torch.matmul(aggregator, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x)
        x = x.squeeze(dim=1)
        return x

    def forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        x = model_utils.patchify_image(
            x,
            (self.img_patch_size, self.img_patch_size),
            (self.img_patch_stride, self.img_patch_stride)
        )

        patch_features = []
        for i in range(x.size(1)):
            patch_features.append(self.mfvit(x[:, i]))
        x = torch.stack(patch_features, dim=1)
        del patch_features

        x = self.patches_attention(x)
        x = self.norm(x)
        x = self.cls_head(x)
        return x

    def forward_arbitrary_resolution_batch(
        self,
        x: list[torch.Tensor],
        feature_extraction_batch_size: int
    ) -> torch.Tensor:
        patched_images = []
        for img in x:
            patched = model_utils.patchify_image(
                img,
                (self.img_patch_size, self.img_patch_size),
                (self.img_patch_stride, self.img_patch_stride)
            )
            if patched.size(1) < self.minimum_patches:
                patched = five_crop(
                    img, [self.img_patch_size, self.img_patch_size])
                patched = torch.stack(patched, dim=1)
            patched_images.append(patched)
        x = patched_images
        del patched_images

        img_patches_num = [img.size(1) for img in x]
        x = torch.cat(x, dim=1)
        x = x.squeeze(dim=0)

        features = []
        for i in range(0, x.size(0), feature_extraction_batch_size):
            features.append(
                self.mfvit(x[i:i + feature_extraction_batch_size]))
        x = torch.cat(features, dim=0)
        del features

        attended = []
        processed_sum = 0
        for i in img_patches_num:
            attended.append(self.patches_attention(
                x[processed_sum:processed_sum + i].unsqueeze(0)))
            processed_sum += i
        x = torch.cat(attended, dim=0)
        del attended

        x = self.norm(x)
        x = self.cls_head(x)
        return x

    def get_vision_transformer(self):
        return self.mfvit.get_vision_transformer()


class MFViT(nn.Module):
    def __init__(
        self,
        vit: vision_transformer.VisionTransformer,
        features_processor: 'FrequencyRestorationEstimator',
        cls_head: Optional[nn.Module],
        masking_radius: int,
        img_size: int,
        frozen_backbone: bool = True,
        initialization_scope: str = "all"
    ):
        super().__init__()
        self.vit = vit
        self.features_processor = features_processor
        self.cls_head = cls_head

        if initialization_scope == "all":
            self.apply(_init_weights)
        elif initialization_scope == "local":
            for m_name, m in self._modules.items():
                if m_name != "vit":
                    m.apply(_init_weights)

        self.frozen_backbone = frozen_backbone

        self.frequencies_mask = nn.Parameter(
            filters.generate_circular_mask(img_size, masking_radius),
            requires_grad=False
        )

        self.backbone_norm = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_freq, hi_freq = filters.filter_image_frequencies(
            x.float(), self.frequencies_mask)

        low_freq = torch.clamp(low_freq, min=0., max=1.).to(x.dtype)
        hi_freq = torch.clamp(hi_freq, min=0., max=1.).to(x.dtype)

        x = self.backbone_norm(x)
        low_freq = self.backbone_norm(low_freq)
        hi_freq = self.backbone_norm(hi_freq)

        if self.frozen_backbone:
            with torch.no_grad():
                x, low_freq, hi_freq = self._extract_features(
                    x, low_freq, hi_freq)
        else:
            x, low_freq, hi_freq = self._extract_features(
                x, low_freq, hi_freq)

        x = self.features_processor(x, low_freq, hi_freq)
        if self.cls_head is not None:
            x = self.cls_head(x)
        return x

    def get_vision_transformer(self):
        return self.vit

    def _extract_features(self, x, low_freq, hi_freq):
        x = self.vit(x)
        low_freq = self.vit(low_freq)
        hi_freq = self.vit(hi_freq)
        return x, low_freq, hi_freq


class FrequencyRestorationEstimator(nn.Module):
    def __init__(
        self,
        features_num: int,
        input_dim: int,
        proj_dim: int,
        proj_layers: int,
        patch_projection: bool = False,
        patch_projection_per_feature: bool = False,
        proj_last_layer_activation_type: Optional[str] = "gelu",
        original_image_features_branch: bool = False,
        dropout: float = 0.5,
        disable_reconstruction_similarity: bool = False
    ):
        super().__init__()

        if proj_last_layer_activation_type == "gelu":
            proj_last_layer_activation = nn.GELU
        elif proj_last_layer_activation_type is None:
            proj_last_layer_activation = nn.Identity
        else:
            raise RuntimeError(
                f"Unsupported activation: {proj_last_layer_activation_type}")

        if patch_projection and patch_projection_per_feature:
            self.patch_projector = FeatureSpecificProjector(
                features_num, proj_layers, input_dim, proj_dim,
                proj_last_layer_activation, dropout=dropout)
        elif patch_projection:
            self.patch_projector = Projector(
                proj_layers, input_dim, proj_dim,
                proj_last_layer_activation, dropout=dropout)
        else:
            self.patch_projector = nn.Identity()

        self.original_features_processor = None
        if original_image_features_branch:
            self.original_features_processor = FeatureImportanceProjector(
                features_num, proj_dim, proj_dim, proj_layers, dropout=dropout)

        self.disable_reconstruction_similarity = disable_reconstruction_similarity

    def forward(self, x, low_freq, hi_freq):
        orig = self.patch_projector(x)
        low_freq = self.patch_projector(low_freq)
        hi_freq = self.patch_projector(hi_freq)

        if self.disable_reconstruction_similarity:
            x = self.original_features_processor(orig)
        else:
            sim_x_low = F.cosine_similarity(orig, low_freq, dim=-1)
            sim_x_hi = F.cosine_similarity(orig, hi_freq, dim=-1)
            sim_low_hi = F.cosine_similarity(low_freq, hi_freq, dim=-1)

            x = torch.cat([
                sim_x_low.mean(dim=-1), sim_x_low.std(dim=-1),
                sim_x_hi.mean(dim=-1), sim_x_hi.std(dim=-1),
                sim_low_hi.mean(dim=-1), sim_low_hi.std(dim=-1),
            ], dim=1)

            if self.original_features_processor is not None:
                orig = self.original_features_processor(orig)
                x = torch.cat([x, orig], dim=1)

        return x


class FeatureSpecificProjector(nn.Module):
    def __init__(self, intermediate_features_num, proj_layers, input_dim,
                 proj_dim, last_layer_activation=nn.GELU, dropout=0.5):
        super().__init__()
        self.projectors = nn.ModuleList([
            Projector(proj_layers, input_dim, proj_dim,
                      last_layer_activation, dropout=dropout)
            for _ in range(intermediate_features_num)
        ])

    def forward(self, x):
        projected = []
        for i, projector in enumerate(self.projectors):
            projected.append(projector(x[:, i, :, :]))
        x = torch.stack(projected, dim=1)
        return x


class Projector(nn.Module):
    def __init__(self, proj_layers, input_dim, proj_dim,
                 last_layer_activation=nn.GELU, input_norm=True,
                 output_norm=True, dropout=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim) if input_norm else nn.Identity()
        patch_proj_layers = [nn.Dropout(dropout)]
        for i in range(proj_layers):
            patch_proj_layers.extend([
                nn.Linear(input_dim if i == 0 else proj_dim, proj_dim),
                nn.GELU() if i < proj_layers - 1 else last_layer_activation(),
                nn.Dropout(dropout),
            ])
        self.projector = nn.Sequential(*patch_proj_layers)
        self.norm2 = nn.LayerNorm(proj_dim) if output_norm else nn.Identity()

    def forward(self, x):
        x = self.norm1(x)
        x = self.projector(x)
        x = self.norm2(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, mlp_ratio=1, dropout=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim * mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * mlp_ratio, input_dim * mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * mlp_ratio, num_classes)
        )

    def forward(self, x):
        return self.head(x)


class FeatureImportanceProjector(nn.Module):
    def __init__(self, intermediate_features_num, input_dim, proj_dim,
                 proj_layers, dropout=0.5):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.randn([1, intermediate_features_num, proj_dim]))
        self.proj1 = Projector(proj_layers, 2 * proj_dim, proj_dim,
                               input_norm=False, dropout=dropout)
        self.proj2 = Projector(proj_layers, proj_dim, proj_dim,
                               input_norm=False, dropout=dropout)

    def forward(self, x):
        x_mean = x.mean(dim=2)
        x_std = x.std(dim=2)
        x = torch.cat([x_mean, x_std], dim=-1)
        x = self.proj1(x)
        x = torch.softmax(self.alpha, dim=1) * x
        x = torch.sum(x, dim=1)
        x = self.proj2(x)
        return x


def _init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(m.weight, std=.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def build_mf_vit(
    img_size=224,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    qkv_bias=True,
    drop_path_rate=0.1,
    init_values=None,
    intermediate_layers=tuple(range(12)),
    projection_dim=1024,
    projection_layers=2,
    masking_radius=16,
    patch_stride=224,
    minimum_patches=4,
    cls_head_mlp_ratio=3,
    sid_dropout=0.5,
    num_classes=2,
    resolution_mode="arbitrary",
):
    vit = vision_transformer.build_vit(
        img_size=img_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path_rate,
        init_values=init_values,
        num_classes=num_classes,
        use_intermediate_layers=True,
        intermediate_layers=intermediate_layers,
    )

    fre = FrequencyRestorationEstimator(
        features_num=len(intermediate_layers),
        input_dim=embed_dim,
        proj_dim=projection_dim,
        proj_layers=projection_layers,
        patch_projection=True,
        patch_projection_per_feature=True,
        proj_last_layer_activation_type=None,
        original_image_features_branch=True,
        dropout=sid_dropout,
        disable_reconstruction_similarity=False,
    )

    cls_vector_dim = 6 * len(intermediate_layers) + projection_dim

    cls_head = ClassificationHead(
        input_dim=cls_vector_dim,
        num_classes=1 if num_classes <= 2 else num_classes,
        mlp_ratio=cls_head_mlp_ratio,
        dropout=sid_dropout,
    )

    if resolution_mode == "arbitrary":
        model = PatchBasedMFViT(
            vit,
            fre,
            cls_head,
            masking_radius=masking_radius,
            img_patch_size=img_size,
            img_patch_stride=patch_stride,
            cls_vector_dim=cls_vector_dim,
            attn_embed_dim=1536,
            num_heads=num_heads,
            dropout=sid_dropout,
            minimum_patches=minimum_patches,
            initialization_scope="all",
        )
    else:
        model = MFViT(
            vit, fre, cls_head,
            masking_radius=masking_radius,
            img_size=img_size,
        )

    return model
