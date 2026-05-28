"""Granite-geospatial-uki burn scar segmentation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from terratorch.registry import BACKBONE_REGISTRY


class GraniteUKIBurnScar(nn.Module):
    """Binary burn scar segmentation using granite-geospatial-uki backbone."""

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()
        self.embed_dim = 768
        self.patch_size = 16

        self.backbone = BACKBONE_REGISTRY.build(
            "ibm-granite/granite-geospatial-uki", pretrained=True
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_in = x.unsqueeze(2)  # (B, C, T=1, H, W)

        features = self.backbone(x_in)
        if isinstance(features, (list, tuple)):
            features = features[-1]

        # Reshape tokens → spatial grid
        if features.dim() == 3:
            h_p, w_p = H // self.patch_size, W // self.patch_size
            expected = h_p * w_p
            if features.shape[1] > expected:
                features = features[:, -expected:, :]
            features = features.permute(0, 2, 1).view(B, self.embed_dim, h_p, w_p)

        logits = self.decoder(features)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
