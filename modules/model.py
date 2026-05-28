"""Granite-geospatial-uki burn scar segmentation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from terratorch.models.backbones.prithvi_vit import PrithviViT


class GraniteUKIBurnScar(nn.Module):
    """Binary burn scar segmentation using granite-geospatial-uki backbone."""

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()
        self.embed_dim = 768
        self.patch_size = 16

        # Load pretrained granite-geospatial-uki (PrithviViT architecture)
        self.backbone = PrithviViT(
            img_size=224,
            num_frames=3,
            patch_size=[1, 16, 16],
            in_chans=8,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
        )
        ckpt_path = hf_hub_download(
            "ibm-granite/granite-geospatial-uki", "granite_geospatial_uki.pt"
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = {
            k: v for k, v in ckpt["model"].items()
            if not k.startswith("decoder") and k not in ("mask_token", "decoder_pos_embed")
        }
        self.backbone.load_state_dict(state, strict=False)

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
        # backbone expects (B, C, T, H, W)
        x_in = x.unsqueeze(2)

        features, _, _ = self.backbone(x_in, mask_ratio=0.0)
        # features: (B, 1+num_patches, embed_dim) — drop cls token
        features = features[:, 1:, :]

        # Reshape tokens → spatial grid
        h_p, w_p = H // self.patch_size, W // self.patch_size
        features = features.permute(0, 2, 1).view(B, self.embed_dim, h_p, w_p)

        logits = self.decoder(features)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
