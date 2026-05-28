"""OLMo Earth encoder + UPerNet segmentation head."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class UPerNetHead(nn.Module):
    """Simplified UPerNet decoder for single-class segmentation."""

    def __init__(self, in_channels_list, hidden_dim=256):
        super().__init__()
        # Lateral convolutions (1x1) to unify channel dims
        self.laterals = nn.ModuleList([nn.Conv2d(c, hidden_dim, 1) for c in in_channels_list])
        # FPN smoothing convolutions
        self.smooths = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1) for _ in in_channels_list]
        )
        # PPM (Pyramid Pooling Module) on last feature map
        self.ppm_pools = nn.ModuleList([nn.AdaptiveAvgPool2d(s) for s in [1, 2, 3, 6]])
        self.ppm_convs = nn.ModuleList(
            [nn.Conv2d(in_channels_list[-1], hidden_dim, 1) for _ in [1, 2, 3, 6]]
        )
        # Fuse PPM + FPN
        self.bottleneck = nn.Conv2d(
            hidden_dim * (len(in_channels_list) + len(self.ppm_pools)),
            hidden_dim,
            3,
            padding=1,
        )
        self.head = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, features):
        """features: list of feature maps from encoder stages, low-res to high-res."""
        target_size = features[0].shape[2:]

        # PPM on deepest feature
        ppm_out = []
        for pool, conv in zip(self.ppm_pools, self.ppm_convs):
            x = pool(features[-1])
            x = conv(x)
            x = F.interpolate(x, size=features[-1].shape[2:], mode="bilinear", align_corners=False)
            ppm_out.append(x)

        # FPN top-down
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        fpn_out = [s(lat) for s, lat in zip(self.smooths, laterals)]

        # Upsample all to same size and concatenate
        upsampled = [
            F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            for f in fpn_out
        ]
        ppm_upsampled = [
            F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            for p in ppm_out
        ]

        fused = torch.cat(upsampled + ppm_upsampled, dim=1)
        fused = self.bottleneck(fused)
        return self.head(fused)


class FirescarModel(nn.Module):
    """OLMo Earth encoder with UPerNet decoder for binary burn segmentation."""

    def __init__(
        self,
        encoder_name="vit_base_patch16_224",
        in_chans=6,
        img_size=224,
        pretrained_encoder=True,
        encoder_weights_path=None,
    ):
        super().__init__()
        self.img_size = img_size

        # Create ViT encoder with intermediate feature extraction
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained_encoder,
            in_chans=in_chans,
            img_size=img_size,
            features_only=False,
            num_classes=0,  # remove classification head
        )
        embed_dim = self.encoder.embed_dim
        num_patches_side = img_size // self.encoder.patch_embed.patch_size[0]

        # Load OLMo Earth weights if provided
        if encoder_weights_path:
            state = torch.load(encoder_weights_path, map_location="cpu", weights_only=True)
            # Handle different checkpoint formats
            if "model" in state:
                state = state["model"]
            if "state_dict" in state:
                state = state["state_dict"]
            # Filter to encoder keys and load
            encoder_state = {
                k.replace("encoder.", ""): v
                for k, v in state.items()
                if "encoder" in k or "patch_embed" in k or "blocks" in k
            }
            if encoder_state:
                self.encoder.load_state_dict(encoder_state, strict=False)

        # We extract features at 4 intermediate stages from the ViT blocks
        num_blocks = len(self.encoder.blocks)
        self.stage_indices = [
            num_blocks // 4 - 1,
            num_blocks // 2 - 1,
            3 * num_blocks // 4 - 1,
            num_blocks - 1,
        ]
        self.num_patches_side = num_patches_side

        # UPerNet decoder
        self.decoder = UPerNetHead(
            in_channels_list=[embed_dim] * 4,
            hidden_dim=256,
        )

    def extract_features(self, x):
        """Extract intermediate features from ViT blocks."""
        B = x.shape[0]
        x = self.encoder.patch_embed(x)
        if hasattr(self.encoder, "cls_token") and self.encoder.cls_token is not None:
            cls_token = self.encoder.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            has_cls = True
        else:
            has_cls = False

        if hasattr(self.encoder, "pos_embed") and self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed
        x = self.encoder.pos_drop(x) if hasattr(self.encoder, "pos_drop") else x
        x = self.encoder.patch_drop(x) if hasattr(self.encoder, "patch_drop") else x
        x = self.encoder.norm_pre(x) if hasattr(self.encoder, "norm_pre") else x

        features = []
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in self.stage_indices:
                tokens = x[:, 1:] if has_cls else x
                feat = tokens.transpose(1, 2).reshape(
                    B, -1, self.num_patches_side, self.num_patches_side
                )
                features.append(feat)
        return features

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.decoder(features)
        # Upsample to input resolution
        logits = F.interpolate(
            logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )
        return logits.squeeze(1)  # (B, H, W)

    def get_param_groups(self, encoder_lr_mult=0.1):
        """Return parameter groups with different LRs for encoder vs decoder."""
        encoder_params = list(self.encoder.parameters())
        decoder_params = list(self.decoder.parameters())
        return [
            {"params": encoder_params, "lr_mult": encoder_lr_mult},
            {"params": decoder_params, "lr_mult": 1.0},
        ]
