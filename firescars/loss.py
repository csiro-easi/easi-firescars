"""Combined BCE + Dice loss for binary segmentation."""

import torch
import torch.nn.functional as F


def dice_loss(logits, targets, smooth=1.0):
    """Dice loss from logits."""
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(-2, -1))
    union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class FirescarLoss(torch.nn.Module):
    """0.5 * BCE + 0.5 * Dice loss."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
