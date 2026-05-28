"""Training loop, loss, and metrics for burn scar fine-tuning."""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    """Combined Dice + BCE loss for binary segmentation."""
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1], targets.float(), reduction="mean"
    )
    probs = torch.softmax(logits, dim=1)[:, 1]
    intersection = (probs * targets.float()).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (probs.sum() + targets.float().sum() + 1.0)
    return bce_weight * bce + (1.0 - bce_weight) * dice


def compute_iou(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """IoU for the burn scar class (class=1)."""
    preds = logits.argmax(dim=1)
    valid = targets != 255
    p, t = preds[valid], targets[valid]
    intersection = ((p == 1) & (t == 1)).sum().float()
    union = ((p == 1) | (t == 1)).sum().float()
    return (intersection / union.clamp(min=1)).item()


def train_epoch(model, loader, optimizer, scaler, device) -> dict:
    """Run one training epoch. Returns dict with loss and iou."""
    model.train()
    total_loss, total_iou, n = 0.0, 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            logits = model(images)
            loss = dice_bce_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_iou += compute_iou(logits.detach(), masks)
        n += 1

    return {"loss": total_loss / max(n, 1), "iou": total_iou / max(n, 1)}


@torch.no_grad()
def val_epoch(model, loader, device) -> dict:
    """Run one validation epoch. Returns dict with loss and iou."""
    model.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with autocast("cuda"):
            logits = model(images)
            loss = dice_bce_loss(logits, masks)
        total_loss += loss.item()
        total_iou += compute_iou(logits, masks)
        n += 1
    return {"loss": total_loss / max(n, 1), "iou": total_iou / max(n, 1)}
