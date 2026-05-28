"""Tests for modules/train.py — loss and metrics."""

import torch
from modules.train import dice_bce_loss, compute_iou


def test_dice_bce_loss_perfect_prediction():
    """Perfect prediction should give low loss."""
    logits = torch.zeros(2, 2, 4, 4)
    logits[:, 1, :, :] = 10.0  # high confidence class 1
    targets = torch.ones(2, 4, 4, dtype=torch.long)
    loss = dice_bce_loss(logits, targets)
    assert loss.item() < 0.1


def test_dice_bce_loss_wrong_prediction():
    """Wrong prediction should give high loss."""
    logits = torch.zeros(2, 2, 4, 4)
    logits[:, 0, :, :] = 10.0  # high confidence class 0
    targets = torch.ones(2, 4, 4, dtype=torch.long)
    loss = dice_bce_loss(logits, targets)
    assert loss.item() > 0.5


def test_compute_iou_perfect():
    logits = torch.zeros(1, 2, 4, 4)
    logits[:, 1, :2, :] = 10.0
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    targets[:, :2, :] = 1
    iou = compute_iou(logits, targets)
    assert iou > 0.99


def test_compute_iou_no_overlap():
    logits = torch.zeros(1, 2, 4, 4)
    logits[:, 1, :2, :] = 10.0  # predict top half
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    targets[:, 2:, :] = 1  # truth is bottom half
    iou = compute_iou(logits, targets)
    assert iou == 0.0


def test_compute_iou_ignores_255():
    logits = torch.zeros(1, 2, 4, 4)
    logits[:, 1, :, :] = 10.0
    targets = torch.full((1, 4, 4), 255, dtype=torch.long)
    # All pixels ignored — division by clamp(min=1) → 0/1 = 0
    iou = compute_iou(logits, targets)
    assert iou == 0.0
