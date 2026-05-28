"""Test-set evaluation and metrics for burn scar segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_iou(pred: np.ndarray, target: np.ndarray, ignore_index: int = -1) -> dict:
    """Compute IoU metrics for binary segmentation.

    Args:
        pred: Predicted mask (H, W) with values in {0, 1}.
        target: Ground truth mask (H, W) with values in {-1, 0, 1}.
        ignore_index: Value to ignore in target.

    Returns:
        Dict with iou_burned, iou_unburned, mean_iou, precision, recall, f1.
    """
    valid = target != ignore_index
    pred_v = pred[valid]
    target_v = target[valid]

    # Burned class (1)
    tp = int(((pred_v == 1) & (target_v == 1)).sum())
    fp = int(((pred_v == 1) & (target_v == 0)).sum())
    fn = int(((pred_v == 0) & (target_v == 1)).sum())
    tn = int(((pred_v == 0) & (target_v == 0)).sum())

    iou_burned = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    iou_unburned = tn / (tn + fn + fp) if (tn + fn + fp) > 0 else 0.0
    mean_iou = (iou_burned + iou_unburned) / 2

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "iou_burned": iou_burned,
        "iou_unburned": iou_unburned,
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_test_set(
    predictions: list[np.ndarray], targets: list[np.ndarray], ignore_index: int = -1
) -> dict:
    """Compute aggregate metrics over a list of prediction/target pairs.

    Args:
        predictions: List of predicted masks, each (H, W).
        targets: List of ground truth masks, each (H, W).
        ignore_index: Value to ignore in targets.

    Returns:
        Dict with aggregate and per-chip metrics.
    """
    per_chip = []
    for pred, target in zip(predictions, targets):
        metrics = compute_iou(pred, target, ignore_index)
        per_chip.append(metrics)

    # Aggregate as mean over chips
    keys = ["iou_burned", "iou_unburned", "mean_iou", "precision", "recall", "f1"]
    aggregate = {k: float(np.mean([c[k] for c in per_chip])) for k in keys}
    aggregate["num_chips"] = len(per_chip)

    return {"aggregate": aggregate, "per_chip": per_chip}


def find_best_worst_chips(
    per_chip_metrics: list[dict], n: int = 5
) -> dict[str, list[int]]:
    """Find indices of best and worst performing chips by burned IoU.

    Args:
        per_chip_metrics: List of per-chip metric dicts.
        n: Number of best/worst to return.

    Returns:
        Dict with 'best' and 'worst' lists of indices.
    """
    scores = [(i, m["iou_burned"]) for i, m in enumerate(per_chip_metrics)]
    scores.sort(key=lambda x: x[1], reverse=True)

    best = [idx for idx, _ in scores[:n]]
    worst = [idx for idx, _ in scores[-n:]]
    return {"best": best, "worst": worst}
