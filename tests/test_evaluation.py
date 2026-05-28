"""Unit tests for modules/evaluation.py"""

import numpy as np

from modules.evaluation import compute_iou, evaluate_test_set, find_best_worst_chips


def test_compute_iou_perfect():
    pred = np.array([[1, 1, 0], [0, 0, 0]])
    target = np.array([[1, 1, 0], [0, 0, 0]])
    result = compute_iou(pred, target)
    assert result["iou_burned"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_compute_iou_no_burned():
    pred = np.zeros((4, 4), dtype=int)
    target = np.zeros((4, 4), dtype=int)
    result = compute_iou(pred, target)
    assert result["iou_burned"] == 0.0  # No burned pixels to measure
    assert result["iou_unburned"] == 1.0


def test_compute_iou_ignores_nodata():
    pred = np.array([[1, 0], [0, 1]])
    target = np.array([[-1, 0], [0, 1]])
    result = compute_iou(pred, target, ignore_index=-1)
    # Only 3 valid pixels: (0,1)=correct, (1,0)=correct, (1,1)=correct
    assert result["iou_burned"] == 1.0
    assert result["recall"] == 1.0


def test_compute_iou_partial():
    # pred has one false positive, one true positive
    pred = np.array([[1, 1], [0, 0]])
    target = np.array([[1, 0], [0, 0]])
    result = compute_iou(pred, target)
    # TP=1, FP=1, FN=0 → IoU = 1/(1+1+0) = 0.5
    assert abs(result["iou_burned"] - 0.5) < 1e-6
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0


def test_evaluate_test_set():
    preds = [np.ones((4, 4), dtype=int), np.zeros((4, 4), dtype=int)]
    targets = [np.ones((4, 4), dtype=int), np.zeros((4, 4), dtype=int)]
    result = evaluate_test_set(preds, targets)
    assert result["aggregate"]["num_chips"] == 2
    assert result["aggregate"]["iou_burned"] == 0.5  # 1.0 + 0.0 / 2


def test_find_best_worst_chips():
    per_chip = [
        {"iou_burned": 0.9},
        {"iou_burned": 0.1},
        {"iou_burned": 0.5},
        {"iou_burned": 0.8},
        {"iou_burned": 0.2},
    ]
    result = find_best_worst_chips(per_chip, n=2)
    assert result["best"] == [0, 3]  # indices with 0.9, 0.8
    assert result["worst"] == [4, 1]  # indices with 0.2, 0.1
