"""Tests for modules/data.py — normalisation and utilities."""

import numpy as np
from modules.data import normalize_s2, normalize_s1, identify_australian_events


def test_normalize_s2_range():
    arr = np.array([0, 5000, 10000, 15000], dtype=np.float32)
    result = normalize_s2(arr)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    assert np.isclose(result[1], 0.5)


def test_normalize_s2_clips():
    arr = np.array([-100, 20000], dtype=np.float32)
    result = normalize_s2(arr)
    assert result[0] == 0.0
    assert result[1] == 1.0


def test_normalize_s1_range():
    # Linear power 0.001 → ~-30 dB, 1.0 → 0 dB
    arr = np.array([0.001, 0.01, 0.1, 1.0], dtype=np.float32)
    result = normalize_s1(arr)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_normalize_s1_zero_handling():
    arr = np.array([0.0, -1.0], dtype=np.float32)
    result = normalize_s1(arr)
    # Both should map to 0.0 (clipped at -35 dB → scaled to 0)
    assert result[0] == 0.0
    assert result[1] == 0.0


def test_identify_australian_events():
    samples = ["EMSR408_11_56HKJ_x1_y2", "EMSR408_12_56HKJ_x3_y4", "EMSR633_1_30TXQ_x5_y6"]
    result = identify_australian_events(samples)
    assert len(result) == 2
    assert all("EMSR408" in s for s in result)


def test_identify_australian_events_empty():
    samples = ["EMSR633_1_30TXQ_x5_y6", "EMSR765_14_21KTA_x1_y2"]
    result = identify_australian_events(samples)
    assert len(result) == 0
