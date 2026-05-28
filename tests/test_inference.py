"""Unit tests for modules/inference.py"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from modules.inference import load_scene, normalise_scene, save_geotiff
from modules.training import BAND_MEANS, BAND_STDS


def test_normalise_scene():
    data = np.zeros((6, 4, 4), dtype=np.float32)
    # Set each band to its mean — normalised should be ~0
    for i in range(6):
        data[i] = BAND_MEANS[i]

    result = normalise_scene(data)
    assert result.shape == (6, 4, 4)
    np.testing.assert_allclose(result, 0.0, atol=1e-6)


def test_normalise_scene_shape_preserved():
    data = np.random.rand(6, 64, 64).astype(np.float32)
    result = normalise_scene(data)
    assert result.shape == data.shape


def test_load_scene(tmp_path):
    scene_path = tmp_path / "scene.tif"
    transform = from_origin(0, 1, 0.001, 0.001)
    data = np.random.rand(6, 32, 32).astype(np.float32)

    with rasterio.open(
        scene_path, "w", driver="GTiff", height=32, width=32,
        count=6, dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)

    loaded, meta = load_scene(scene_path)
    assert loaded.shape == (6, 32, 32)
    assert meta["crs"] is not None
    assert meta["width"] == 32
    np.testing.assert_allclose(loaded, data, atol=1e-6)


def test_save_geotiff(tmp_path):
    mask = np.ones((32, 32), dtype=np.int8)
    transform = from_origin(0, 1, 0.001, 0.001)
    output_path = tmp_path / "output" / "pred.tif"

    result = save_geotiff(mask, output_path, crs="EPSG:4326", transform=transform)

    assert result.exists()
    with rasterio.open(result) as src:
        read_data = src.read(1)
        assert read_data.shape == (32, 32)
        assert (read_data == 1).all()
