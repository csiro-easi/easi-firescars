"""Unit tests for modules/data_access.py"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from modules.data_access import get_chip_paths, load_splits, verify_chip


@pytest.fixture
def sample_chip(tmp_path):
    """Create a synthetic image/mask pair for testing."""
    data_dir = tmp_path / "hls_burn_scars" / "data"
    data_dir.mkdir(parents=True)

    # Write 6-band image
    img_path = data_dir / "chip001_merged.tif"
    transform = from_origin(0, 1, 0.001, 0.001)
    with rasterio.open(
        img_path, "w", driver="GTiff", height=512, width=512,
        count=6, dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(np.random.rand(6, 512, 512).astype(np.float32))

    # Write mask
    mask_path = data_dir / "chip001.mask.tif"
    mask_data = np.zeros((1, 512, 512), dtype=np.int16)
    mask_data[0, 100:200, 100:200] = 1  # Burned region
    mask_data[0, 0:10, 0:10] = -1  # No data
    with rasterio.open(
        mask_path, "w", driver="GTiff", height=512, width=512,
        count=1, dtype="int16", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(mask_data)

    return tmp_path / "hls_burn_scars", "chip001"


@pytest.fixture
def sample_splits(tmp_path):
    """Create synthetic split files."""
    dataset_dir = tmp_path / "hls_burn_scars"
    splits_dir = dataset_dir / "splits"
    splits_dir.mkdir(parents=True)

    (splits_dir / "train.txt").write_text("chip001\nchip002\nchip003\n")
    (splits_dir / "val.txt").write_text("chip004\n")
    (splits_dir / "test.txt").write_text("chip005\n")

    return dataset_dir


def test_load_splits(sample_splits):
    splits = load_splits(sample_splits)
    assert len(splits["train"]) == 3
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1
    assert "chip001" in splits["train"]


def test_load_splits_with_max_chips(sample_splits):
    splits = load_splits(sample_splits, max_chips=3)
    total = sum(len(v) for v in splits.values())
    assert total <= 3


def test_verify_chip(sample_chip):
    dataset_dir, chip_name = sample_chip
    img_path, mask_path = get_chip_paths(dataset_dir, chip_name)
    info = verify_chip(img_path, mask_path)

    assert info["image_shape"] == (6, 512, 512)
    assert info["image_dtype"] == "float32"
    assert -1 in info["mask_values"]
    assert 0 in info["mask_values"]
    assert 1 in info["mask_values"]
    assert info["has_burned"] is True


def test_get_chip_paths():
    dataset_dir = Path("/fake/hls_burn_scars")
    img, mask = get_chip_paths(dataset_dir, "chip001")
    assert img == dataset_dir / "data" / "chip001_merged.tif"
    assert mask == dataset_dir / "data" / "chip001.mask.tif"
