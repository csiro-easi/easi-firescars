"""Unit tests for modules/training.py"""

import tempfile
from pathlib import Path

import yaml

from modules.training import BAND_MEANS, BAND_STDS, BAND_NAMES, check_gpu, generate_config


def test_band_constants():
    assert len(BAND_MEANS) == 6
    assert len(BAND_STDS) == 6
    assert len(BAND_NAMES) == 6
    assert all(m > 0 for m in BAND_MEANS)
    assert all(s > 0 for s in BAND_STDS)


def test_check_gpu_returns_dict():
    result = check_gpu()
    assert "available" in result
    assert "name" in result
    assert "memory_gb" in result
    assert isinstance(result["available"], bool)


def test_generate_config(tmp_path):
    dataset_dir = tmp_path / "hls_burn_scars"
    dataset_dir.mkdir()
    (dataset_dir / "splits").mkdir()
    (dataset_dir / "data").mkdir()

    config_path = tmp_path / "config.yaml"
    result = generate_config(
        dataset_dir=dataset_dir,
        output_path=config_path,
        max_epochs=5,
        batch_size=4,
        lr=1e-4,
    )

    assert result.exists()
    with open(result) as f:
        config = yaml.safe_load(f)

    assert config["trainer"]["max_epochs"] == 5
    assert config["data"]["init_args"]["batch_size"] == 4
    assert config["model"]["init_args"]["model_args"]["backbone"] == "prithvi_eo_v2_300"
    assert config["model"]["init_args"]["model_args"]["num_classes"] == 2
    assert len(config["data"]["init_args"]["means"]) == 6
