"""Training configuration and execution for Prithvi-EO-2.0 fire scar fine-tuning."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml


# Normalisation values from Prithvi-EO-2.0-300M-BurnScars pretrained config
BAND_MEANS = [
    0.033349706741586264,
    0.05701185520536176,
    0.05889748132001316,
    0.2323245113436119,
    0.1972854853760658,
    0.11944914225186566,
]
BAND_STDS = [
    0.02269135568823774,
    0.026807560223070237,
    0.04004109844362779,
    0.07791732423672691,
    0.08708738838140137,
    0.07241979477437814,
]

BAND_NAMES = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]


def check_gpu() -> dict:
    """Check GPU availability and return device info.

    Returns:
        Dict with 'available', 'name', 'memory_gb' keys.
    """
    try:
        import torch

        available = torch.cuda.is_available()
        if available:
            name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {"available": True, "name": name, "memory_gb": round(memory_gb, 1)}
        return {"available": False, "name": None, "memory_gb": None}
    except ImportError:
        return {"available": False, "name": None, "memory_gb": None}


def generate_config(
    dataset_dir: str | Path,
    output_path: str | Path,
    max_epochs: int = 80,
    batch_size: int = 8,
    lr: float = 1e-4,
    precision: str = "bf16-mixed",
) -> Path:
    """Generate a TerraTorch training config YAML.

    Args:
        dataset_dir: Path to hls_burn_scars directory.
        output_path: Where to write the config YAML.
        max_epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        precision: Training precision (bf16-mixed, 16-mixed, 32).

    Returns:
        Path to the written config file.
    """
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)

    config = {
        "seed_everything": 42,
        "trainer": {
            "logger": True,
            "max_epochs": max_epochs,
            "log_every_n_steps": 1,
            "callbacks": [
                {
                    "class_path": "EarlyStopping",
                    "init_args": {"monitor": "val/loss", "patience": 15},
                },
                {
                    "class_path": "LearningRateMonitor",
                    "init_args": {"logging_interval": "epoch"},
                },
            ],
            "enable_progress_bar": True,
            "precision": precision,
        },
        "model": {
            "class_path": "terratorch.tasks.SemanticSegmentationTask",
            "init_args": {
                "model_factory": "EncoderDecoderFactory",
                "model_args": {
                    "backbone": "prithvi_eo_v2_300",
                    "backbone_pretrained": True,
                    "backbone_bands": BAND_NAMES,
                    "necks": [
                        {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                        {"name": "ReshapeTokensToImage"},
                        {"name": "LearnedInterpolateToPyramidal"},
                    ],
                    "decoder": "UNetDecoder",
                    "decoder_channels": [512, 256, 128, 64],
                    "num_classes": 2,
                },
                "loss": "ce",
                "ignore_index": -1,
                "freeze_backbone": False,
                "class_names": ["Not burned", "Burn scar"],
            },
        },
        "optimizer": {
            "class_path": "torch.optim.AdamW",
            "init_args": {"lr": lr},
        },
        "lr_scheduler": {
            "class_path": "ReduceLROnPlateau",
            "init_args": {"monitor": "val/loss", "factor": 0.5, "patience": 4},
        },
        "data": {
            "class_path": "GenericNonGeoSegmentationDataModule",
            "init_args": {
                "batch_size": batch_size,
                "num_workers": 4,
                "dataset_bands": BAND_NAMES,
                "output_bands": BAND_NAMES,
                "rgb_indices": [2, 1, 0],
                "train_data_root": str(dataset_dir / "data"),
                "val_data_root": str(dataset_dir / "data"),
                "test_data_root": str(dataset_dir / "data"),
                "train_split": str(dataset_dir / "splits" / "train.txt"),
                "val_split": str(dataset_dir / "splits" / "val.txt"),
                "test_split": str(dataset_dir / "splits" / "test.txt"),
                "img_grep": "*_merged.tif",
                "label_grep": "*.mask.tif",
                "means": BAND_MEANS,
                "stds": BAND_STDS,
                "num_classes": 2,
                "train_transform": [
                    {"class_path": "albumentations.D4"},
                    {"class_path": "ToTensorV2"},
                ],
                "test_transform": [{"class_path": "ToTensorV2"}],
                "no_data_replace": 0,
                "no_label_replace": -1,
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Config written to {output_path}", flush=True)
    return output_path


def run_training(config_path: str | Path, output_dir: Optional[str | Path] = None) -> int:
    """Run TerraTorch fine-tuning via CLI.

    Args:
        config_path: Path to the YAML config file.
        output_dir: Optional override for output/checkpoint directory.

    Returns:
        Return code from the training process.
    """
    config_path = Path(config_path)
    cmd = [sys.executable, "-m", "terratorch", "fit", "-c", str(config_path)]

    if output_dir:
        cmd.extend(["--trainer.default_root_dir", str(output_dir)])

    print(f"Starting training: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print("Training completed successfully.", flush=True)
    else:
        print(f"Training failed with return code {result.returncode}", flush=True)

    return result.returncode
