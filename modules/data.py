"""Dataset and normalisation for ImpactMesh-Fire with granite-geospatial-uki."""

from pathlib import Path
from typing import List

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def normalize_s2(arr: np.ndarray) -> np.ndarray:
    """Sentinel-2 L2A reflectance: scale to [0, 1]."""
    return np.clip(arr.astype(np.float32) / 10000.0, 0.0, 1.0)


def normalize_s1(arr: np.ndarray) -> np.ndarray:
    """Sentinel-1 RTC linear power → dB, clip [-35, 10], scale to [0, 1]."""
    arr = arr.astype(np.float32)
    arr = np.where(arr > 0, 10.0 * np.log10(arr + 1e-10), -35.0)
    arr = np.clip(arr, -35.0, 10.0)
    return (arr + 35.0) / 45.0


def get_split_samples(data_root: Path, split: str) -> List[str]:
    """Load sample IDs from the split txt file."""
    split_file = data_root / "split" / f"impactmesh_wildfire_{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    return [s.strip() for s in split_file.read_text().split() if s.strip()]


def identify_australian_events(samples: List[str], prefixes: tuple = ("EMSR408",)) -> List[str]:
    """Filter sample IDs to those matching Australian EMSR activation codes."""
    return [s for s in samples if any(s.startswith(p) for p in prefixes)]


class ImpactMeshFireDataset(Dataset):
    """PyTorch Dataset for ImpactMesh-Fire burn scar segmentation."""

    def __init__(self, data_root: str, split: str = "train", samples: List[str] = None):
        self.data_root = Path(data_root)
        if samples is not None:
            self.samples = samples
        else:
            self.samples = get_split_samples(self.data_root, split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.samples[idx]

        # Load S2 (6 bands)
        s2_path = self.data_root / "data" / "S2L2A" / f"{sample_id}.tif"
        with rasterio.open(s2_path) as src:
            s2 = src.read()
        s2 = normalize_s2(s2[:6])

        # Load S1 (2 bands)
        s1_path = self.data_root / "data" / "S1RTC" / f"{sample_id}.tif"
        with rasterio.open(s1_path) as src:
            s1 = src.read()
        s1 = normalize_s1(s1[:2])

        # 8-band input
        image = np.concatenate([s2, s1], axis=0)

        # Load mask
        mask_path = self.data_root / "data" / "MASK" / f"{sample_id}.tif"
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        mask = (mask > 0).astype(np.int64)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
        }
