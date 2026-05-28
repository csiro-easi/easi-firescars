"""Dataset and normalisation for ImpactMesh-Fire with granite-geospatial-uki.

Data format (discovered):
- S2L2A: zarr.zip, shape (4, 12, 256, 256) int16, bands: B01-B12
- S1RTC: zarr.zip, shape (4, 2, 256, 256), bands: vv, vh
- MASK: GeoTIFF, shape (1, 256, 256) int8, binary 0/1
- 4 timestamps per sample: pre-month(0), pre-event(1), event(2), post-event(3)
"""

from pathlib import Path
from typing import List

import numpy as np
import rasterio
import torch
import zarr
from torch.utils.data import Dataset

# Band indices in S2L2A zarr (B01=0, B02=1, ..., B12=11)
# granite-uki expects: Blue(B02), Green(B03), Red(B04), NIR(B8A), SWIR1(B11), SWIR2(B12)
S2_BAND_INDICES = [1, 2, 3, 8, 10, 11]  # B02, B03, B04, B8A, B11, B12

# Temporal index for "event" timestamp
EVENT_TIME_IDX = 2


def normalize_s2(arr: np.ndarray) -> np.ndarray:
    """Sentinel-2 L2A reflectance (int16 scaled by 10000): scale to [0, 1]."""
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
    """PyTorch Dataset for ImpactMesh-Fire burn scar segmentation (zarr.zip format)."""

    def __init__(self, data_root: str, split: str = "train", samples: List[str] = None):
        self.data_root = Path(data_root)
        if samples is not None:
            self.samples = samples
        else:
            self.samples = get_split_samples(self.data_root, split)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_zarr(self, path: Path, band_indices: list, time_idx: int) -> np.ndarray:
        """Load bands from a zarr.zip at the specified timestamp."""
        store = zarr.storage.ZipStore(str(path), mode="r")
        root = zarr.open(store, mode="r")
        # bands shape: (T, C, H, W)
        data = root["bands"][time_idx, band_indices, :, :]
        store.close()
        return np.array(data)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.samples[idx]

        # Load S2 (6 bands at event timestamp)
        s2_path = self.data_root / "data" / "S2L2A" / f"{sample_id}_S2L2A.zarr.zip"
        s2 = self._load_zarr(s2_path, S2_BAND_INDICES, EVENT_TIME_IDX)
        s2 = normalize_s2(s2)

        # Load S1 (2 bands at event timestamp)
        s1_path = self.data_root / "data" / "S1RTC" / f"{sample_id}_S1RTC.zarr.zip"
        s1 = self._load_zarr(s1_path, [0, 1], EVENT_TIME_IDX)
        s1 = normalize_s1(s1)

        # 8-band input: [B02, B03, B04, B8A, B11, B12, VV, VH]
        image = np.concatenate([s2, s1], axis=0)

        # Load mask
        mask_path = self.data_root / "data" / "MASK" / f"{sample_id}_annotation_wildfire.tif"
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        mask = mask.astype(np.int64)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
        }
