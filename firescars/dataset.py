"""Zarr-backed dataset for firescar chips."""

import json

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class FirescarDataset(Dataset):
    """Loads imagery/mask chip pairs from a Zarr store."""

    def __init__(self, zarr_path, split="train", band_stats=None, augment=False):
        store = zarr.open(zarr_path, mode="r")
        self.imagery = store[split]["imagery"]  # (N, 6, 224, 224) int16
        self.masks = store[split]["masks"]  # (N, 224, 224) uint8
        self.augment = augment
        self.band_stats = band_stats  # dict with per-band mean/std

    def __len__(self):
        return self.imagery.shape[0]

    def __getitem__(self, idx):
        img = self.imagery[idx].astype(np.float32)  # (6, H, W)
        mask = self.masks[idx].astype(np.float32)  # (H, W)

        # Replace nodata (-999) with 0
        img[img == -999] = 0.0

        # Normalise to [0, 1] using band stats
        if self.band_stats:
            for i, band in enumerate(self.band_stats):
                stats = self.band_stats[band]
                img[i] = (img[i] - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)
        else:
            img = img / 10000.0  # default S2 reflectance scaling

        img = np.clip(img, 0.0, 1.0)

        # Augmentation
        if self.augment:
            img, mask = self._augment(img, mask)

        return torch.from_numpy(img), torch.from_numpy(mask)

    def _augment(self, img, mask):
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = img[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
        # Random vertical flip
        if np.random.random() > 0.5:
            img = img[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()
        # Random 90-degree rotation
        k = np.random.randint(4)
        if k > 0:
            img = np.rot90(img, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return img, mask


def load_band_stats(metadata_path):
    """Load band statistics from metadata.json."""
    with open(metadata_path) as f:
        meta = json.load(f)
    return meta.get("band_stats")
