"""Dataset download, split loading, and verification for HLS Burn Scars."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio


def download_dataset(data_dir: str | Path, max_chips: Optional[int] = None) -> Path:
    """Download HLS Burn Scars dataset from Hugging Face.

    Args:
        data_dir: Directory to store the dataset.
        max_chips: If set, only download/use this many chips (for quick testing).

    Returns:
        Path to the dataset root directory.
    """
    from huggingface_hub import snapshot_download

    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    repo_path = data_dir / "hls_burn_scars"
    if not repo_path.exists():
        snapshot_download(
            repo_id="ibm-nasa-geospatial/hls_burn_scars",
            repo_type="dataset",
            local_dir=str(repo_path),
        )
        print(f"Downloaded dataset to {repo_path}", flush=True)
    else:
        print(f"Dataset already exists at {repo_path}", flush=True)

    return repo_path


def load_splits(
    dataset_dir: str | Path, max_chips: Optional[int] = None
) -> dict[str, list[str]]:
    """Load train/val/test split file listings.

    Args:
        dataset_dir: Path to the hls_burn_scars directory.
        max_chips: If set, subsample each split proportionally.

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to lists of chip basenames.
    """
    dataset_dir = Path(dataset_dir)
    splits_dir = dataset_dir / "splits"

    splits = {}
    for split_name in ("train", "val", "test"):
        split_file = splits_dir / f"{split_name}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        chips = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        splits[split_name] = chips

    if max_chips is not None:
        total = sum(len(v) for v in splits.values())
        if max_chips < total:
            ratio = max_chips / total
            for key in splits:
                n = max(1, int(len(splits[key]) * ratio))
                random.seed(42)
                splits[key] = random.sample(splits[key], n)

    return splits


def verify_chip(image_path: str | Path, mask_path: str | Path) -> dict:
    """Verify a single image/mask pair has expected structure.

    Args:
        image_path: Path to the 6-band image GeoTIFF.
        mask_path: Path to the mask GeoTIFF.

    Returns:
        Dict with shape, dtype, and mask value info.
    """
    image_path, mask_path = Path(image_path), Path(mask_path)

    with rasterio.open(image_path) as src:
        img_shape = (src.count, src.height, src.width)
        img_dtype = src.dtypes[0]

    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        mask_values = set(np.unique(mask_data).tolist())

    return {
        "image_shape": img_shape,
        "image_dtype": img_dtype,
        "mask_values": mask_values,
        "has_burned": 1 in mask_values,
    }


def get_chip_paths(
    dataset_dir: str | Path, chip_name: str
) -> tuple[Path, Path]:
    """Get image and mask paths for a chip basename.

    Args:
        dataset_dir: Path to the hls_burn_scars directory.
        chip_name: Chip basename (without extension).

    Returns:
        Tuple of (image_path, mask_path).
    """
    dataset_dir = Path(dataset_dir)
    data_dir = dataset_dir / "data"
    image_path = data_dir / f"{chip_name}_merged.tif"
    mask_path = data_dir / f"{chip_name}.mask.tif"
    return image_path, mask_path


def verify_dataset(dataset_dir: str | Path, splits: dict[str, list[str]]) -> dict:
    """Verify the full dataset structure.

    Args:
        dataset_dir: Path to the hls_burn_scars directory.
        splits: Split dict from load_splits().

    Returns:
        Summary dict with counts and any issues found.
    """
    dataset_dir = Path(dataset_dir)
    total_chips = sum(len(v) for v in splits.values())
    issues = []
    checked = 0

    # Spot-check first 5 chips from each split
    for split_name, chips in splits.items():
        for chip_name in chips[:5]:
            img_path, mask_path = get_chip_paths(dataset_dir, chip_name)
            if not img_path.exists():
                issues.append(f"Missing image: {img_path}")
                continue
            if not mask_path.exists():
                issues.append(f"Missing mask: {mask_path}")
                continue
            info = verify_chip(img_path, mask_path)
            if info["image_shape"] != (6, 512, 512):
                issues.append(f"{chip_name}: unexpected shape {info['image_shape']}")
            checked += 1

    return {
        "total_chips": total_chips,
        "splits": {k: len(v) for k, v in splits.items()},
        "chips_verified": checked,
        "issues": issues,
    }
