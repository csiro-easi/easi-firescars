"""Inference: load fine-tuned model, preprocess scenes, predict, and save GeoTIFF."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from modules.training import BAND_MEANS, BAND_STDS


def normalise_scene(data: np.ndarray) -> np.ndarray:
    """Normalise a 6-band scene using Prithvi training statistics.

    Args:
        data: Array of shape (6, H, W) in raw reflectance values.

    Returns:
        Normalised array of same shape.
    """
    means = np.array(BAND_MEANS).reshape(6, 1, 1)
    stds = np.array(BAND_STDS).reshape(6, 1, 1)
    return (data - means) / stds


def load_scene(scene_path: str | Path) -> tuple[np.ndarray, dict]:
    """Load a 6-band HLS GeoTIFF scene.

    Args:
        scene_path: Path to the GeoTIFF file.

    Returns:
        Tuple of (data array shape (6, H, W), metadata dict with crs and transform).
    """
    with rasterio.open(scene_path) as src:
        data = src.read().astype(np.float32)
        meta = {"crs": src.crs, "transform": src.transform, "width": src.width, "height": src.height}
    return data, meta


def predict(model, data: np.ndarray) -> np.ndarray:
    """Run inference on a normalised scene.

    Args:
        model: Loaded PyTorch model in eval mode.
        data: Normalised array of shape (6, H, W).

    Returns:
        Binary prediction mask (H, W) with values {0, 1}.
    """
    import torch

    device = next(model.parameters()).device
    tensor = torch.from_numpy(data).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(tensor)
        if hasattr(logits, "output"):
            logits = logits.output
        elif isinstance(logits, dict):
            logits = logits["output"]
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    return pred.astype(np.int8)


def save_geotiff(
    mask: np.ndarray,
    output_path: str | Path,
    crs,
    transform,
) -> Path:
    """Save a prediction mask as a georeferenced GeoTIFF.

    Args:
        mask: Binary mask (H, W).
        output_path: Output file path.
        crs: Coordinate reference system.
        transform: Affine transform.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype="int8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask, 1)

    print(f"Saved prediction to {output_path}", flush=True)
    return output_path


def load_model(checkpoint_path: str | Path):
    """Load a fine-tuned model from a TerraTorch checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file.

    Returns:
        Model in eval mode on available device.
    """
    import torch
    from terratorch.tasks import SemanticSegmentationTask

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = SemanticSegmentationTask.load_from_checkpoint(str(checkpoint_path), map_location=device)
    task.eval()
    task.to(device)
    return task.model
