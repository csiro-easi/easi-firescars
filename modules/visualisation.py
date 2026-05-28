"""Visualisation: false-colour composites, training curves, and comparison maps."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_false_colour(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    title: str = "",
    band_indices: tuple[int, int, int] = (4, 3, 2),
) -> None:
    """Display a false-colour composite (SWIR1/NIR/Red) with optional mask overlay.

    Args:
        image: Array of shape (6, H, W).
        mask: Optional mask (H, W) to display alongside.
        title: Plot title.
        band_indices: Indices for R, G, B channels in the composite.
    """
    import matplotlib.pyplot as plt

    rgb = np.stack([image[i] for i in band_indices], axis=-1)
    # Stretch to [0, 1] for display
    p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if rgb.max() > 0 else (0, 1)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-10), 0, 1)

    ncols = 2 if mask is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(rgb)
    axes[0].set_title(f"{title} (SWIR1/NIR/Red)")
    axes[0].axis("off")

    if mask is not None:
        axes[1].imshow(mask, cmap="RdYlGn_r", vmin=-1, vmax=1)
        axes[1].set_title("Burn Mask")
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_sample_chips(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    titles: Optional[list[str]] = None,
    max_display: int = 5,
) -> None:
    """Display a grid of sample chips with their masks.

    Args:
        images: List of (6, H, W) arrays.
        masks: List of (H, W) arrays.
        titles: Optional titles per chip.
        max_display: Maximum number of chips to show.
    """
    import matplotlib.pyplot as plt

    n = min(len(images), max_display)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        rgb = np.stack([images[i][4], images[i][3], images[i][2]], axis=-1)
        p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if rgb.max() > 0 else (0, 1)
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-10), 0, 1)

        axes[0, i].imshow(rgb)
        axes[0, i].axis("off")
        if titles:
            axes[0, i].set_title(titles[i][:20], fontsize=8)

        axes[1, i].imshow(masks[i], cmap="RdYlGn_r", vmin=-1, vmax=1)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("False colour")
    axes[1, 0].set_ylabel("Mask")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_training_curves(log_dir: str | Path) -> None:
    """Plot training loss and validation IoU from TerraTorch logs.

    Args:
        log_dir: Path to the lightning_logs/version_X/ directory.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    log_dir = Path(log_dir)
    metrics_file = log_dir / "metrics.csv"

    if not metrics_file.exists():
        print(f"No metrics file found at {metrics_file}", flush=True)
        return

    df = pd.read_csv(metrics_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if "train/loss" in df.columns:
        loss = df[["epoch", "train/loss"]].dropna()
        ax1.plot(loss["epoch"], loss["train/loss"])
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss")
        ax1.set_title("Training Loss")

    if "val/Multiclass_Jaccard_Index" in df.columns:
        iou = df[["epoch", "val/Multiclass_Jaccard_Index"]].dropna()
        ax2.plot(iou["epoch"], iou["val/Multiclass_Jaccard_Index"])
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation mIoU")
        ax2.set_title("Validation Mean IoU")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_prediction_comparison(
    image: np.ndarray,
    prediction: np.ndarray,
    reference: Optional[np.ndarray] = None,
    title: str = "",
) -> None:
    """Side-by-side comparison of input, prediction, and optional reference.

    Args:
        image: Input scene (6, H, W).
        prediction: Model prediction (H, W).
        reference: Optional ground truth (H, W).
        title: Overall title.
    """
    import matplotlib.pyplot as plt

    ncols = 3 if reference is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    # False colour
    rgb = np.stack([image[4], image[3], image[2]], axis=-1)
    p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if rgb.max() > 0 else (0, 1)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-10), 0, 1)

    axes[0].imshow(rgb)
    axes[0].set_title("False Colour (SWIR1/NIR/Red)")
    axes[0].axis("off")

    axes[1].imshow(prediction, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    if reference is not None:
        axes[2].imshow(reference, cmap="Reds", vmin=0, vmax=1)
        axes[2].set_title("Reference")
        axes[2].axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
