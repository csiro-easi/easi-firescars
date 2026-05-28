"""Visualisation utilities for burn scar training and evaluation."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(history: list, title: str = "Training Progress"):
    """Plot loss and IoU curves from training history."""
    epochs = range(1, len(history) + 1)
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    val_iou = [h["val"]["iou"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_loss, label="Train")
    ax1.plot(epochs, val_loss, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss")

    ax2.plot(epochs, val_iou, label="Val IoU", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.legend()
    ax2.set_title("Validation IoU")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_prediction_panels(image: np.ndarray, mask_true: np.ndarray, mask_pred: np.ndarray, sample_id: str = ""):
    """Show 4-panel: S2 RGB, S1 VV, ground truth, prediction."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # S2 RGB (bands 2,1,0 = R,G,B)
    rgb = np.clip(image[[2, 1, 0]].transpose(1, 2, 0) * 3, 0, 1)
    axes[0].imshow(rgb)
    axes[0].set_title("S2 RGB")

    # S1 VV (band 6)
    axes[1].imshow(image[6], cmap="gray")
    axes[1].set_title("S1 VV")

    # Ground truth mask
    axes[2].imshow(mask_true, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth")

    # Prediction
    axes[3].imshow(mask_pred, cmap="Reds", vmin=0, vmax=1)
    axes[3].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Sample: {sample_id}" if sample_id else "Prediction")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def display_sample(image: np.ndarray, mask: np.ndarray, sample_id: str = ""):
    """Display a single dataset sample (S2 RGB + S1 VV + mask)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    rgb = np.clip(image[[2, 1, 0]].transpose(1, 2, 0) * 3, 0, 1)
    axes[0].imshow(rgb)
    axes[0].set_title("S2 RGB")

    axes[1].imshow(image[6], cmap="gray")
    axes[1].set_title("S1 VV")

    axes[2].imshow(mask, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Burn Scar Mask")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(sample_id or "Sample")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
