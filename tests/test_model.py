"""Tests for firescar model, loss, and dataset."""

import tempfile

import numpy as np
import torch
import zarr

from firescars.dataset import FirescarDataset
from firescars.evaluate import compute_metrics
from firescars.loss import FirescarLoss, dice_loss
from firescars.model import FirescarModel


def test_model_forward_shape():
    model = FirescarModel(in_chans=6, img_size=224, pretrained_encoder=False)
    x = torch.randn(2, 6, 224, 224)
    out = model(x)
    assert out.shape == (2, 224, 224)


def test_model_param_groups():
    model = FirescarModel(in_chans=6, img_size=224, pretrained_encoder=False)
    groups = model.get_param_groups(encoder_lr_mult=0.1)
    assert len(groups) == 2
    assert groups[0]["lr_mult"] == 0.1
    assert groups[1]["lr_mult"] == 1.0
    assert len(groups[0]["params"]) > 0
    assert len(groups[1]["params"]) > 0


def test_loss_output():
    criterion = FirescarLoss(bce_weight=0.5, dice_weight=0.5)
    logits = torch.randn(4, 224, 224)
    targets = torch.randint(0, 2, (4, 224, 224)).float()
    loss = criterion(logits, targets)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_dice_loss_perfect():
    logits = torch.ones(2, 10, 10) * 10  # high confidence positive
    targets = torch.ones(2, 10, 10)
    loss = dice_loss(logits, targets)
    assert loss.item() < 0.01


def test_compute_metrics():
    logits = torch.ones(2, 10, 10) * 10
    targets = torch.ones(2, 10, 10)
    metrics = compute_metrics(logits, targets)
    assert metrics["iou"] > 0.99
    assert metrics["f1"] > 0.99
    assert metrics["precision"] > 0.99
    assert metrics["recall"] > 0.99


def test_dataset_loading():
    """Test dataset with a temporary Zarr store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = zarr.open(tmpdir, mode="w")
        grp = store.create_group("train")
        n = 8
        grp.create_array(
            "imagery", data=np.random.randint(0, 3000, (n, 6, 224, 224), dtype=np.int16)
        )
        grp.create_array("masks", data=np.random.randint(0, 2, (n, 224, 224), dtype=np.uint8))

        ds = FirescarDataset(tmpdir, split="train", augment=True)
        assert len(ds) == n

        img, mask = ds[0]
        assert img.shape == (6, 224, 224)
        assert mask.shape == (224, 224)
        assert img.dtype == torch.float32
        assert mask.dtype == torch.float32
        assert img.min() >= 0.0
        assert img.max() <= 1.0
