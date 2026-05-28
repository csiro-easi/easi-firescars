"""Tests for modules/model.py — model forward pass shape validation."""

import pytest
import torch


@pytest.fixture
def dummy_input():
    return torch.randn(2, 8, 224, 224)


def test_model_output_shape(dummy_input):
    """Model produces (B, 2, H, W) from (B, 8, H, W) input."""
    pytest.importorskip("terratorch")
    from modules.model import GraniteUKIBurnScar

    model = GraniteUKIBurnScar(num_classes=2, freeze_backbone=True)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 2, 224, 224)


def test_model_frozen_backbone_grads(dummy_input):
    """When backbone is frozen, only decoder params require grad."""
    pytest.importorskip("terratorch")
    from modules.model import GraniteUKIBurnScar

    model = GraniteUKIBurnScar(num_classes=2, freeze_backbone=True)
    backbone_params = sum(p.requires_grad for p in model.backbone.parameters())
    decoder_params = sum(p.requires_grad for p in model.decoder.parameters())
    assert backbone_params == 0
    assert decoder_params > 0
