"""Unit tests for modules/visualisation.py"""

import numpy as np
import pytest

from modules.visualisation import plot_false_colour, plot_sample_chips, plot_prediction_comparison


@pytest.fixture
def sample_image():
    return np.random.rand(6, 64, 64).astype(np.float32)


@pytest.fixture
def sample_mask():
    mask = np.zeros((64, 64), dtype=np.int8)
    mask[20:40, 20:40] = 1
    return mask


def test_plot_false_colour_no_error(sample_image, sample_mask, monkeypatch):
    """Verify plot_false_colour runs without error."""
    import matplotlib
    monkeypatch.setattr(matplotlib, "use", lambda x: None)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_false_colour(sample_image, sample_mask, title="Test")


def test_plot_false_colour_no_mask(sample_image, monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_false_colour(sample_image, mask=None, title="No mask")


def test_plot_sample_chips(sample_image, sample_mask, monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    images = [sample_image] * 3
    masks = [sample_mask] * 3
    plot_sample_chips(images, masks, max_display=3)


def test_plot_prediction_comparison(sample_image, sample_mask, monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_prediction_comparison(sample_image, sample_mask, reference=sample_mask, title="Compare")
