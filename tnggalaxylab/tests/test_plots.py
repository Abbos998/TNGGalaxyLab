"""Tests for plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt

from tnggalaxylab.plots.style import apply_publication_style, save_figure


def test_save_figure(tmp_path) -> None:
    apply_publication_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    output = save_figure(fig, tmp_path / "figure.png")
    assert output.exists()
