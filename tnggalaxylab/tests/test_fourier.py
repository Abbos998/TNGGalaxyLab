"""Tests for the Fourier workflow."""

from __future__ import annotations

import numpy as np

from tnggalaxylab.analysis.fourier_fft import FourierAnalyzer


def test_fourier_workflow_exports(tmp_path) -> None:
    rng = np.random.default_rng(5)
    n = 4000
    radius = rng.gamma(2.0, 1.0, n)
    theta = rng.uniform(-np.pi, np.pi, n)
    theta[radius < 2.0] = rng.normal(0.2, 0.2, np.count_nonzero(radius < 2.0))
    positions = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros(n)])
    analyzer = FourierAnalyzer(positions, max_mode=8, output_dir=tmp_path, label="test")
    report = analyzer.report(bins=64, n_radial=32, n_azimuth=64, make_plots=False)
    assert "A2_A0_global" in report["global_modes"]
    assert report["diagnostics"]["bar_length"] >= 0.0
    assert report["csv"].exists()
    assert report["npz"].exists()
