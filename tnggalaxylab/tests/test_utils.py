"""Tests for utility helpers."""

from __future__ import annotations

import numpy as np

from tnggalaxylab.utils.coordinates import cartesian_to_polar, rotate_positions
from tnggalaxylab.utils.fft_utils import azimuthal_fft, density_histogram2d, polar_resample
from tnggalaxylab.utils.interpolation import monotonic_interpolator
from tnggalaxylab.utils.statistics import bootstrap_ci, weighted_percentile


def test_coordinate_transforms() -> None:
    radius, theta = cartesian_to_polar([1.0, 0.0], [0.0, 1.0])
    assert np.allclose(radius, [1.0, 1.0])
    assert np.allclose(theta, [0.0, np.pi / 2.0])
    rotated = rotate_positions([[1.0, 0.0, 0.0]], np.eye(3))
    assert np.allclose(rotated, [[1.0, 0.0, 0.0]])


def test_statistics() -> None:
    assert np.allclose(weighted_percentile([0.0, 10.0], [50.0], [1.0, 1.0]), [5.0])
    central, lower, upper = bootstrap_ci([1.0, 2.0, 3.0], n_bootstrap=8, seed=1)
    assert lower <= central <= upper


def test_fft_helpers() -> None:
    x = np.array([-1.0, 1.0, -1.0, 1.0])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    density, xc, yc = density_histogram2d(x, y, bins=16, extent=2.0)
    polar, radius, _ = polar_resample(density, xc, yc, n_radial=8, n_azimuth=32)
    coeff = azimuthal_fft(polar, max_mode=4)
    assert density.shape == (16, 16)
    assert polar.shape == (8, 32)
    assert radius.size == 8
    assert coeff.shape == (5, 8)


def test_interpolation() -> None:
    interp = monotonic_interpolator([2.0, 0.0, 1.0], [4.0, 0.0, 1.0])
    assert np.isclose(interp(1.5), 2.5)
