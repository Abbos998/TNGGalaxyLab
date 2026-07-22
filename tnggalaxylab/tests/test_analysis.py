"""Tests for scientific analysis modules."""

from __future__ import annotations

import numpy as np

from tnggalaxylab.analysis.bar import BarDiagnostics
from tnggalaxylab.analysis.morphology import MorphologyAnalyzer
from tnggalaxylab.analysis.radial_profile import RadialProfileAnalyzer
from tnggalaxylab.analysis.rotation_curve import RotationCurveAnalyzer


def test_radial_profile_and_fit() -> None:
    rng = np.random.default_rng(0)
    radius = rng.exponential(2.0, 5000)
    theta = rng.uniform(-np.pi, np.pi, radius.size)
    positions = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(radius)])
    analyzer = RadialProfileAnalyzer(positions)
    profile = analyzer.surface_density(bins=20, r_max=8.0)
    assert profile.table.shape[0] == 20
    scale = analyzer.disk_scale_length(radius=profile.radius, surface_density=profile.surface_density)
    assert scale > 0.0


def test_rotation_curve() -> None:
    positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    masses = np.array([1.0e10, 1.0e10])
    table = RotationCurveAnalyzer(
        stellar_positions=positions,
        stellar_masses=masses,
        dark_matter_positions=positions,
        dark_matter_masses=masses,
    ).dark_matter_fraction([1.0, 2.0])
    assert np.allclose(table["dark_matter_fraction"], [0.5, 0.5])


def test_morphology() -> None:
    image = np.zeros((32, 32))
    image[14:18, 14:18] = 1.0
    summary = MorphologyAnalyzer(image).summary()
    assert summary["concentration"] > 0.0
    assert summary["gini"] > 0.0


def test_bar_diagnostics() -> None:
    radius = np.linspace(0.0, 10.0, 20)
    amp = np.exp(-0.5 * ((radius - 4.0) / 1.0) ** 2)
    phase = np.full_like(radius, 0.8)
    diagnostic = BarDiagnostics(radius, amp, phase)
    assert diagnostic.bar_length() >= radius[np.argmax(amp)]
    assert np.isclose(diagnostic.bar_angle(), 0.4)
    assert np.isclose(BarDiagnostics.pattern_speed([0.0, 1.0], [0.0, 2.0]), 0.5)
