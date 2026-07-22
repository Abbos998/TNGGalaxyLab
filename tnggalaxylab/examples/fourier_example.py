"""Synthetic example for the Fourier analysis workflow."""

from __future__ import annotations

import numpy as np

from tnggalaxylab.analysis.fourier_fft import FourierAnalyzer


def make_barred_disk(n_particles: int = 100_000, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple barred stellar disk for demonstration."""

    rng = np.random.default_rng(seed)
    radius = rng.gamma(shape=2.0, scale=2.0, size=n_particles)
    theta = rng.uniform(-np.pi, np.pi, size=n_particles)
    bar_mask = radius < 4.0
    theta[bar_mask] = rng.normal(0.4, 0.25, size=np.count_nonzero(bar_mask))
    z = rng.normal(0.0, 0.15, size=n_particles)
    positions = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])
    masses = rng.lognormal(mean=0.0, sigma=0.1, size=n_particles)
    return positions, masses


def main() -> None:
    """Run the synthetic example."""

    positions, masses = make_barred_disk()
    analyzer = FourierAnalyzer(positions, masses, label="synthetic_bar")
    report = analyzer.report(bins=192, n_radial=96, n_azimuth=192)
    print(report["diagnostics"])


if __name__ == "__main__":
    main()
