"""FFT-based stellar Fourier mode analysis.

This module implements the full workflow requested for bar, lopsidedness, and
azimuthal structure measurements:

particles -> 2D stellar density map -> Gaussian smoothing -> polar transform
-> azimuthal FFT -> complex Fourier coefficients -> amplitudes/phases ->
global modes -> bar diagnostics -> CSV/NPZ export -> publication figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from numpy.typing import ArrayLike, NDArray

from tnggalaxylab.analysis.bar import BarDiagnostics
from tnggalaxylab.plots.style import apply_publication_style, save_figure
from tnggalaxylab.utils.fft_utils import azimuthal_fft, density_histogram2d, polar_resample, smooth_density


@njit(cache=True)
def _global_amplitudes_numba(
    amplitudes: NDArray[np.float64],
    a0: NDArray[np.float64],
    radius: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Integrate radial amplitudes with A0 weighting."""

    n_modes = amplitudes.shape[0]
    result = np.zeros(n_modes, dtype=np.float64)
    denominator = 0.0
    for i in range(radius.size - 1):
        dr = radius[i + 1] - radius[i]
        weight = 0.5 * (a0[i + 1] + a0[i]) * dr
        denominator += weight
        for mode in range(n_modes):
            result[mode] += 0.5 * (amplitudes[mode, i + 1] + amplitudes[mode, i]) * weight
    if denominator > 0.0:
        for mode in range(n_modes):
            result[mode] /= denominator
    return result


@dataclass(slots=True)
class FourierProducts:
    """Intermediate and final Fourier analysis products."""

    density: NDArray[np.float64] | None = None
    smoothed_density: NDArray[np.float64] | None = None
    x_centers: NDArray[np.float64] | None = None
    y_centers: NDArray[np.float64] | None = None
    polar_density: NDArray[np.float64] | None = None
    radial_grid: NDArray[np.float64] | None = None
    azimuth_grid: NDArray[np.float64] | None = None
    coefficients: NDArray[np.complex128] | None = None
    amplitudes: NDArray[np.float64] | None = None
    normalized_amplitudes: NDArray[np.float64] | None = None
    phases: NDArray[np.float64] | None = None
    global_modes: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)


class FourierAnalyzer:
    """Analyze stellar Fourier modes from particle data."""

    def __init__(
        self,
        positions: ArrayLike,
        masses: ArrayLike | None = None,
        max_mode: int = 8,
        output_dir: str | Path = "TNGGalaxyLab/output",
        label: str = "galaxy",
    ) -> None:
        """Initialize the Fourier analyzer.

        Args:
            positions: Stellar coordinates with shape ``(N, 3)``. The x-y
                plane is treated as the analysis plane.
            masses: Optional stellar masses or luminosity weights.
            max_mode: Highest Fourier mode to measure. Defaults to A0-A8.
            output_dir: Directory used by plotting and export methods.
            label: Name prefix for output products.
        """

        self.positions = np.asarray(positions, dtype=np.float64)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        self.masses = (
            np.ones(self.positions.shape[0], dtype=np.float64)
            if masses is None
            else np.asarray(masses, dtype=np.float64)
        )
        if self.masses.shape != (self.positions.shape[0],):
            raise ValueError("masses must have shape (N,)")
        if max_mode < 1:
            raise ValueError("max_mode must be at least 1")
        self.max_mode = int(max_mode)
        self.output_dir = Path(output_dir)
        self.label = label
        self.products = FourierProducts()

    def make_density_map(
        self,
        bins: int = 256,
        extent: float | None = None,
    ) -> NDArray[np.float64]:
        """Create a projected 2D stellar surface-density map.

        Args:
            bins: Number of Cartesian pixels per side.
            extent: Optional half-width of the map.

        Returns:
            Surface-density map.
        """

        density, x_centers, y_centers = density_histogram2d(
            self.positions[:, 0],
            self.positions[:, 1],
            weights=self.masses,
            bins=bins,
            extent=extent,
        )
        self.products.density = density
        self.products.x_centers = x_centers
        self.products.y_centers = y_centers
        return density

    def gaussian_smooth(self, sigma: float = 1.5) -> NDArray[np.float64]:
        """Gaussian-smooth the density map."""

        if self.products.density is None:
            self.make_density_map()
        assert self.products.density is not None
        self.products.smoothed_density = smooth_density(self.products.density, sigma=sigma)
        return self.products.smoothed_density

    def polar_transform(
        self,
        n_radial: int = 128,
        n_azimuth: int = 256,
        r_max: float | None = None,
        use_smoothed: bool = True,
    ) -> NDArray[np.float64]:
        """Transform the density map to polar coordinates."""

        image = self.products.smoothed_density if use_smoothed else self.products.density
        if image is None:
            image = self.gaussian_smooth()
        if self.products.x_centers is None or self.products.y_centers is None:
            raise RuntimeError("density map coordinate grids are missing")
        polar, radial, azimuth = polar_resample(
            image,
            self.products.x_centers,
            self.products.y_centers,
            n_radial=n_radial,
            n_azimuth=n_azimuth,
            r_max=r_max,
        )
        self.products.polar_density = polar
        self.products.radial_grid = radial
        self.products.azimuth_grid = azimuth
        return polar

    def compute_fft(self) -> NDArray[np.complex128]:
        """Compute complex Fourier coefficients along azimuth."""

        if self.products.polar_density is None:
            self.polar_transform()
        assert self.products.polar_density is not None
        self.products.coefficients = azimuthal_fft(self.products.polar_density, self.max_mode)
        return self.products.coefficients

    def compute_modes(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute A0-Amax amplitudes and amplitudes normalized by A0."""

        if self.products.coefficients is None:
            self.compute_fft()
        assert self.products.coefficients is not None
        amplitudes = np.abs(self.products.coefficients)
        a0 = amplitudes[0]
        normalized = np.divide(
            amplitudes,
            a0[None, :],
            out=np.zeros_like(amplitudes),
            where=a0[None, :] > 0.0,
        )
        self.products.amplitudes = amplitudes
        self.products.normalized_amplitudes = normalized
        return amplitudes, normalized

    def compute_phase(self) -> NDArray[np.float64]:
        """Compute Fourier phases for all modes."""

        if self.products.coefficients is None:
            self.compute_fft()
        assert self.products.coefficients is not None
        phases = np.angle(self.products.coefficients)
        phases[0, :] = 0.0
        self.products.phases = phases
        return phases

    def compute_global_modes(
        self,
        radial_range: tuple[float, float] | None = None,
    ) -> dict[str, float]:
        """Compute A1/A0 ... Amax/A0 global amplitudes.

        Args:
            radial_range: Optional radial integration limits.

        Returns:
            Dictionary keyed as ``A1_A0_global`` etc.
        """

        if self.products.normalized_amplitudes is None or self.products.amplitudes is None:
            self.compute_modes()
        if self.products.radial_grid is None:
            raise RuntimeError("radial grid is missing")
        assert self.products.normalized_amplitudes is not None
        assert self.products.amplitudes is not None
        radius = self.products.radial_grid
        mask = np.ones(radius.size, dtype=bool)
        if radial_range is not None:
            mask = (radius >= radial_range[0]) & (radius <= radial_range[1])
        if np.count_nonzero(mask) < 2:
            raise ValueError("radial_range leaves fewer than two bins")
        global_values = _global_amplitudes_numba(
            self.products.normalized_amplitudes[:, mask],
            self.products.amplitudes[0, mask],
            radius[mask],
        )
        self.products.global_modes = {
            f"A{mode}_A0_global": float(global_values[mode])
            for mode in range(1, self.max_mode + 1)
        }
        return self.products.global_modes

    def compute_bar_angle(self) -> float:
        """Estimate bar angle from the peak m=2 phase."""

        self._require_mode(2)
        assert self.products.normalized_amplitudes is not None
        assert self.products.phases is not None
        assert self.products.radial_grid is not None
        diagnostic = BarDiagnostics(
            self.products.radial_grid,
            self.products.normalized_amplitudes[2],
            self.products.phases[2],
        )
        angle = diagnostic.bar_angle()
        self.products.diagnostics["bar_angle_rad"] = angle
        return angle

    def compute_bar_length(
        self,
        amplitude_fraction: float = 0.5,
        phase_threshold: float = np.deg2rad(10.0),
    ) -> float:
        """Estimate bar length from A2/A0 and m=2 phase stability."""

        self._require_mode(2)
        assert self.products.normalized_amplitudes is not None
        assert self.products.phases is not None
        assert self.products.radial_grid is not None
        diagnostic = BarDiagnostics(
            self.products.radial_grid,
            self.products.normalized_amplitudes[2],
            self.products.phases[2],
        )
        length = diagnostic.bar_length(amplitude_fraction, phase_threshold)
        self.products.diagnostics["bar_length"] = length
        return length

    def compute_lopsidedness(
        self,
        radial_range: tuple[float, float] | None = None,
    ) -> float:
        """Return global lopsidedness, defined as global A1/A0."""

        globals_ = self.compute_global_modes(radial_range=radial_range)
        value = float(globals_.get("A1_A0_global", 0.0))
        self.products.diagnostics["lopsidedness"] = value
        return value

    def plot_density(self, path: str | Path | None = None) -> Path:
        """Save a publication-quality density map figure."""

        if self.products.density is None:
            self.make_density_map()
        assert self.products.density is not None
        assert self.products.x_centers is not None
        assert self.products.y_centers is not None
        apply_publication_style()
        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        extent = [
            self.products.x_centers[0],
            self.products.x_centers[-1],
            self.products.y_centers[0],
            self.products.y_centers[-1],
        ]
        image = ax.imshow(
            np.log10(self.products.density + 1.0e-12),
            origin="lower",
            extent=extent,
            cmap="magma",
            aspect="equal",
        )
        fig.colorbar(image, ax=ax, label=r"$\log_{10}\,\Sigma_\star$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Stellar surface density")
        return save_figure(fig, self._path(path, "density.png"))

    def plot_polar(self, path: str | Path | None = None) -> Path:
        """Save a polar density map figure."""

        if self.products.polar_density is None:
            self.polar_transform()
        assert self.products.polar_density is not None
        assert self.products.radial_grid is not None
        assert self.products.azimuth_grid is not None
        apply_publication_style()
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        image = ax.imshow(
            np.log10(self.products.polar_density + 1.0e-12),
            origin="lower",
            aspect="auto",
            extent=[
                self.products.azimuth_grid[0],
                self.products.azimuth_grid[-1],
                self.products.radial_grid[0],
                self.products.radial_grid[-1],
            ],
            cmap="viridis",
        )
        fig.colorbar(image, ax=ax, label=r"$\log_{10}\,\Sigma_\star$")
        ax.set_xlabel(r"$\phi$ [rad]")
        ax.set_ylabel("R")
        ax.set_title("Polar surface density")
        return save_figure(fig, self._path(path, "polar.png"))

    def plot_modes(self, path: str | Path | None = None) -> Path:
        """Save radial Fourier amplitude profiles."""

        if self.products.normalized_amplitudes is None:
            self.compute_modes()
        assert self.products.normalized_amplitudes is not None
        assert self.products.radial_grid is not None
        apply_publication_style()
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for mode in range(1, self.max_mode + 1):
            ax.plot(
                self.products.radial_grid,
                self.products.normalized_amplitudes[mode],
                label=f"A{mode}/A0",
                lw=1.5,
            )
        ax.set_xlabel("R")
        ax.set_ylabel("Mode amplitude")
        ax.set_title("Fourier amplitudes")
        ax.legend(ncol=2, fontsize=8)
        return save_figure(fig, self._path(path, "modes.png"))

    def plot_phase(self, path: str | Path | None = None) -> Path:
        """Save Fourier phase profiles."""

        if self.products.phases is None:
            self.compute_phase()
        assert self.products.phases is not None
        assert self.products.radial_grid is not None
        apply_publication_style()
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for mode in range(1, self.max_mode + 1):
            ax.plot(self.products.radial_grid, np.unwrap(self.products.phases[mode]), label=f"m={mode}", lw=1.2)
        ax.set_xlabel("R")
        ax.set_ylabel("Phase [rad]")
        ax.set_title("Fourier phases")
        ax.legend(ncol=2, fontsize=8)
        return save_figure(fig, self._path(path, "phase.png"))

    def plot_bar(self, path: str | Path | None = None) -> Path:
        """Save a density map annotated with bar angle and length."""

        angle = self.compute_bar_angle()
        length = self.compute_bar_length()
        if self.products.density is None:
            self.make_density_map()
        assert self.products.density is not None
        assert self.products.x_centers is not None
        assert self.products.y_centers is not None
        apply_publication_style()
        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        extent = [
            self.products.x_centers[0],
            self.products.x_centers[-1],
            self.products.y_centers[0],
            self.products.y_centers[-1],
        ]
        ax.imshow(
            np.log10(self.products.density + 1.0e-12),
            origin="lower",
            extent=extent,
            cmap="magma",
            aspect="equal",
        )
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        ax.plot([-dx, dx], [-dy, dy], color="cyan", lw=2.0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Bar diagnostic")
        return save_figure(fig, self._path(path, "bar.png"))

    def save_csv(self, path: str | Path | None = None) -> Path:
        """Export radial Fourier profiles and diagnostics to CSV."""

        if self.products.normalized_amplitudes is None:
            self.compute_modes()
        if self.products.phases is None:
            self.compute_phase()
        assert self.products.radial_grid is not None
        assert self.products.amplitudes is not None
        assert self.products.normalized_amplitudes is not None
        assert self.products.phases is not None
        data: dict[str, Any] = {"radius": self.products.radial_grid}
        for mode in range(self.max_mode + 1):
            data[f"A{mode}"] = self.products.amplitudes[mode]
            data[f"A{mode}_A0"] = self.products.normalized_amplitudes[mode]
            data[f"phase{mode}"] = self.products.phases[mode]
        table = pd.DataFrame(data)
        output = self._path(path, "fourier_profiles.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
        return output

    def save_npz(self, path: str | Path | None = None) -> Path:
        """Export Fourier products to compressed NPZ."""

        if self.products.coefficients is None:
            self.compute_fft()
        output = self._path(path, "fourier_products.npz")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            density=self.products.density,
            smoothed_density=self.products.smoothed_density,
            x_centers=self.products.x_centers,
            y_centers=self.products.y_centers,
            polar_density=self.products.polar_density,
            radial_grid=self.products.radial_grid,
            azimuth_grid=self.products.azimuth_grid,
            coefficients=self.products.coefficients,
            amplitudes=self.products.amplitudes,
            normalized_amplitudes=self.products.normalized_amplitudes,
            phases=self.products.phases,
        )
        return output

    def report(
        self,
        bins: int = 256,
        smooth_sigma: float = 1.5,
        n_radial: int = 128,
        n_azimuth: int = 256,
        make_plots: bool = True,
    ) -> dict[str, Any]:
        """Run the full Fourier workflow and write standard products.

        Args:
            bins: Cartesian density-map pixel count per side.
            smooth_sigma: Gaussian smoothing sigma in pixels.
            n_radial: Number of radial polar bins.
            n_azimuth: Number of azimuth polar bins.
            make_plots: Whether to save PNG figures.

        Returns:
            Dictionary containing diagnostics and output paths.
        """

        self.make_density_map(bins=bins)
        self.gaussian_smooth(sigma=smooth_sigma)
        self.polar_transform(n_radial=n_radial, n_azimuth=n_azimuth)
        self.compute_fft()
        self.compute_modes()
        self.compute_phase()
        global_modes = self.compute_global_modes()
        diagnostics = {
            "bar_angle_rad": self.compute_bar_angle(),
            "bar_length": self.compute_bar_length(),
            "lopsidedness": self.compute_lopsidedness(),
        }
        csv_path = self.save_csv()
        npz_path = self.save_npz()
        plot_paths: list[Path] = []
        if make_plots:
            plot_paths = [
                self.plot_density(),
                self.plot_polar(),
                self.plot_modes(),
                self.plot_phase(),
                self.plot_bar(),
            ]
        return {
            "label": self.label,
            "global_modes": global_modes,
            "diagnostics": diagnostics,
            "csv": csv_path,
            "npz": npz_path,
            "plots": plot_paths,
        }

    def _require_mode(self, mode: int) -> None:
        if self.max_mode < mode:
            raise ValueError(f"max_mode must be at least {mode}")
        if self.products.normalized_amplitudes is None:
            self.compute_modes()
        if self.products.phases is None:
            self.compute_phase()

    def _path(self, path: str | Path | None, suffix: str) -> Path:
        if path is not None:
            return Path(path)
        return self.output_dir / f"{self.label}_{suffix}"
