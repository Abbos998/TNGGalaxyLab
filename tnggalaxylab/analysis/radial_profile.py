"""Radial surface-density and bulge/disk profile analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit

from tnggalaxylab.utils.coordinates import cylindrical_radius


def exponential_profile(radius: NDArray[np.float64], sigma0: float, scale_length: float) -> NDArray[np.float64]:
    """Exponential disk surface-density model."""

    return sigma0 * np.exp(-radius / scale_length)


def sersic_profile(
    radius: NDArray[np.float64],
    sigma_e: float,
    r_e: float,
    n: float,
) -> NDArray[np.float64]:
    """Sersic bulge surface-density model."""

    b_n = 2.0 * n - 1.0 / 3.0
    return sigma_e * np.exp(-b_n * ((radius / r_e) ** (1.0 / n) - 1.0))


@dataclass(slots=True)
class RadialProfileResult:
    """Container for radial profile measurements."""

    radius: NDArray[np.float64]
    surface_density: NDArray[np.float64]
    counts: NDArray[np.int64]
    table: pd.DataFrame


class RadialProfileAnalyzer:
    """Measure surface-density profiles and parametric disk/bulge fits."""

    def __init__(
        self,
        positions: ArrayLike,
        masses: ArrayLike | None = None,
    ) -> None:
        """Initialize with particle coordinates and optional masses."""

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

    def surface_density(
        self,
        bins: int | ArrayLike = 50,
        r_max: float | None = None,
    ) -> RadialProfileResult:
        """Compute annular surface density.

        Args:
            bins: Number of radial bins or explicit bin edges.
            r_max: Optional maximum radius if ``bins`` is an integer.

        Returns:
            RadialProfileResult with a pandas table.
        """

        radius = cylindrical_radius(self.positions)
        if np.isscalar(bins):
            maximum = float(np.nanmax(radius) if r_max is None else r_max)
            edges = np.linspace(0.0, maximum, int(bins) + 1)
        else:
            edges = np.asarray(bins, dtype=np.float64)
        mass_sum, _ = np.histogram(radius, bins=edges, weights=self.masses)
        counts, _ = np.histogram(radius, bins=edges)
        area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        sigma = np.divide(mass_sum, area, out=np.zeros_like(mass_sum), where=area > 0.0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        table = pd.DataFrame(
            {
                "radius": centers,
                "surface_density": sigma,
                "counts": counts.astype(np.int64),
            }
        )
        return RadialProfileResult(centers, sigma, counts.astype(np.int64), table)

    def exponential_fit(
        self,
        radius: ArrayLike | None = None,
        surface_density: ArrayLike | None = None,
        fit_range: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """Fit an exponential disk and return ``(sigma0, scale_length)``."""

        if radius is None or surface_density is None:
            profile = self.surface_density()
            r = profile.radius
            sigma = profile.surface_density
        else:
            r = np.asarray(radius, dtype=np.float64)
            sigma = np.asarray(surface_density, dtype=np.float64)
        mask = np.isfinite(r) & np.isfinite(sigma) & (sigma > 0.0)
        if fit_range is not None:
            mask &= (r >= fit_range[0]) & (r <= fit_range[1])
        if np.count_nonzero(mask) < 3:
            raise ValueError("not enough valid bins for exponential fit")
        popt, _ = curve_fit(
            exponential_profile,
            r[mask],
            sigma[mask],
            p0=(float(np.nanmax(sigma[mask])), float(np.nanmedian(r[mask]))),
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            maxfev=10000,
        )
        return float(popt[0]), float(popt[1])

    def disk_scale_length(self, **kwargs: object) -> float:
        """Return fitted exponential disk scale length."""

        return self.exponential_fit(**kwargs)[1]

    def bulge_profile(
        self,
        radius: ArrayLike | None = None,
        surface_density: ArrayLike | None = None,
        fit_range: tuple[float, float] | None = None,
    ) -> tuple[float, float, float]:
        """Fit a Sersic profile and return ``(sigma_e, r_e, n)``."""

        if radius is None or surface_density is None:
            profile = self.surface_density()
            r = profile.radius
            sigma = profile.surface_density
        else:
            r = np.asarray(radius, dtype=np.float64)
            sigma = np.asarray(surface_density, dtype=np.float64)
        mask = np.isfinite(r) & np.isfinite(sigma) & (sigma > 0.0) & (r > 0.0)
        if fit_range is not None:
            mask &= (r >= fit_range[0]) & (r <= fit_range[1])
        if np.count_nonzero(mask) < 4:
            raise ValueError("not enough valid bins for Sersic fit")
        popt, _ = curve_fit(
            sersic_profile,
            r[mask],
            sigma[mask],
            p0=(float(np.nanmedian(sigma[mask])), float(np.nanmedian(r[mask])), 2.0),
            bounds=([0.0, 1.0e-8, 0.2], [np.inf, np.inf, 8.0]),
            maxfev=20000,
        )
        return float(popt[0]), float(popt[1]), float(popt[2])
