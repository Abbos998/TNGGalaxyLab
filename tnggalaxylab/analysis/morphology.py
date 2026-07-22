"""Non-parametric galaxy morphology diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter, rotate


class MorphologyAnalyzer:
    """Measure CAS, Gini, and M20 diagnostics from a 2D image."""

    def __init__(self, image: ArrayLike) -> None:
        """Initialize with a non-negative surface-brightness image."""

        self.image = np.asarray(image, dtype=np.float64)
        if self.image.ndim != 2:
            raise ValueError("image must be two-dimensional")
        self.image = np.nan_to_num(self.image, nan=0.0, posinf=0.0, neginf=0.0)

    def asymmetry(self) -> float:
        """Compute 180-degree rotational asymmetry."""

        rotated = rotate(self.image, 180.0, reshape=False, order=1)
        denom = np.sum(np.abs(self.image))
        if denom == 0.0:
            return 0.0
        return float(np.sum(np.abs(self.image - rotated)) / denom)

    def concentration(
        self,
        center: tuple[float, float] | None = None,
    ) -> float:
        """Compute ``5 log10(r80 / r20)`` concentration."""

        y, x = np.indices(self.image.shape)
        if center is None:
            total = np.sum(self.image)
            if total <= 0.0:
                return 0.0
            center = (float(np.sum(x * self.image) / total), float(np.sum(y * self.image) / total))
        radius = np.hypot(x - center[0], y - center[1]).ravel()
        flux = np.clip(self.image.ravel(), 0.0, None)
        order = np.argsort(radius)
        cumulative = np.cumsum(flux[order])
        if cumulative[-1] <= 0.0:
            return 0.0
        r20 = np.interp(0.2 * cumulative[-1], cumulative, radius[order])
        r80 = np.interp(0.8 * cumulative[-1], cumulative, radius[order])
        if r20 <= 0.0:
            return 0.0
        return float(5.0 * np.log10(r80 / r20))

    def smoothness(self, sigma: float = 2.0) -> float:
        """Compute clumpiness via residual from a smoothed image."""

        smoothed = gaussian_filter(self.image, sigma=sigma)
        denom = np.sum(np.abs(self.image))
        if denom == 0.0:
            return 0.0
        return float(np.sum(np.abs(self.image - smoothed)) / denom)

    def gini(self) -> float:
        """Compute the Gini coefficient."""

        values = np.sort(np.clip(self.image.ravel(), 0.0, None))
        n = values.size
        total = np.sum(values)
        if n == 0 or total == 0.0:
            return 0.0
        index = np.arange(1, n + 1)
        return float(np.sum((2 * index - n - 1) * values) / (n * total))

    def m20(self) -> float:
        """Compute the normalized second moment of the brightest 20 percent."""

        flux = np.clip(self.image, 0.0, None)
        total_flux = np.sum(flux)
        if total_flux <= 0.0:
            return 0.0
        y, x = np.indices(flux.shape)
        x_c = np.sum(x * flux) / total_flux
        y_c = np.sum(y * flux) / total_flux
        moment = flux * ((x - x_c) ** 2 + (y - y_c) ** 2)
        total_moment = np.sum(moment)
        if total_moment <= 0.0:
            return 0.0
        flat_flux = flux.ravel()
        flat_moment = moment.ravel()
        order = np.argsort(flat_flux)[::-1]
        cumulative = np.cumsum(flat_flux[order])
        selected = cumulative <= 0.2 * total_flux
        if not np.any(selected):
            selected[0] = True
        return float(np.log10(np.sum(flat_moment[order][selected]) / total_moment))

    def cas_parameters(self) -> dict[str, float]:
        """Return concentration, asymmetry, and smoothness."""

        return {
            "concentration": self.concentration(),
            "asymmetry": self.asymmetry(),
            "smoothness": self.smoothness(),
        }

    def summary(self) -> dict[str, float]:
        """Return all implemented morphology diagnostics."""

        result = self.cas_parameters()
        result["gini"] = self.gini()
        result["m20"] = self.m20()
        return result
