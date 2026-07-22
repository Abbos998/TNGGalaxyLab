"""Fourier and image-grid helper functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, map_coordinates


def density_histogram2d(
    x: ArrayLike,
    y: ArrayLike,
    weights: ArrayLike | None = None,
    bins: int = 256,
    extent: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build a square 2D surface-density map.

    Args:
        x: Projected x coordinates.
        y: Projected y coordinates.
        weights: Optional particle masses or luminosities.
        bins: Number of pixels per side.
        extent: Half-width of the square map. If omitted, a robust maximum
            projected radius is used.

    Returns:
        Density map, x bin centers, and y bin centers.
    """

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    if x_arr.size == 0:
        raise ValueError("at least one particle is required")
    if bins < 8:
        raise ValueError("bins must be at least 8")
    w_arr = None if weights is None else np.asarray(weights, dtype=np.float64)
    if w_arr is not None and w_arr.shape != x_arr.shape:
        raise ValueError("weights must match x and y")

    if extent is None:
        radius = np.hypot(x_arr, y_arr)
        extent = float(np.nanpercentile(radius, 99.5))
        if not np.isfinite(extent) or extent <= 0.0:
            extent = float(np.nanmax(np.abs(np.r_[x_arr, y_arr])))
    if extent <= 0.0:
        raise ValueError("extent must be positive")

    hist, x_edges, y_edges = np.histogram2d(
        x_arr,
        y_arr,
        bins=bins,
        range=[[-extent, extent], [-extent, extent]],
        weights=w_arr,
    )
    dx = np.diff(x_edges)[0]
    dy = np.diff(y_edges)[0]
    density = hist.T / (dx * dy)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return density.astype(np.float64), x_centers, y_centers


def smooth_density(
    density: ArrayLike,
    sigma: float = 1.5,
) -> NDArray[np.float64]:
    """Apply Gaussian smoothing to a density map."""

    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    arr = np.asarray(density, dtype=np.float64)
    return gaussian_filter(arr, sigma=sigma, mode="nearest")


def polar_resample(
    image: ArrayLike,
    x_centers: ArrayLike,
    y_centers: ArrayLike,
    n_radial: int = 128,
    n_azimuth: int = 256,
    r_max: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Resample a Cartesian image onto a polar grid.

    Args:
        image: Cartesian image with y, x indexing.
        x_centers: X coordinate centers for image columns.
        y_centers: Y coordinate centers for image rows.
        n_radial: Number of radial bins.
        n_azimuth: Number of azimuth bins.
        r_max: Maximum polar radius. Defaults to largest inscribed radius.

    Returns:
        Polar image with shape ``(n_radial, n_azimuth)``, radial centers,
        and azimuth centers in radians.
    """

    img = np.asarray(image, dtype=np.float64)
    x = np.asarray(x_centers, dtype=np.float64)
    y = np.asarray(y_centers, dtype=np.float64)
    if img.shape != (y.size, x.size):
        raise ValueError("image shape must match y and x centers")
    if n_radial < 2 or n_azimuth < 8:
        raise ValueError("polar grid is too small")

    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    if r_max is None:
        r_max = float(min(np.max(np.abs(x)), np.max(np.abs(y))))
    if r_max <= 0.0:
        raise ValueError("r_max must be positive")

    radial = np.linspace(0.0, r_max, n_radial)
    azimuth = np.linspace(-np.pi, np.pi, n_azimuth, endpoint=False)
    rr, tt = np.meshgrid(radial, azimuth, indexing="ij")
    xp = rr * np.cos(tt)
    yp = rr * np.sin(tt)
    col = (xp - x[0]) / dx
    row = (yp - y[0]) / dy
    polar = map_coordinates(img, [row, col], order=1, mode="constant", cval=0.0)
    return polar.astype(np.float64), radial, azimuth


def azimuthal_fft(
    polar_density: ArrayLike,
    max_mode: int = 8,
) -> NDArray[np.complex128]:
    """Compute complex azimuthal Fourier coefficients up to ``max_mode``.

    Args:
        polar_density: Polar surface-density map, indexed as radius, azimuth.
        max_mode: Highest azimuthal mode to retain.

    Returns:
        Complex coefficients with shape ``(max_mode + 1, n_radial)``.
    """

    polar = np.asarray(polar_density, dtype=np.float64)
    if polar.ndim != 2:
        raise ValueError("polar_density must be a 2D array")
    if max_mode < 0 or max_mode >= polar.shape[1] // 2:
        raise ValueError("max_mode must be non-negative and below Nyquist")
    coeff = np.fft.rfft(polar, axis=1) / polar.shape[1]
    coeff[:, 1:] *= 2.0
    return coeff[:, : max_mode + 1].T.astype(np.complex128)
