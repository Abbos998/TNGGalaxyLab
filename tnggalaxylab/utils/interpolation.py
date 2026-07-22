"""Interpolation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d


def monotonic_interpolator(
    x: ArrayLike,
    y: ArrayLike,
    fill_value: str | tuple[float, float] = "extrapolate",
) -> interp1d:
    """Create a one-dimensional interpolator sorted by ``x``.

    Args:
        x: Coordinate values.
        y: Data values.
        fill_value: Fill behavior passed to SciPy.

    Returns:
        SciPy ``interp1d`` callable.
    """

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.size != y_arr.size:
        raise ValueError("x and y must be one-dimensional arrays with equal size")
    order = np.argsort(x_arr)
    return interp1d(
        x_arr[order],
        y_arr[order],
        bounds_error=False,
        fill_value=fill_value,
        assume_sorted=True,
    )
