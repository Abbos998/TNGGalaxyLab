"""Coordinate transforms used by particle and image analyses."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_position_array(positions: ArrayLike) -> NDArray[np.float64]:
    """Validate and return an ``(N, 3)`` position array.

    Args:
        positions: Cartesian particle coordinates.

    Returns:
        Float64 NumPy array with shape ``(N, 3)``.

    Raises:
        ValueError: If the input does not have shape ``(N, 3)``.
    """

    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    return arr


def cartesian_to_polar(
    x: ArrayLike,
    y: ArrayLike,
    center: tuple[float, float] = (0.0, 0.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert Cartesian coordinates to polar coordinates.

    Args:
        x: X coordinates.
        y: Y coordinates.
        center: Origin used for the transform.

    Returns:
        Radius and azimuth arrays. Azimuth is in radians in ``[-pi, pi]``.
    """

    x_arr = np.asarray(x, dtype=np.float64) - center[0]
    y_arr = np.asarray(y, dtype=np.float64) - center[1]
    radius = np.hypot(x_arr, y_arr)
    theta = np.arctan2(y_arr, x_arr)
    return radius, theta


def rotate_positions(
    positions: ArrayLike,
    rotation_matrix: ArrayLike,
) -> NDArray[np.float64]:
    """Rotate Cartesian positions using a right-multiplication matrix.

    Args:
        positions: Cartesian coordinates with shape ``(N, 3)``.
        rotation_matrix: Rotation matrix with shape ``(3, 3)``.

    Returns:
        Rotated position array.
    """

    pos = as_position_array(positions)
    rot = np.asarray(rotation_matrix, dtype=np.float64)
    if rot.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3, 3)")
    return pos @ rot.T


def cylindrical_radius(positions: ArrayLike) -> NDArray[np.float64]:
    """Return projected radius in the x-y plane.

    Args:
        positions: Cartesian coordinates with shape ``(N, 3)``.

    Returns:
        Cylindrical radius array.
    """

    pos = as_position_array(positions)
    return np.hypot(pos[:, 0], pos[:, 1])
