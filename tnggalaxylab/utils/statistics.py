"""Statistical helpers for robust galaxy diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def weighted_percentile(
    values: ArrayLike,
    percentiles: ArrayLike,
    weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute weighted percentiles.

    Args:
        values: Input values.
        percentiles: Percentiles in the inclusive range ``[0, 100]``.
        weights: Optional non-negative weights.

    Returns:
        Percentile values.
    """

    vals = np.asarray(values, dtype=np.float64).ravel()
    pct = np.asarray(percentiles, dtype=np.float64)
    if vals.size == 0:
        raise ValueError("values must not be empty")
    if np.any((pct < 0.0) | (pct > 100.0)):
        raise ValueError("percentiles must be in [0, 100]")
    if weights is None:
        return np.percentile(vals, pct)

    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != vals.shape:
        raise ValueError("weights must match values")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    if not np.any(w > 0.0):
        raise ValueError("at least one weight must be positive")

    sorter = np.argsort(vals)
    vals = vals[sorter]
    w = w[sorter]
    cumulative = np.cumsum(w)
    cumulative = 100.0 * (cumulative - 0.5 * w) / cumulative[-1]
    return np.interp(pct, cumulative, vals, left=vals[0], right=vals[-1])


def bootstrap_ci(
    values: ArrayLike,
    statistic=np.nanmean,
    confidence: float = 0.68,
    n_bootstrap: int = 512,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Estimate a bootstrap confidence interval.

    Args:
        values: One-dimensional sample.
        statistic: Callable evaluated on each bootstrap sample.
        confidence: Central confidence probability.
        n_bootstrap: Number of bootstrap realizations.
        seed: Optional random seed.

    Returns:
        Tuple of ``(central_value, lower_bound, upper_bound)``.
    """

    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("values must contain finite data")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")

    rng = np.random.default_rng(seed)
    samples = rng.choice(vals, size=(n_bootstrap, vals.size), replace=True)
    estimates = np.apply_along_axis(statistic, 1, samples)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(statistic(vals)), float(lower), float(upper)
