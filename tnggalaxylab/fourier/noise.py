"""
tnggalaxylab.fourier.noise
Shot-noise floor and noise-corrected global Fourier amplitudes.

A_m = 2|c_m|/c_0 is positive-definite, so Poisson noise biases it upward
(E[A_m] ~ sqrt(pi/N_eff) per annulus for pure noise).  The noise floor is
measured by randomising particle azimuths (radii and masses kept), and
    A_m,corr = sqrt(max(<A_m>^2 - <A_m,noise>^2, 0))
is applied to aperture averages.  Validated in Paper I on TNG50-1 discs
downsampled to TNG100-1 mass resolution (recovery to ~10 per cent).
"""
from __future__ import annotations
import copy
import dataclasses
import warnings
from typing import Optional, Tuple

import numpy as np

from .core import compute_fourier, global_lopsidedness


def _with_amplitudes(profile, amplitudes):
    try:
        return dataclasses.replace(profile, amplitudes=amplitudes)
    except TypeError:
        p = copy.copy(profile)
        p.amplitudes = amplitudes
        return p


def noise_floor_profile(x, y, mass, *, n_shuffle=5, seed=0, **compute_kw):
    """Mean amplitude profile of n_shuffle azimuth-randomised copies."""
    compute_kw.setdefault("method", "particles")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    R = np.hypot(x, y)
    rng = np.random.default_rng(seed)
    amps, prof = [], None
    for _ in range(int(n_shuffle)):
        phi = rng.uniform(-np.pi, np.pi, R.size)
        prof = compute_fourier(R * np.cos(phi), R * np.sin(phi), mass, **compute_kw)
        amps.append(np.asarray(prof.amplitudes, dtype=np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_amp = np.nanmean(np.stack(amps), axis=0)
    return _with_amplitudes(prof, mean_amp)


def neff_in_range(x, y, mass, r_range: Tuple[float, float]) -> float:
    """Effective particle number (sum m)^2 / sum m^2 inside an annulus."""
    R = np.hypot(np.asarray(x, float), np.asarray(y, float))
    s = (R >= r_range[0]) & (R <= r_range[1])
    if not s.any():
        return 0.0
    m = np.asarray(mass, float)[s]
    return float(m.sum() ** 2 / np.sum(m ** 2))


def _corr(a, n):
    if not (np.isfinite(a) and np.isfinite(n)):
        return np.nan
    return float(np.sqrt(max(a * a - n * n, 0.0)))


def noise_corrected_global(x, y, mass, *, scale_length: Optional[float] = None,
                           r_range: Optional[Tuple[float, float]] = None,
                           profile=None, noise_profile=None,
                           n_shuffle=5, seed=0, **compute_kw) -> dict:
    """A1, A2, their noise floors and noise-corrected values in an aperture."""
    compute_kw.setdefault("method", "particles")
    if profile is None:
        profile = compute_fourier(x, y, mass, **compute_kw)
    if noise_profile is None:
        noise_profile = noise_floor_profile(x, y, mass, n_shuffle=n_shuffle,
                                            seed=seed, **compute_kw)
    gm = global_lopsidedness(profile, scale_length=scale_length, r_range=r_range)
    gn = global_lopsidedness(noise_profile, scale_length=scale_length, r_range=r_range)
    lo, hi = gm.r_range_kpc
    return dict(
        A1=float(gm.A1_literature), A2=float(gm.A2_literature),
        A1_noise=float(gn.A1_literature), A2_noise=float(gn.A2_literature),
        A1_corr=_corr(gm.A1_literature, gn.A1_literature),
        A2_corr=_corr(gm.A2_literature, gn.A2_literature),
        r_range_kpc=(float(lo), float(hi)),
        neff_aperture=neff_in_range(x, y, mass, (lo, hi)),
    )


def bootstrap_bias_corrected(estimate, boot_mean):
    """Efron bias-corrected estimate: 2*theta_hat - mean(theta_boot)."""
    if not (np.isfinite(estimate) and np.isfinite(boot_mean)):
        return np.nan
    return float(2.0 * estimate - boot_mean)
