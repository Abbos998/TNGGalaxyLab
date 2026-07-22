"""
tnggalaxylab.fourier.bootstrap
==============================
Bootstrap resampling for Fourier amplitude uncertainties.

The bootstrap method (Efron 1979) estimates the sampling distribution
of A_m(R) by repeatedly resampling the particle set with replacement
and re-computing the Fourier profile.  The standard deviation across
bootstrap iterations gives Пѓ(A_m(R)).

This is particularly important for:
- N-body snapshots with low particle counts per annulus
- Distinguishing physical asymmetry from shot noise
- Providing error bars for publication plots

References
----------
Efron 1979, Ann. Stat. 7, 1
Chequers et al. 2016, MNRAS 463, 1631  (bootstrap in bar diagnostics)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .core import FourierProfile, compute_fourier, global_lopsidedness, GlobalModes


def bootstrap_fourier(
    x: NDArray,
    y: NDArray,
    mass: NDArray,
    *,
    method: str = "particles",
    n_bootstrap: int = 500,
    seed: int = 0,
    r_in: float = 0.0,
    r_out: float = 20.0,
    n_bins: int = 40,
    m_max: int = 4,
    # FFT kwargs forwarded when method="fft"
    n_pix: int = 512,
    sigma_smooth: float = 2.0,
    theta_bins: int = 256,
    # Global mode kwargs
    scale_length: Optional[float] = None,
    r_range: Optional[tuple] = None,
) -> dict:
    """
    Bootstrap uncertainties for Fourier amplitudes and global modes.

    Parameters
    ----------
    x, y : ndarray    Particle positions [kpc].
    mass : ndarray    Particle masses.
    method : str      "particles" (recommended) or "fft".
    n_bootstrap : int Number of bootstrap iterations.  100вЂ“500 is typical.
    seed : int        RNG seed for reproducibility.
    r_in, r_out, n_bins, m_max :
                      Passed to :func:`compute_fourier`.
    n_pix, sigma_smooth, theta_bins :
                      FFT-specific kwargs.
    scale_length, r_range :
                      Passed to :func:`global_lopsidedness` if provided.

    Returns
    -------
    dict with keys:
        profile : FourierProfile
            Mean profile with amp_err and phase_err filled.
        A_mean : ndarray, shape (n_bins, m_max)
            Mean amplitude across bootstrap samples.
        A_std : ndarray, shape (n_bins, m_max)
            Standard deviation (= Пѓ(A_m)).
        Phi_mean : ndarray, shape (n_bins, m_max)
            Circular mean phase.
        Phi_std : ndarray, shape (n_bins, m_max)
            Circular standard deviation of phase.
        global_A1_mean : float or None
        global_A1_std : float or None
        global_A2_mean : float or None
        global_A2_std : float or None
        bar_length_mean : float or None
        bar_length_std : float or None
        bar_angle_mean : float or None
        bar_angle_std : float or None
        n_bootstrap : int
    """
    rng = np.random.default_rng(seed)
    n = len(x)

    fft_kw = dict(n_pix=n_pix, sigma_smooth=sigma_smooth, theta_bins=theta_bins)
    common_kw = dict(
        method=method, r_in=r_in, r_out=r_out, n_bins=n_bins, m_max=m_max,
        **fft_kw,
    )

    amp_stack  = []
    phase_stack = []
    A1_stack, A2_stack = [], []
    bl_stack, ba_stack = [], []

    do_global = (scale_length is not None) or (r_range is not None)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb, yb, mb = x[idx], y[idx], mass[idx]
        prof_b = compute_fourier(xb, yb, mb, **common_kw)
        amp_stack.append(prof_b.amplitudes)
        phase_stack.append(prof_b.phases)

        if do_global:
            gm_b = global_lopsidedness(
                prof_b,
                scale_length=scale_length,
                r_range=r_range,
            )
            A1_stack.append(gm_b.A1_literature)
            A2_stack.append(gm_b.A2_literature)

        # Bar diagnostics
        if m_max >= 2:
            A2_r = prof_b.A(2)
            bl = _bar_length(prof_b.r_bins, A2_r, threshold=0.2)
            ba = _bar_angle(prof_b, r_in=r_in, r_out=r_out)
            bl_stack.append(bl)
            ba_stack.append(ba)

    amp_arr  = np.array(amp_stack)   # (n_boot, n_bins, m_max)
    phase_arr = np.array(phase_stack)

    A_mean = amp_arr.mean(axis=0)
    A_std  = amp_arr.std(axis=0)

    # Circular statistics for phases
    exp_phi = np.exp(1j * phase_arr)
    Phi_mean_c = np.angle(exp_phi.mean(axis=0))
    R_bar = np.abs(exp_phi.mean(axis=0))
    R_bar = np.clip(R_bar, 0.0, 1.0)
    Phi_std = np.sqrt(-2.0 * np.log(R_bar + 1e-15))

    # Assemble mean profile with uncertainties
    base_profile = compute_fourier(x, y, mass, **common_kw)
    base_profile.amp_err   = A_std
    base_profile.phase_err = Phi_std

    result = dict(
        profile=base_profile,
        A_mean=A_mean,
        A_std=A_std,
        Phi_mean=Phi_mean_c,
        Phi_std=Phi_std,
        n_bootstrap=n_bootstrap,
        global_A1_mean=float(np.mean(A1_stack)) if A1_stack else None,
        global_A1_std=float(np.std(A1_stack))  if A1_stack else None,
        global_A2_mean=float(np.mean(A2_stack)) if A2_stack else None,
        global_A2_std=float(np.std(A2_stack))  if A2_stack else None,
        bar_length_mean=float(np.mean(bl_stack)) if bl_stack else None,
        bar_length_std=float(np.std(bl_stack))  if bl_stack else None,
        bar_angle_mean=float(np.degrees(
            np.angle(np.mean(np.exp(1j * np.array(ba_stack)[~np.isnan(ba_stack)])))
        )) if ba_stack else None,
        bar_angle_std=None,  # filled below
    )

    if ba_stack:
        ba_arr = np.array(ba_stack)
        ba_arr = ba_arr[~np.isnan(ba_arr)]
        if len(ba_arr) > 1:
            R_bar_ba = np.abs(np.mean(np.exp(1j * ba_arr)))
            R_bar_ba = np.clip(R_bar_ba, 0.0, 1.0)
            ba_std_rad = float(np.sqrt(-2.0 * np.log(R_bar_ba + 1e-15)))
            result["bar_angle_std"] = float(np.degrees(ba_std_rad))

    return result


# ---------------------------------------------------------------------------
# Internal helpers shared with diagnostics
# ---------------------------------------------------------------------------

def _bar_length(r: NDArray, A2: NDArray, threshold: float = 0.2) -> float:
    mask = A2 > threshold
    return float(r[mask][-1]) if mask.any() else 0.0


def _bar_angle(profile: FourierProfile, r_in: float, r_out: float) -> float:
    from .diagnostics import _circular_mean
    r = profile.r_bins
    mask = (r >= r_in) & (r <= r_out)
    if not mask.any():
        return np.nan
    Phi2 = profile.Phi(2)[mask]
    A2   = profile.A(2)[mask]
    if A2.sum() <= 0:
        return np.nan
    w = A2 / A2.sum()
    return float(np.angle(np.sum(w * np.exp(1j * Phi2))))
