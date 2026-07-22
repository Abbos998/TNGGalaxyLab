"""
tnggalaxylab.fourier.diagnostics
=================================
Pattern diagnostics derived from Fourier profiles.

References
----------
Chequers, Widrow & Darling 2016, MNRAS 463, 1631
    Bar angle, bar length, and coherence length from phase flatness
    of the m=2 mode.
Aguerri, Elias-Rosa & Corsini 2009, A&A 494, 891
    Bar length from the A_2 radial profile maximum.
Debattista & Sellwood 2000, ApJ 543, 704
    Pattern speed from temporal phase evolution (not implemented here;
    requires multi-snapshot input).
Jog 2002, A&A 391, 471
    Dominant lopsided mode characterisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .core import FourierProfile


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PatternDiagnostics:
    """
    Full pattern diagnostics derived from a :class:`FourierProfile`.

    Attributes
    ----------
    dominant_mode : int
        Mode *m* that carries the highest disk-averaged amplitude.
    pattern_angle : float
        Position angle of the dominant mode [rad], computed as the
        amplitude-weighted mean phase over the valid radial range.
    bar_angle : float
        Position angle of the bar (dominant m=2 phase) [rad].
    bar_length : float
        Outermost radius where A_2 > *bar_threshold* [kpc].
    pattern_coherence : float
        Coherence length: fraction of radial bins where the m=2 phase
        is within В±*coherence_tol* rad of the mean bar angle.
    phase_scatter_m1 : float
        Circular standard deviation of the m=1 phase profile [rad].
    phase_scatter_m2 : float
        Circular standard deviation of the m=2 phase profile [rad].
    phase_stability : dict
        Circular standard deviation for each mode m в†’ Пѓ(Phi_m) [rad].
    bar_threshold : float
        A_2 amplitude threshold used to define the bar region.
    coherence_tol : float
        Phase tolerance used for the coherence calculation [rad].
    r_bar_profile : ndarray
        Radii of the A_2 profile used for bar diagnostics [kpc].
    A2_profile : ndarray
        A_2(R) values at those radii.
    """
    dominant_mode: int
    pattern_angle: float
    bar_angle: float
    bar_length: float
    pattern_coherence: float
    phase_scatter_m1: float
    phase_scatter_m2: float
    phase_stability: dict
    bar_threshold: float
    coherence_tol: float
    r_bar_profile: NDArray[np.float64]
    A2_profile: NDArray[np.float64]


# ---------------------------------------------------------------------------
# Helper: circular statistics
# ---------------------------------------------------------------------------

def _circular_std(angles: NDArray[np.float64]) -> float:
    """
    Circular standard deviation of an array of angles [rad].

    Uses the standard definition:
        Пѓ_c = sqrt(-2 ln RМ„)
    where RМ„ = |mean(exp(i Оё))| is the mean resultant length.
    """
    if len(angles) == 0:
        return np.nan
    R_bar = np.abs(np.mean(np.exp(1j * angles)))
    R_bar = np.clip(R_bar, 0.0, 1.0)
    val = max(-2.0 * np.log(max(R_bar, 1e-15)), 0.0)
    return float(np.sqrt(val))


def _circular_mean(angles: NDArray[np.float64]) -> float:
    """Circular mean of angles [rad]."""
    if len(angles) == 0:
        return np.nan
    return float(np.angle(np.mean(np.exp(1j * angles))))


# ---------------------------------------------------------------------------
# Main diagnostics function
# ---------------------------------------------------------------------------

def compute_pattern_diagnostics(
    profile: FourierProfile,
    *,
    bar_threshold: float = 0.2,
    coherence_tol: float = np.pi / 8.0,
    r_min: float = 0.0,
    r_max: Optional[float] = None,
) -> PatternDiagnostics:
    """
    Compute pattern diagnostics from a Fourier profile.

    Parameters
    ----------
    profile : FourierProfile
        Output of :func:`~tnggalaxylab.fourier.core.compute_fourier`.
    bar_threshold : float
        A_2 amplitude above which a radius is considered part of the bar.
        Default 0.2 (typical observational threshold, Aguerri et al. 2009).
    coherence_tol : float
        Phase tolerance [rad] for the coherence calculation.
        Default ПЂ/8 в‰€ 22.5В°.
    r_min, r_max : float
        Radial range [kpc] over which diagnostics are computed.
        Defaults to the full profile range.

    Returns
    -------
    PatternDiagnostics
    """
    if r_max is None:
        r_max = profile.r_out

    r = profile.r_bins
    mask = (r >= r_min) & (r <= r_max)
    r_sel = r[mask]

    # --- Dominant mode -------------------------------------------------------
    # Average amplitude across the selected radial range for each mode
    mean_amps = profile.amplitudes[mask].mean(axis=0)  # shape (m_max,)
    dominant_mode = int(np.argmax(mean_amps)) + 1       # 1-indexed

    # --- Pattern angle (amplitude-weighted circular mean of Phi_dominant) ----
    Phi_dom = profile.phases[mask, dominant_mode - 1]
    amp_dom = profile.amplitudes[mask, dominant_mode - 1]
    if amp_dom.sum() > 0:
        weights = amp_dom / amp_dom.sum()
        pattern_angle = float(np.angle(np.sum(weights * np.exp(1j * Phi_dom))))
    else:
        pattern_angle = np.nan

    # --- Bar diagnostics (m=2) -----------------------------------------------
    if profile.m_max >= 2:
        A2_sel = profile.A(2)[mask]
        Phi2_sel = profile.Phi(2)[mask]

        # Bar angle: amplitude-weighted circular mean of m=2 phase
        if A2_sel.sum() > 0:
            w2 = A2_sel / A2_sel.sum()
            bar_angle = float(np.angle(np.sum(w2 * np.exp(1j * Phi2_sel))))
        else:
            bar_angle = np.nan

        # --- Bar length: phase-coherent definition (Aguerri+2009 §3.1) -------
        #
        # The bar ends where the m=2 phase begins to twist away from the
        # bar phase by more than `coherence_tol` (~10–22.5°).  This avoids
        # spurious "bar" detections in shot-noise-dominated outer bins
        # where A_2 randomly crosses the amplitude threshold but the
        # phase wanders.  Standard reference:
        #   Aguerri J. A. L., Elias-Rosa N., Corsini E. M. 2009 A&A 494 891
        #   "The bar ends where the m=2 phase begins to twist."
        # Algorithm:
        #   1. Find inner-disk reference phase from amplitude-weighted mean
        #      over [r_min, r_anchor] (anchor at peak of A_2 if no R_d given).
        #   2. Walk outward; bar continues while
        #        (a) A_2(R) > bar_threshold
        #        (b) phase deviation from reference < coherence_tol
        #   3. Bar length = outermost R satisfying both, OR 0 if condition
        #      breaks at the first eligible radius.
        if (A2_sel > bar_threshold).any() and not np.isnan(bar_angle):
            # Reference phase: anchor at A_2 peak (Chequers+2016 §2.2)
            peak_idx = int(np.argmax(A2_sel))
            ref_phi = float(Phi2_sel[peak_idx])

            # Walk outward from peak; require A_2 high AND phase coherent
            bar_length = 0.0
            for i in range(peak_idx, len(r_sel)):
                if A2_sel[i] <= bar_threshold:
                    break
                dphi = (Phi2_sel[i] - ref_phi + np.pi) % (2 * np.pi) - np.pi
                if abs(dphi) > coherence_tol:
                    break
                bar_length = float(r_sel[i])
        else:
            bar_length = 0.0

        # Phase coherence: fraction of bins (where A_2 > threshold) that
        # lie within ±coherence_tol of the bar phase.  This is reported as
        # a quality flag — coherence < 0.5 indicates a non-coherent
        # detection (likely shot noise).
        if not np.isnan(bar_angle) and len(Phi2_sel) > 0:
            strong = A2_sel > bar_threshold
            if strong.any():
                delta_phi = (Phi2_sel[strong] - bar_angle + np.pi) % (2 * np.pi) - np.pi
                coherent_frac = float(np.mean(np.abs(delta_phi) < coherence_tol))
            else:
                coherent_frac = 0.0
        else:
            coherent_frac = np.nan

        phase_scatter_m2 = _circular_std(Phi2_sel)
    else:
        A2_sel = np.zeros(mask.sum())
        bar_angle   = np.nan
        bar_length  = 0.0
        coherent_frac = np.nan
        phase_scatter_m2 = np.nan

    # --- Phase scatter for m=1 -----------------------------------------------
    if profile.m_max >= 1:
        Phi1_sel = profile.Phi(1)[mask]
        phase_scatter_m1 = _circular_std(Phi1_sel)
    else:
        phase_scatter_m1 = np.nan

    # --- Phase stability for all modes ----------------------------------------
    phase_stability = {}
    for mi in range(1, profile.m_max + 1):
        phi_m = profile.phases[mask, mi - 1]
        phase_stability[mi] = _circular_std(phi_m)

    return PatternDiagnostics(
        dominant_mode=dominant_mode,
        pattern_angle=pattern_angle,
        bar_angle=bar_angle,
        bar_length=bar_length,
        pattern_coherence=coherent_frac,
        phase_scatter_m1=phase_scatter_m1,
        phase_scatter_m2=phase_scatter_m2,
        phase_stability=phase_stability,
        bar_threshold=bar_threshold,
        coherence_tol=coherence_tol,
        r_bar_profile=r_sel,
        A2_profile=A2_sel,
    )
