"""
tnggalaxylab.fourier.core
=========================
Publication-quality Fourier decomposition of galaxy surface density.

Scientific references
---------------------
Rix & Zaritsky 1995, ApJ 447, 82  (R&Z95)
    Standard definition of A_m: amplitude of cosine Fourier mode.
    A_1 measured over 1.5вЂ“2.5 R_d to characterise the lopsidedness.

Zaritsky & Rix 1997, ApJ 475, 118  (ZR97)
    Extended catalogue of A_1 in late-type spirals; average definition.

Bournaud, Combes & Jog 2005, A&A 437, 69  (BCJ05)
    Lopsidedness driven by tidal interactions; uses the same R&Z95
    normalisation.  Confirms the 1.5вЂ“2.5 R_d aperture.

Jog 2002, A&A 391, 471  (J02)
    Theoretical dispersion relation for m=1; global A_1 integral
    over the full disk as a complementary measure.

Saha, Combes & Jog 2007, MNRAS 382, 419
    Direct particle Fourier sums for N-body snapshots.

Chequers, Widrow & Darling 2016, MNRAS 463, 1631
    Bar diagnostics from A_2 phase; pattern-speed coherence length.

Mathematical conventions
------------------------
The surface-density Fourier expansion is

    Sigma(R, phi) = a_0(R)/2 + sum_{m=1}^{M} a_m(R) cos(m phi)
                                               + b_m(R) sin(m phi)

The *amplitude* of mode m at radius R is

    A_m(R) = sqrt(a_m^2 + b_m^2) / (a_0/2)          [Eq. 1, R&Z95]

and the *phase* is

    Phi_m(R) = (1/m) arctan2(b_m, a_m)               [Eq. 2, R&Z95]

For the **particle method** (Saha et al. 2007 estimator, normalised to
the R&Z95 convention), with particle masses m_j:

    C_m(R) = sum_{j in annulus} m_j cos(m theta_j)
    S_m(R) = sum_{j in annulus} m_j sin(m theta_j)
    M_0(R) = sum_{j in annulus} m_j

    A_m(R) = 2 sqrt(C_m^2 + S_m^2) / M_0            [Eq. 3, R&Z95-normalised]
    Phi_m(R) = (1/m) arctan2(S_m, C_m)              [Eq. 4]

The factor of 2 reconciles the particle moment estimator with the
R&Z95 amplitude defined as a_m/(a_0/2).  Without it, the recovered
amplitude is half the standard literature value (the Saha 2007 I_m
convention).

Global lopsidedness (R&Z95 / BCJ05 standard):

    <A_1> = (1 / N_bins) sum_{R_i in [1.5, 2.5] R_d} A_1(R_i)   [Eq. 5]

or equivalently the integral form (J02):

    A_1_global = integral_{R_in}^{R_out} A_1(R) Sigma(R) R dR
                 / integral_{R_in}^{R_out} Sigma(R) R dR          [Eq. 6]

Both are returned; the default reported value follows R&Z95.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Public result containers
# ---------------------------------------------------------------------------

@dataclass
class FourierProfile:
    """
    Radial Fourier mode profiles for a single snapshot.

    Attributes
    ----------
    r_bins : ndarray, shape (N,)
        Bin centres [kpc].
    amplitudes : ndarray, shape (N, M)
        A_m(R) for m = 1, 2, ..., M.  Column 0 в†’ m=1.
    phases : ndarray, shape (N, M)
        Phi_m(R) [rad] for m = 1, 2, ..., M.
    m_max : int
        Highest mode stored.
    method : str
        "fft" or "particles".
    n_bins : int
        Number of radial bins.
    r_in : float
        Inner radius [kpc].
    r_out : float
        Outer radius [kpc].
    """
    r_bins: NDArray[np.float64]
    amplitudes: NDArray[np.float64]   # shape (N, M)
    phases: NDArray[np.float64]       # shape (N, M)
    m_max: int
    method: str
    n_bins: int
    r_in: float
    r_out: float
    # optional per-bin uncertainties from bootstrap
    amp_err: Optional[NDArray[np.float64]] = None   # shape (N, M)
    phase_err: Optional[NDArray[np.float64]] = None

    def A(self, m: int) -> NDArray[np.float64]:
        """Return radial profile of amplitude for mode *m* (1-indexed)."""
        if not 1 <= m <= self.m_max:
            raise ValueError(f"m must be in [1, {self.m_max}].")
        return self.amplitudes[:, m - 1]

    def Phi(self, m: int) -> NDArray[np.float64]:
        """Return radial profile of phase [rad] for mode *m* (1-indexed)."""
        if not 1 <= m <= self.m_max:
            raise ValueError(f"m must be in [1, {self.m_max}].")
        return self.phases[:, m - 1]


@dataclass
class GlobalModes:
    """
    Disk-integrated Fourier amplitudes.

    Attributes
    ----------
    A1_literature : float
        Mean A_1 over [1.5, 2.5] R_d (R&Z95 standard).
    A2_literature : float
        Mean A_2 over the same aperture.
    A1_integral : float
        Mass-weighted integral A_1 over the same aperture (J02).
    A2_integral : float
        Mass-weighted integral A_2.
    r_range_kpc : tuple
        (R_in, R_out) [kpc] used in the integration.
    scale_length_kpc : float or None
        Disk scale length R_d used.  None if user supplied r_range directly.
    """
    A1_literature: float
    A2_literature: float
    A1_integral: float
    A2_integral: float
    r_range_kpc: Tuple[float, float]
    scale_length_kpc: Optional[float] = None
    # backward-compatible aliases
    A1: float = field(init=False)
    A2: float = field(init=False)

    def __post_init__(self):
        # Default reported values follow R&Z95 (literature average)
        self.A1 = self.A1_literature
        self.A2 = self.A2_literature


# ---------------------------------------------------------------------------
# FFT-based Fourier decomposition
# ---------------------------------------------------------------------------

def _project_particles(
    x: NDArray, y: NDArray, mass: NDArray,
    n_pix: int, r_out: float,
) -> NDArray[np.float64]:
    """
    Project particles onto a 2-D surface density grid.

    Parameters
    ----------
    x, y : ndarray   Cartesian coords [kpc], centred on galaxy.
    mass : ndarray   Particle masses [M_sun or code units].
    n_pix : int      Grid side length.
    r_out : float    Half-size of the grid [kpc].

    Returns
    -------
    grid : ndarray, shape (n_pix, n_pix)
        Surface density in mass units / kpc^2.
    """
    pix_size = 2.0 * r_out / n_pix
    i_idx = ((x + r_out) / pix_size).astype(int)
    j_idx = ((y + r_out) / pix_size).astype(int)
    mask = (i_idx >= 0) & (i_idx < n_pix) & (j_idx >= 0) & (j_idx < n_pix)
    grid = np.zeros((n_pix, n_pix), dtype=np.float64)
    np.add.at(grid, (j_idx[mask], i_idx[mask]), mass[mask])
    return grid / pix_size**2


def fft_fourier(
    x: NDArray,
    y: NDArray,
    mass: NDArray,
    *,
    r_in: float = 0.0,
    r_out: float = 20.0,
    n_bins: int = 40,
    m_max: int = 4,
    n_pix: int = 512,
    sigma_smooth: float = 2.0,
    theta_bins: int = 256,
) -> FourierProfile:
    """
    FFT-based azimuthal Fourier decomposition.

    The surface-density map is constructed by particle projection,
    optionally Gaussian-smoothed, then sampled on a polar grid.
    The azimuthal FFT along each radial annulus gives the complex
    Fourier coefficients; amplitudes are normalised by the m=0
    coefficient (Eq. 1).

    Parameters
    ----------
    x, y : ndarray
        Particle positions [kpc], centred on the galaxy.
    mass : ndarray
        Particle masses [M_sun or code units].
    r_in, r_out : float
        Radial range [kpc] for the Fourier profile.
    n_bins : int
        Number of radial bins.
    m_max : int
        Highest azimuthal mode to compute.
    n_pix : int
        Cartesian grid side [pixels].  Must be even.
    sigma_smooth : float
        Gaussian smoothing width [pixels]; 0 disables smoothing.
    theta_bins : int
        Number of azimuthal sampling points per annulus.

    Returns
    -------
    FourierProfile
    """
    # --- 1. Surface density map -------------------------------------------
    grid = _project_particles(x, y, mass, n_pix=n_pix, r_out=r_out)
    if sigma_smooth > 0.0:
        grid = gaussian_filter(grid.astype(float), sigma=sigma_smooth)

    # --- 2. Polar sampling -----------------------------------------------
    r_edges = np.linspace(r_in, r_out, n_bins + 1)
    r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])
    theta = np.linspace(0.0, 2.0 * np.pi, theta_bins, endpoint=False)

    pix_size = 2.0 * r_out / n_pix
    cx, cy = n_pix / 2.0, n_pix / 2.0

    amplitudes = np.zeros((n_bins, m_max))
    phases     = np.zeros((n_bins, m_max))

    for i, r in enumerate(r_centres):
        xi = cx + r / pix_size * np.cos(theta)
        yi = cy + r / pix_size * np.sin(theta)

        # bilinear interpolation
        xi0 = np.floor(xi).astype(int)
        yi0 = np.floor(yi).astype(int)
        xi1 = xi0 + 1
        yi1 = yi0 + 1

        # clip to valid range
        xi0c = np.clip(xi0, 0, n_pix - 1)
        yi0c = np.clip(yi0, 0, n_pix - 1)
        xi1c = np.clip(xi1, 0, n_pix - 1)
        yi1c = np.clip(yi1, 0, n_pix - 1)

        fx = xi - xi0
        fy = yi - yi0

        profile = (
            grid[yi0c, xi0c] * (1 - fx) * (1 - fy)
            + grid[yi0c, xi1c] * fx * (1 - fy)
            + grid[yi1c, xi0c] * (1 - fx) * fy
            + grid[yi1c, xi1c] * fx * fy
        )

        a0_half = np.mean(profile)    # a_0 / 2
        if a0_half <= 0.0:
            continue

        # FFT along theta
        fft_c = np.fft.rfft(profile) / theta_bins

        for mi, m in enumerate(range(1, m_max + 1)):
            if m >= len(fft_c):
                break
            c_m = fft_c[m]
            # c_m = (a_m - i b_m) / 2  in numpy convention
            a_m = 2.0 * c_m.real
            b_m = -2.0 * c_m.imag
            amplitudes[i, mi] = np.sqrt(a_m**2 + b_m**2) / a0_half
            phases[i, mi]     = np.arctan2(b_m, a_m) / m

    return FourierProfile(
        r_bins=r_centres,
        amplitudes=amplitudes,
        phases=phases,
        m_max=m_max,
        method="fft",
        n_bins=n_bins,
        r_in=r_in,
        r_out=r_out,
    )


# ---------------------------------------------------------------------------
# Particle-direct Fourier decomposition
# ---------------------------------------------------------------------------

def particle_fourier(
    x: NDArray,
    y: NDArray,
    mass: NDArray,
    *,
    r_in: float = 0.0,
    r_out: float = 20.0,
    n_bins: int = 40,
    m_max: int = 4,
) -> FourierProfile:
    """
    Particle-direct Fourier decomposition (Saha, Combes & Jog 2007).

    No gridding, smoothing, or FFT image is used.  For each radial
    annulus the complex Fourier coefficients are computed by direct
    mass-weighted summation over particles (Eqs. 3вЂ“4).

    This method is free of projection biases and pixel-size artefacts
    and is preferred for N-body validation.  It becomes noisy for
    underpopulated annuli (see bootstrap for uncertainties).

    Parameters
    ----------
    x, y : ndarray
        Particle positions [kpc].
    mass : ndarray
        Particle masses [M_sun or code units].
    r_in, r_out : float
        Radial range [kpc].
    n_bins : int
        Number of radial bins.
    m_max : int
        Highest azimuthal mode.

    Returns
    -------
    FourierProfile
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)   # [-pi, pi]

    r_edges  = np.linspace(r_in, r_out, n_bins + 1)
    r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])

    amplitudes = np.zeros((n_bins, m_max))
    phases     = np.zeros((n_bins, m_max))

    for i in range(n_bins):
        mask = (r >= r_edges[i]) & (r < r_edges[i + 1])
        m_ann = mass[mask]
        t_ann = theta[mask]

        M0 = m_ann.sum()
        if M0 <= 0.0:
            continue

        for mi, m_ord in enumerate(range(1, m_max + 1)):
            C_m = np.sum(m_ann * np.cos(m_ord * t_ann))
            S_m = np.sum(m_ann * np.sin(m_ord * t_ann))
            # R&Z95 normalisation: A_m = a_m/(a_0/2)
            # In continuum limit: C_m/M0 = a_m/a_0 = (a_m/(a_0/2)) / 2 = A_m/2
            # Therefore A_m_RZ95 = 2 * sqrt(C_m^2 + S_m^2) / M0
            amplitudes[i, mi] = 2.0 * np.sqrt(C_m**2 + S_m**2) / M0
            phases[i, mi]     = np.arctan2(S_m, C_m) / m_ord

    return FourierProfile(
        r_bins=r_centres,
        amplitudes=amplitudes,
        phases=phases,
        m_max=m_max,
        method="particles",
        n_bins=n_bins,
        r_in=r_in,
        r_out=r_out,
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def compute_fourier(
    x: NDArray,
    y: NDArray,
    mass: NDArray,
    *,
    method: str = "fft",
    r_in: float = 0.0,
    r_out: float = 20.0,
    n_bins: int = 40,
    m_max: int = 4,
    # FFT-specific kwargs
    n_pix: int = 512,
    sigma_smooth: float = 2.0,
    theta_bins: int = 256,
) -> FourierProfile:
    """
    Compute Fourier decomposition of a galaxy's projected surface density.

    This is the main entry point.  Choose *method* to select the
    algorithm; all other arguments are common.

    Parameters
    ----------
    x, y : ndarray
        Particle positions [kpc], centred on the galaxy.
    mass : ndarray
        Particle masses.
    method : {"fft", "particles"}
        "fft"       вЂ” project onto a pixel grid then FFT azimuthal
                      strips (fast, suitable for large particle counts).
        "particles" вЂ” direct mass-weighted complex sum per annulus
                      (Saha et al. 2007; preferred for validation).
    r_in, r_out : float
        Radial range [kpc].
    n_bins : int
        Number of radial bins.
    m_max : int
        Highest mode.
    n_pix : int
        Pixel grid side (*fft* only).
    sigma_smooth : float
        Gaussian smoothing [pixels] (*fft* only).  0 в†’ no smoothing.
    theta_bins : int
        Azimuthal sampling points per annulus (*fft* only).

    Returns
    -------
    FourierProfile
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)

    if method == "fft":
        return fft_fourier(
            x, y, mass,
            r_in=r_in, r_out=r_out, n_bins=n_bins, m_max=m_max,
            n_pix=n_pix, sigma_smooth=sigma_smooth, theta_bins=theta_bins,
        )
    elif method == "particles":
        return particle_fourier(
            x, y, mass,
            r_in=r_in, r_out=r_out, n_bins=n_bins, m_max=m_max,
        )
    else:
        raise ValueError(f"method must be 'fft' or 'particles', got '{method}'.")


# ---------------------------------------------------------------------------
# Global lopsidedness
# ---------------------------------------------------------------------------

def global_lopsidedness(
    profile: FourierProfile,
    scale_length: Optional[float] = None,
    r_range: Optional[Tuple[float, float]] = None,
    method: str = "literature_average",
    surface_density: Optional[NDArray] = None,
) -> GlobalModes:
    """
    Compute disk-integrated Fourier amplitudes.

    Two definitions are supported (both returned):

    ``literature_average``  (default, R&Z95 / BCJ05)
        Simple mean of A_m(R) over the aperture [1.5, 2.5] R_d.
        This is the standard value quoted in lopsidedness catalogues.

    ``global_integral``  (J02)
        Mass-weighted radial integral over the aperture.  If
        *surface_density* Σ(R) is provided, the **fully rigorous**
        Jog 2002 (Eq. 6) form is used:

            <A_m>_Σ = ∫ A_m(R) · Σ(R) · 2π R dR  /  ∫ Σ(R) · 2π R dR

        If *surface_density* is not supplied, a geometric area-weight
        proxy (w = R · dR) is used.  For an exponential disk this
        differs from the rigorous J02 form by an exp(−R/R_d) weighting;
        over [1.5, 2.5] R_d the difference can reach ~30%.

    Parameters
    ----------
    profile : FourierProfile
        Output of :func:`compute_fourier`.
    scale_length : float or None
        Exponential disk scale length R_d [kpc].  If given the default
        aperture is [1.5, 2.5] R_d.  Must supply either *scale_length*
        or *r_range*.
    r_range : (float, float) or None
        Explicit (R_in, R_out) [kpc] aperture.  Overrides
        *scale_length*.
    method : {"literature_average", "global_integral"}
        Which value is returned as the primary A1/A2.  Both are always
        computed and stored.
    surface_density : ndarray or None
        Optional array of Σ(R) values evaluated at *profile.r_bins*.
        Same length as profile.r_bins.  When provided, the integral
        form uses the rigorous mass-weighting of Jog 2002 (Eq. 6).
        When omitted, the integral falls back to area weighting
        (backward-compatible default).

    Returns
    -------
    GlobalModes
    """
    if r_range is not None:
        R_in, R_out = float(r_range[0]), float(r_range[1])
        Rd = None
    elif scale_length is not None:
        Rd = float(scale_length)
        R_in, R_out = 1.5 * Rd, 2.5 * Rd
    else:
        raise ValueError("Supply scale_length or r_range.")

    r = profile.r_bins
    mask = (r >= R_in) & (r <= R_out)

    if mask.sum() == 0:
        warnings.warn(
            f"No radial bins found in [{R_in:.2f}, {R_out:.2f}] kpc. "
            "Check scale_length and r_out.",
            stacklevel=2,
        )
        return GlobalModes(
            A1_literature=np.nan, A2_literature=np.nan,
            A1_integral=np.nan, A2_integral=np.nan,
            r_range_kpc=(R_in, R_out), scale_length_kpc=Rd,
        )

    A1_r = profile.A(1)[mask]
    A2_r = profile.A(2)[mask] if profile.m_max >= 2 else np.zeros_like(A1_r)

    # Literature average (R&Z95, Eq. 5)
    A1_lit = float(np.mean(A1_r))
    A2_lit = float(np.mean(A2_r))

    # Integral form (J02 Eq. 6) -- now supports rigorous Σ(R) weighting
    r_sel = r[mask]
    dr    = np.diff(profile.r_bins)
    dr_sel = np.interp(r_sel,
                       0.5 * (profile.r_bins[:-1] + profile.r_bins[1:]),
                       dr)
    if surface_density is not None:
        # Rigorous Jog 2002 form: weights = Σ(R) · 2π R · dR
        sig = np.asarray(surface_density, dtype=np.float64)
        if sig.shape != profile.r_bins.shape:
            raise ValueError(
                f"surface_density shape {sig.shape} must match "
                f"profile.r_bins shape {profile.r_bins.shape}"
            )
        sig_sel = sig[mask]
        weights = sig_sel * r_sel * dr_sel   # 2π drops in ratio
    else:
        # Backward-compatible: area weighting (geometric proxy)
        weights = r_sel * dr_sel

    norm = weights.sum()
    A1_int = float(np.sum(A1_r * weights) / norm) if norm > 0 else np.nan
    A2_int = float(np.sum(A2_r * weights) / norm) if norm > 0 else np.nan

    gm = GlobalModes(
        A1_literature=A1_lit,
        A2_literature=A2_lit,
        A1_integral=A1_int,
        A2_integral=A2_int,
        r_range_kpc=(R_in, R_out),
        scale_length_kpc=Rd,
    )

    # Override primary A1/A2 according to requested method
    if method == "global_integral":
        gm.A1 = A1_int
        gm.A2 = A2_int
    # else: defaults remain literature_average (set in __post_init__)

    return gm
