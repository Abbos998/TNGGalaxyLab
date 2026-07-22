"""
tnggalaxylab.fourier.synthetic
==============================
Synthetic galaxy generators with analytically known Fourier amplitudes.

Each generator returns (x, y, mass) particle arrays suitable for
passing directly to :func:`~tnggalaxylab.fourier.core.compute_fourier`.
The analytic Fourier amplitudes are also returned so that numerical
recovery can be validated automatically.

Mathematical background
-----------------------
Exponential disk
    Sigma(R) = Sigma_0 exp(-R / R_d)
    All A_m = 0 by construction (axisymmetric).

Lopsided disk (m=1 perturbation, Jog 2002)
    Sigma(R, phi) = Sigma_0(R) [1 + epsilon_1 cos(phi - phi_1)]
    A_1(R) = epsilon_1  (constant, independent of R)
    A_m = 0 for m != 1

Barred disk (m=2 perturbation, Chequers et al. 2016)
    Sigma(R, phi) = Sigma_0(R) [1 + epsilon_2 cos(2 phi - 2 phi_2)]
    A_2(R) = epsilon_2
    A_m = 0 for m != 2

Logarithmic spiral (m=2 tightly wound)
    Sigma(R, phi) = Sigma_0(R) [1 + epsilon_s cos(2 phi - k ln R)]
    The Fourier amplitude A_2(R) в‰€ epsilon_s for all R (slowly varying
    pitch angle case; see Grand et al. 2015).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class SyntheticGalaxy:
    """
    Container for a synthetic galaxy particle distribution.

    Attributes
    ----------
    x, y : ndarray     Positions [kpc].
    mass : ndarray     Masses (equal-mass particles).
    name : str         Generator name.
    analytic_amps : dict
        {m: callable(R) -> A_m(R)} analytic amplitude functions.
    analytic_phases : dict
        {m: callable(R) -> Phi_m(R)} analytic phase functions [rad].
    params : dict      Generator parameters.
    """
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    mass: NDArray[np.float64]
    name: str
    analytic_amps: Dict[int, Callable]
    analytic_phases: Dict[int, Callable]
    params: dict

    def analytic_A(self, m: int, r: NDArray) -> NDArray:
        f = self.analytic_amps.get(m)
        return f(r) if f is not None else np.zeros_like(r)

    def analytic_Phi(self, m: int, r: NDArray) -> NDArray:
        f = self.analytic_phases.get(m)
        return f(r) if f is not None else np.zeros_like(r)


# ---------------------------------------------------------------------------
# Common helper
# ---------------------------------------------------------------------------

def _sample_exponential(
    n_particles: int,
    R_d: float,
    r_max: float,
    rng: np.random.Generator,
) -> Tuple[NDArray, NDArray]:
    """
    Sample (R, phi) from an exponential disk Sigma в€ќ exp(-R/R_d).

    Uses inverse CDF: the CDF of R is
        F(R) = 1 - (1 + R/R_d) exp(-R/R_d)   (normalised to Rв†’в€ћ)
    We use rejection sampling for simplicity, which is accurate for
    any truncation radius.
    """
    radii = []
    total_accepted = 0
    batch = max(n_particles * 4, 1000)
    while total_accepted < n_particles:
        R_try = rng.uniform(0.0, r_max, size=batch)
        u     = rng.uniform(0.0, 1.0,   size=batch)
        # pdf(R) в€ќ R exp(-R/R_d); normalise by max at R = R_d
        pdf = R_try * np.exp(-R_try / R_d)
        pdf_max = R_d * np.exp(-1.0)
        accepted = R_try[u * pdf_max < pdf]
        radii.append(accepted)
        total_accepted += len(accepted)
    R = np.concatenate(radii)[:n_particles]
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_particles)
    return R, phi


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def make_exponential_disk(
    n_particles: int = 100_000,
    R_d: float = 3.0,
    r_max: float = 15.0,
    seed: int = 42,
) -> SyntheticGalaxy:
    """
    Pure axisymmetric exponential disk.

    Analytic amplitudes: A_m = 0 for all m в‰Ґ 1.

    Parameters
    ----------
    n_particles : int   Number of equal-mass particles.
    R_d : float         Scale length [kpc].
    r_max : float       Truncation radius [kpc].
    seed : int          RNG seed.
    """
    rng = np.random.default_rng(seed)
    R, phi = _sample_exponential(n_particles, R_d, r_max, rng)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    mass = np.ones(n_particles) / n_particles

    return SyntheticGalaxy(
        x=x, y=y, mass=mass, name="exponential_disk",
        analytic_amps   = {m: (lambda r, _m=m: np.zeros_like(r)) for m in range(1, 5)},
        analytic_phases = {m: (lambda r, _m=m: np.zeros_like(r)) for m in range(1, 5)},
        params=dict(n_particles=n_particles, R_d=R_d, r_max=r_max),
    )


def make_lopsided_disk(
    n_particles: int = 200_000,
    R_d: float = 3.0,
    epsilon_1: float = 0.3,
    phi_1: float = 0.0,
    r_max: float = 15.0,
    seed: int = 42,
) -> SyntheticGalaxy:
    """
    Exponential disk with an m=1 azimuthal perturbation.

    The surface density is
        Sigma(R, phi) = Sigma_0(R) [1 + epsilon_1 cos(phi - phi_1)]
    so the analytic Fourier amplitude is A_1 = epsilon_1 everywhere,
    and the phase is Phi_1 = phi_1.

    Particles are generated by rejection sampling with the above
    probability density.

    Parameters
    ----------
    epsilon_1 : float   Lopsidedness amplitude (must be < 1).
    phi_1 : float       Phase angle of lopsided mode [rad].
    """
    if not 0.0 < epsilon_1 < 1.0:
        raise ValueError("epsilon_1 must be in (0, 1).")

    rng = np.random.default_rng(seed)
    # Sample base distribution then apply angular modulation by rejection
    result_x, result_y = [], []
    batch = max(n_particles * 4, 10_000)

    while sum(len(a) for a in result_x) < n_particles:
        R, phi = _sample_exponential(batch, R_d, r_max, rng)
        # Angular weight: [1 + epsilon_1 cos(phi - phi_1)] / (1 + epsilon_1)
        w_max = 1.0 + epsilon_1
        w     = 1.0 + epsilon_1 * np.cos(phi - phi_1)
        u     = rng.uniform(0.0, w_max, size=len(R))
        keep  = u < w
        result_x.append(R[keep] * np.cos(phi[keep]))
        result_y.append(R[keep] * np.sin(phi[keep]))

    x = np.concatenate(result_x)[:n_particles]
    y = np.concatenate(result_y)[:n_particles]
    mass = np.ones(n_particles) / n_particles

    return SyntheticGalaxy(
        x=x, y=y, mass=mass, name="lopsided_disk",
        analytic_amps={
            1: lambda r: np.full_like(r, epsilon_1),
            2: lambda r: np.zeros_like(r),
            3: lambda r: np.zeros_like(r),
            4: lambda r: np.zeros_like(r),
        },
        analytic_phases={
            1: lambda r: np.full_like(r, phi_1),
            2: lambda r: np.zeros_like(r),
        },
        params=dict(
            n_particles=n_particles, R_d=R_d, epsilon_1=epsilon_1,
            phi_1=phi_1, r_max=r_max,
        ),
    )


def make_barred_disk(
    n_particles: int = 200_000,
    R_d: float = 3.0,
    epsilon_2: float = 0.4,
    phi_2: float = 0.0,
    r_max: float = 15.0,
    seed: int = 42,
) -> SyntheticGalaxy:
    """
    Exponential disk with an m=2 bar perturbation.

    Sigma(R, phi) = Sigma_0(R) [1 + epsilon_2 cos(2 phi - 2 phi_2)]
    Analytic: A_2 = epsilon_2, Phi_2 = phi_2.

    Parameters
    ----------
    epsilon_2 : float   Bar amplitude (< 1).
    phi_2 : float       Bar position angle [rad].
    """
    if not 0.0 < epsilon_2 < 1.0:
        raise ValueError("epsilon_2 must be in (0, 1).")

    rng = np.random.default_rng(seed)
    result_x, result_y = [], []
    batch = max(n_particles * 4, 10_000)

    while sum(len(a) for a in result_x) < n_particles:
        R, phi = _sample_exponential(batch, R_d, r_max, rng)
        w_max = 1.0 + epsilon_2
        w     = 1.0 + epsilon_2 * np.cos(2.0 * phi - 2.0 * phi_2)
        u     = rng.uniform(0.0, w_max, size=len(R))
        keep  = u < w
        result_x.append(R[keep] * np.cos(phi[keep]))
        result_y.append(R[keep] * np.sin(phi[keep]))

    x = np.concatenate(result_x)[:n_particles]
    y = np.concatenate(result_y)[:n_particles]
    mass = np.ones(n_particles) / n_particles

    return SyntheticGalaxy(
        x=x, y=y, mass=mass, name="barred_disk",
        analytic_amps={
            1: lambda r: np.zeros_like(r),
            2: lambda r: np.full_like(r, epsilon_2),
            3: lambda r: np.zeros_like(r),
            4: lambda r: np.zeros_like(r),
        },
        analytic_phases={
            1: lambda r: np.zeros_like(r),
            2: lambda r: np.full_like(r, phi_2),
        },
        params=dict(
            n_particles=n_particles, R_d=R_d, epsilon_2=epsilon_2,
            phi_2=phi_2, r_max=r_max,
        ),
    )


def make_logarithmic_spiral(
    n_particles: int = 200_000,
    R_d: float = 3.0,
    epsilon_s: float = 0.3,
    pitch_angle_deg: float = 15.0,
    r_max: float = 15.0,
    seed: int = 42,
) -> SyntheticGalaxy:
    """
    Exponential disk with a two-armed logarithmic spiral.

    The spiral pattern is:
        Sigma(R, phi) = Sigma_0(R) [1 + epsilon_s cos(2 phi - k ln R)]

    where k = 2 / tan(pitch_angle) is the winding number.

    The analytic m=2 amplitude is A_2 = epsilon_s (approximately
    constant in R for a tightly-wound spiral; strictly valid only in
    the limit of constant pitch angle and negligible radial variation
    of the perturbation).

    Parameters
    ----------
    epsilon_s : float           Spiral arm amplitude.
    pitch_angle_deg : float     Pitch angle [degrees].  Typical: 5вЂ“25В°.
    """
    if not 0.0 < epsilon_s < 1.0:
        raise ValueError("epsilon_s must be in (0, 1).")

    pitch_angle = np.radians(pitch_angle_deg)
    k = 2.0 / np.tan(pitch_angle)   # winding number

    rng = np.random.default_rng(seed)
    result_x, result_y, result_R = [], [], []
    batch = max(n_particles * 4, 10_000)

    while sum(len(a) for a in result_x) < n_particles:
        R, phi = _sample_exponential(batch, R_d, r_max, rng)
        R = np.clip(R, 0.01, r_max)  # avoid log(0)
        w_max = 1.0 + epsilon_s
        w     = 1.0 + epsilon_s * np.cos(2.0 * phi - k * np.log(R))
        u     = rng.uniform(0.0, w_max, size=len(R))
        keep  = u < w
        result_x.append(R[keep] * np.cos(phi[keep]))
        result_y.append(R[keep] * np.sin(phi[keep]))
        result_R.append(R[keep])

    x = np.concatenate(result_x)[:n_particles]
    y = np.concatenate(result_y)[:n_particles]
    mass = np.ones(n_particles) / n_particles

    # Analytic phase varies with R: Phi_2(R) = (k/2) ln R
    return SyntheticGalaxy(
        x=x, y=y, mass=mass, name="logarithmic_spiral",
        analytic_amps={
            1: lambda r: np.zeros_like(r),
            2: lambda r: np.full_like(r, epsilon_s),
            3: lambda r: np.zeros_like(r),
            4: lambda r: np.zeros_like(r),
        },
        analytic_phases={
            1: lambda r: np.zeros_like(r),
            2: lambda r: 0.5 * k * np.log(np.clip(r, 0.01, None)),
        },
        params=dict(
            n_particles=n_particles, R_d=R_d, epsilon_s=epsilon_s,
            pitch_angle_deg=pitch_angle_deg, k=k, r_max=r_max,
        ),
    )
