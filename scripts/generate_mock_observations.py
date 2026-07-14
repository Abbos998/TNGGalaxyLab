"""
generate_mock_observations.py
=============================

Phase 3.2 #9 — Mock S⁴G-like observations of TNG50 galaxies.

This script produces a Zaritsky+2013-style observational analogue of a
100-galaxy TNG50 subsample by degrading the raw particle data through:

    1. Random inclination sampling (isotropic in cos i, i = 0-60 deg)
    2. Random distance sampling (uniform in [10, 40] Mpc, matching S⁴G)
    3. Projection to 2D mass surface density map
    4. Spitzer 3.6 μm PSF convolution (FWHM = 2 arcsec at target distance)
    5. Photon-limited Gaussian noise addition (mock SNR ~ 30 per pixel)
    6. Threshold-based particle re-sampling (mimic detection limit)
    7. Re-running the standard Fourier pipeline on the "observed" system

The result is a mock catalogue that can be compared to Zaritsky+2013
S⁴G measurements on genuinely equal footing --- both are magnitude-
limited, PSF-broadened, noise-degraded surface-density measurements.

Comparison with the raw TNG50 A₁ then quantifies the observational-
degradation systematic in disc-lopsidedness measurements.

Usage
-----
    python generate_mock_observations.py \\
        --n-galaxies 100 \\
        --n-inclinations 3 \\
        --output validation/mock

Outputs
-------
    validation/mock/mock_catalog.csv         --- one row per (galaxy, mock)
    validation/mock/fig11_mock_comparison.pdf --- 3-panel comparison
    validation/mock/MOCK_REPORT.md            --- text summary

Author: Abbos Omonov et al. (2026)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats

from tnggalaxylab.core.center import Centering
from tnggalaxylab.fourier import compute_fourier, global_lopsidedness


# =============================================================================
# Configuration
# =============================================================================
# TNG50 code units
HUBBLE_H = 0.6774
MASS_UNIT_MSUN = 1e10 / HUBBLE_H       # 10^10 Msun/h → Msun
LENGTH_UNIT_KPC = 1.0 / HUBBLE_H       # kpc/h → kpc  (z=0)

# S⁴G / Spitzer 3.6 μm specifications
PSF_FWHM_ARCSEC = 2.0                   # Spitzer IRAC 3.6 μm PSF
PIXEL_SCALE_ARCSEC = 0.75               # IRAC ch1 native pixel scale
SNR_TARGET = 30.0                       # Typical S⁴G SNR per pixel in disc

# Mock observation parameters
DISTANCE_MPC_MIN = 10.0
DISTANCE_MPC_MAX = 40.0
INCLINATION_DEG_MIN = 0.0
INCLINATION_DEG_MAX = 60.0

# Image and analysis parameters
IMAGE_HALF_WIDTH_KPC = 20.0             # ±20 kpc field of view
IMAGE_N_PIX = 256                       # 256×256 image
DISC_Z_CUT_KPC = 3.0                    # Height cut for face-on selection


# =============================================================================
# Physical conversions
# =============================================================================
def kpc_to_arcsec(kpc: float, distance_mpc: float) -> float:
    """Angular size in arcsec of physical size in kpc at given distance."""
    return (kpc / (distance_mpc * 1000)) * 206265.0


def arcsec_to_kpc(arcsec: float, distance_mpc: float) -> float:
    """Physical size in kpc of angular size in arcsec at given distance."""
    return (arcsec / 206265.0) * (distance_mpc * 1000)


# =============================================================================
# Load and prepare particles for a galaxy
# =============================================================================
def load_and_prepare_particles(cutout_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load stellar particles, centre them, orient face-on.

    Returns
    -------
    pos_centred : (N, 3) array in kpc, disc plane is xy
    mass : (N,) array in Msun
    """
    with h5py.File(cutout_path, "r") as f:
        pos = f["PartType4/Coordinates"][:]        # kpc/h comoving
        mass = f["PartType4/Masses"][:]            # 10^10 Msun/h
        vel = f["PartType4/Velocities"][:]         # km/s * sqrt(a)

    # Convert to physical Msun and kpc
    mass_phys = mass * MASS_UNIT_MSUN
    pos_kpc = pos * LENGTH_UNIT_KPC

    # Iterative shrinking-sphere centring
    centering = Centering()
    centre = centering.shrinking_sphere(pos_kpc, mass_phys)
    pos_centred = pos_kpc - centre

    # Compute angular momentum inside 10 kpc → face-on orientation
    r = np.linalg.norm(pos_centred, axis=1)
    inner = r < 10.0
    if inner.sum() < 100:
        return None, None
    L = np.cross(pos_centred[inner], vel[inner] * mass_phys[inner, None])
    L_tot = L.sum(axis=0)
    L_hat = L_tot / np.linalg.norm(L_tot)

    # Rotate so L_hat → z_hat
    z_hat = np.array([0.0, 0.0, 1.0])
    v = np.cross(L_hat, z_hat)
    s = np.linalg.norm(v)
    c = np.dot(L_hat, z_hat)
    if s < 1e-8:
        R = np.eye(3) if c > 0 else np.diag([1, 1, -1])
    else:
        v_x = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        R = np.eye(3) + v_x + v_x @ v_x * ((1 - c) / (s * s))

    pos_faceon = pos_centred @ R.T

    # Height cut: |z| < 3 kpc
    disc_mask = np.abs(pos_faceon[:, 2]) < DISC_Z_CUT_KPC
    return pos_faceon[disc_mask], mass_phys[disc_mask]


# =============================================================================
# Apply inclination
# =============================================================================
def apply_inclination(pos: np.ndarray, inclination_deg: float) -> np.ndarray:
    """Tilt the disc by inclination (rotation about x-axis).

    Face-on i = 0, edge-on i = 90.  Returns 3D positions after tilt.
    """
    i_rad = np.deg2rad(inclination_deg)
    c, s = np.cos(i_rad), np.sin(i_rad)
    R = np.array([[1, 0, 0],
                    [0, c, -s],
                    [0, s, c]])
    return pos @ R.T


# =============================================================================
# Project to 2D image (mass surface density map)
# =============================================================================
def make_mass_image(pos: np.ndarray, mass: np.ndarray,
                     half_width_kpc: float = IMAGE_HALF_WIDTH_KPC,
                     n_pix: int = IMAGE_N_PIX) -> np.ndarray:
    """Project (x, y) positions onto a 2D mass map."""
    edges = np.linspace(-half_width_kpc, half_width_kpc, n_pix + 1)
    H, _, _ = np.histogram2d(pos[:, 0], pos[:, 1],
                              bins=[edges, edges], weights=mass)
    return H, edges


# =============================================================================
# PSF convolution (Gaussian approximation to Spitzer IRAC ch1)
# =============================================================================
def apply_psf(image: np.ndarray, psf_sigma_pix: float) -> np.ndarray:
    """Convolve image with Gaussian PSF of given sigma in pixels."""
    return ndimage.gaussian_filter(image, sigma=psf_sigma_pix, mode="constant")


# =============================================================================
# Add photon-limited noise
# =============================================================================
def add_noise(image: np.ndarray, snr_target: float, rng: np.random.Generator
               ) -> np.ndarray:
    """Add Gaussian noise scaled so peak SNR ~ snr_target."""
    peak = image.max()
    if peak == 0:
        return image
    noise_sigma = peak / snr_target
    noisy = image + rng.normal(0, noise_sigma, size=image.shape)
    noisy = np.maximum(noisy, 0)  # detector floor
    return noisy


# =============================================================================
# Convert observed image back to (x, y, mass) for pipeline
# =============================================================================
def image_to_particles(image: np.ndarray, edges: np.ndarray,
                        detection_threshold: float = 3.0,
                        noise_level: float | None = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert 2D image back to (x, y, mass) pseudo-particles for pipeline.

    Each pixel above detection threshold becomes a particle at the pixel
    centre with mass equal to the pixel value.  This mimics observing a
    surface-brightness map and extracting a mass map from it (as done in
    S⁴G analyses using stellar-mass-to-light conversions).
    """
    n_pix = image.shape[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    XX, YY = np.meshgrid(centres, centres, indexing="ij")

    # Detection threshold: relative to noise floor
    if noise_level is None:
        # Estimate from image std
        noise_level = np.std(image[image > 0]) / 5.0
    thresh = detection_threshold * noise_level

    detected = image > thresh
    return XX[detected], YY[detected], image[detected]


# =============================================================================
# Run pipeline on mock observation
# =============================================================================
def analyze_mock(x: np.ndarray, y: np.ndarray, mass: np.ndarray,
                  R_d_prior: float) -> tuple[float, float]:
    """Run Fourier decomposition + global amplitude on mock 'particles'.

    Returns (A_1, A_2) in the R&Z95 aperture at [1.5, 2.5] R_d.
    """
    if len(x) < 500:
        return np.nan, np.nan
    profile = compute_fourier(x, y, mass,
                               method="particles",
                               r_in=0.0, r_out=15.0,
                               n_bins=40, m_max=2)
    gm = global_lopsidedness(profile, scale_length=R_d_prior)
    return float(gm.A1_literature), float(gm.A2_literature)


# =============================================================================
# Process a single galaxy — multiple mock inclinations
# =============================================================================
def process_galaxy_mock(subhalo_id: int, R_d: float, A1_original: float,
                        cutout_dir: Path,
                        n_inclinations: int,
                        rng: np.random.Generator) -> list[dict]:
    """Generate n_inclinations mock observations of one galaxy.

    Returns a list of dicts (one per mock realisation).
    """
    cutout_path = cutout_dir / f"cutout_{subhalo_id}.hdf5"
    if not cutout_path.exists():
        return []

    # Load particles once
    pos_faceon, mass = load_and_prepare_particles(cutout_path)
    if pos_faceon is None or len(pos_faceon) < 500:
        return []

    results = []
    for k in range(n_inclinations):
        # Random inclination & distance
        cos_i = rng.uniform(np.cos(np.deg2rad(INCLINATION_DEG_MAX)), 1.0)
        inc_deg = float(np.rad2deg(np.arccos(cos_i)))
        distance_mpc = float(rng.uniform(DISTANCE_MPC_MIN, DISTANCE_MPC_MAX))

        # Tilt the disc
        pos_tilted = apply_inclination(pos_faceon, inc_deg)

        # PSF size at this distance
        # (Convert PSF from arcsec → kpc at the target distance)
        psf_fwhm_kpc = arcsec_to_kpc(PSF_FWHM_ARCSEC, distance_mpc)
        pixel_kpc = (2 * IMAGE_HALF_WIDTH_KPC) / IMAGE_N_PIX
        psf_sigma_pix = psf_fwhm_kpc / 2.355 / pixel_kpc  # FWHM → σ

        # Project to 2D mass image
        image, edges = make_mass_image(pos_tilted, mass)
        raw_peak = image.max()

        # Apply PSF and noise
        image_psf = apply_psf(image, psf_sigma_pix)
        image_noisy = add_noise(image_psf, SNR_TARGET, rng)

        # Extract pseudo-particles and analyze
        x_p, y_p, m_p = image_to_particles(image_noisy, edges,
                                             detection_threshold=3.0)
        A1_mock, A2_mock = analyze_mock(x_p, y_p, m_p, R_d_prior=R_d)

        results.append({
            "subhalo_id": subhalo_id,
            "R_d_kpc": R_d,
            "A1_original": A1_original,
            "inc_deg": inc_deg,
            "distance_mpc": distance_mpc,
            "psf_fwhm_kpc": psf_fwhm_kpc,
            "psf_sigma_pix": psf_sigma_pix,
            "n_pixels_detected": int((image_noisy > 0).sum()),
            "A1_mock": A1_mock,
            "A2_mock": A2_mock,
            "A1_delta": A1_mock - A1_original,
        })
    return results


# =============================================================================
# Plot: 3-panel Figure 11
# =============================================================================
def plot_mock_comparison(mock_df: pd.DataFrame, outpath_pdf: Path,
                          outpath_png: Path) -> None:
    """3-panel comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    # Discard failed mocks and quality-cut extreme outliers (A_1<1)
    good = mock_df.dropna(subset=["A1_mock", "A1_original"])
    n_before = len(good)
    good = good[(good["A1_mock"] < 1.0) & (good["A1_original"] < 1.0)]
    n_dropped = n_before - len(good)
    if n_dropped > 0:
        print(f"  Quality cut: dropped {n_dropped} mocks with A_1 > 1")

    # ------------------------------------------------------------------------
    # Panel A: A1_mock vs A1_original scatter
    # ------------------------------------------------------------------------
    ax = axes[0]
    sc = ax.scatter(good["A1_original"], good["A1_mock"],
                     c=good["inc_deg"], cmap="viridis", s=25, alpha=0.7,
                     edgecolor="none", rasterized=True)
    # 1:1 line
    lim = max(good["A1_original"].max(), good["A1_mock"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.5, label="1:1")

    ax.set_xlabel(r"$\langle A_1 \rangle$ (raw TNG50)")
    ax.set_ylabel(r"$\langle A_1 \rangle$ (mock observation)")
    ax.set_title(r"Panel A: Mock vs. raw $A_1$", fontsize=11)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=ax, label="Inclination [deg]",
                  pad=0.02, shrink=0.85)

    # ------------------------------------------------------------------------
    # Panel B: Bias vs inclination
    # ------------------------------------------------------------------------
    ax = axes[1]
    ax.scatter(good["inc_deg"], good["A1_delta"], alpha=0.5, s=20,
                edgecolor="none", color="tab:orange", rasterized=True)

    # Running median in inclination bins
    inc_bins = np.array([0, 15, 30, 45, 60])
    med, xc = [], []
    for i in range(len(inc_bins) - 1):
        mask = (good["inc_deg"] >= inc_bins[i]) & (good["inc_deg"] < inc_bins[i+1])
        if mask.sum() >= 3:
            med.append(good.loc[mask, "A1_delta"].median())
            xc.append(0.5 * (inc_bins[i] + inc_bins[i+1]))
    if med:
        ax.plot(xc, med, "-D", color="crimson", lw=2, markersize=8,
                 markerfacecolor="white", label="Running median")

    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_xlabel("Inclination [degrees]")
    ax.set_ylabel(r"$\Delta A_1$ (mock $-$ raw)")
    ax.set_title("Panel B: Bias vs. inclination", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # ------------------------------------------------------------------------
    # Panel C: CDF comparison — raw vs mock
    # ------------------------------------------------------------------------
    ax = axes[2]
    for label, values, color, ls in [
        ("Raw TNG50", good["A1_original"].values, "tab:blue", "-"),
        ("Mock observations", good["A1_mock"].values, "tab:red", "--"),
    ]:
        vals = np.sort(values)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, ls, color=color, lw=2,
                 label=f"{label} (med={np.median(vals):.3f})")

    # KS test raw vs mock
    ks = stats.ks_2samp(good["A1_original"].values, good["A1_mock"].values)
    ax.text(0.03, 0.97,
             f"KS $D = {ks.statistic:.3f}$, $p = {ks.pvalue:.2e}$",
             transform=ax.transAxes, fontsize=9, ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.3",
                        facecolor="white", edgecolor="grey"))

    ax.set_xlabel(r"$\langle A_1 \rangle$")
    ax.set_ylabel("Cumulative distribution")
    ax.set_title("Panel C: Raw vs. mock CDF", fontsize=11)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.savefig(outpath_pdf, bbox_inches="tight")
    fig.savefig(outpath_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure written:\n  {outpath_pdf}\n  {outpath_png}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="batch_tng50/batch_output/catalog.csv")
    parser.add_argument("--cutout-dir", default="batch_tng50/batch_output/cutouts")
    parser.add_argument("--n-galaxies", type=int, default=100,
                        help="Number of galaxies to mock-observe")
    parser.add_argument("--n-inclinations", type=int, default=3,
                        help="Mock realisations per galaxy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="validation/mock")
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading catalog: {args.catalog}")
    cat = pd.read_csv(args.catalog)

    # Select a random subsample of N galaxies (mass-weighted)
    rng = np.random.default_rng(args.seed)
    # Only use galaxies that have valid A_1 and R_d
    valid = cat[(cat["A1_bootstrap_mean"] > 0)
                & (cat["R_d_kpc"] > 0.5)
                & (cat["n_disk_particles"] > 500)].copy()
    print(f"Valid galaxies with N_disk >= 500: {len(valid)}")

    n_take = min(args.n_galaxies, len(valid))
    sample = valid.sample(n=n_take, random_state=args.seed).reset_index(drop=True)
    print(f"Selected {n_take} galaxies for mock observation.\n")

    # Process each galaxy
    all_results = []
    cutout_dir = Path(args.cutout_dir)
    t0 = time.time()
    for i, row in sample.iterrows():
        results = process_galaxy_mock(
            subhalo_id=int(row["subhalo_id"]),
            R_d=float(row["R_d_kpc"]),
            A1_original=float(row["A1_bootstrap_mean"]),
            cutout_dir=cutout_dir,
            n_inclinations=args.n_inclinations,
            rng=rng,
        )
        all_results.extend(results)

        # Progress
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (n_take - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1:3d}/{n_take}] subhalo {int(row['subhalo_id']):7d}  "
              f"→ {len(results)} mocks  "
              f"({elapsed:.1f}s elapsed, ETA {eta:.0f}s)")

    runtime = time.time() - t0
    print(f"\nCompleted {len(all_results)} mocks in {runtime:.1f}s "
          f"({runtime/60:.1f} min).")

    # Save catalog
    mock_df = pd.DataFrame(all_results)
    mock_df.to_csv(outdir / "mock_catalog.csv", index=False)
    print(f"Mock catalog written: {outdir / 'mock_catalog.csv'}")

    # Summary
    valid = mock_df.dropna(subset=["A1_mock"])
    print(f"\n{'='*70}")
    print("MOCK OBSERVATIONS SUMMARY")
    print(f"{'='*70}")
    print(f"  Successful mocks: {len(valid)} / {len(mock_df)}")
    print(f"  Raw A_1  median : {valid['A1_original'].median():.4f}")
    print(f"  Mock A_1 median : {valid['A1_mock'].median():.4f}")
    print(f"  Bias median     : {valid['A1_delta'].median():+.4f}")
    print(f"  Bias mean       : {valid['A1_delta'].mean():+.4f}")

    # KS test raw vs mock
    ks = stats.ks_2samp(valid["A1_original"].values, valid["A1_mock"].values)
    print(f"  KS test raw vs mock: D = {ks.statistic:.4f}, p = {ks.pvalue:.3e}")

    # Plot
    plot_mock_comparison(mock_df,
                          outdir / "fig11_mock_comparison.pdf",
                          outdir / "fig11_mock_comparison.png")

    # Write report
    with open(outdir / "MOCK_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# Mock Observations Report\n\n")
        f.write(f"## Setup\n\n")
        f.write(f"* Galaxies: {n_take}\n")
        f.write(f"* Mock realisations per galaxy: {args.n_inclinations}\n")
        f.write(f"* Total mocks: {len(mock_df)}\n")
        f.write(f"* Successful (A_1 recovered): {len(valid)}\n")
        f.write(f"* Distance range: {DISTANCE_MPC_MIN}-{DISTANCE_MPC_MAX} Mpc\n")
        f.write(f"* Inclination range: {INCLINATION_DEG_MIN}-{INCLINATION_DEG_MAX} deg\n")
        f.write(f"* PSF FWHM (arcsec): {PSF_FWHM_ARCSEC}\n")
        f.write(f"* Target SNR: {SNR_TARGET}\n")
        f.write(f"* Runtime: {runtime:.1f}s ({runtime/60:.1f} min)\n\n")

        f.write(f"## Results\n\n")
        f.write(f"| Statistic | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Raw A_1  median | {valid['A1_original'].median():.4f} |\n")
        f.write(f"| Mock A_1 median | {valid['A1_mock'].median():.4f} |\n")
        f.write(f"| Bias median | {valid['A1_delta'].median():+.4f} |\n")
        f.write(f"| Bias mean | {valid['A1_delta'].mean():+.4f} |\n")
        f.write(f"| KS D | {ks.statistic:.4f} |\n")
        f.write(f"| KS p | {ks.pvalue:.3e} |\n")

    print(f"\nAll outputs in: {outdir}")


if __name__ == "__main__":
    main()
