"""
systematic_error_propagation.py
================================

Phase 3.2 #15 — Systematic error propagation on a 30-galaxy TNG50 subsample.

For each of the pipeline's discretionary choices, we re-run the Fourier
decomposition on 30 real TNG50 galaxies with an alternative setting and
measure the resulting shift in <A_1>.  This gives \emph{real}, not
synthetic, systematic uncertainties for Table 2 of the paper.

Alternatives tested:
    Aperture:      [1.5, 2.5] R_d (fiducial)  vs  [1.0, 3.0] R_d
    Radial bins:   40 (fiducial)              vs  20  and  80
    Height cut:    |z|<3 kpc (fiducial)       vs  |z|<2 and |z|<5 kpc

For each galaxy we record the fiducial A_1 and all alternative A_1 values.
The fractional shift |ΔA_1/A_1| is then aggregated over the 30-galaxy
sample.  The final systematic uncertainty for each choice is the
median of the per-galaxy fractional shifts.

Runtime: 30 galaxies × ~6 pipeline calls per galaxy = 180 runs.
Expected wall time: 3-8 minutes.

Usage
-----
    python systematic_error_propagation.py \\
        --n-galaxies 30 \\
        --output validation/systematics

Outputs:
    validation/systematics/systematic_table.csv
    validation/systematics/systematic_summary.md
    validation/systematics/fig12_systematic_uncertainties.pdf

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

from tnggalaxylab.core.center import Centering
from tnggalaxylab.fourier import compute_fourier, global_lopsidedness


# =============================================================================
# Configuration
# =============================================================================
HUBBLE_H = 0.6774
MASS_UNIT_MSUN = 1e10 / HUBBLE_H
LENGTH_UNIT_KPC = 1.0 / HUBBLE_H

# Fiducial pipeline settings
FIDUCIAL = {
    "aperture_lo": 1.5,
    "aperture_hi": 2.5,
    "n_bins": 40,
    "z_cut_kpc": 3.0,
}

# Alternative settings to test (one at a time)
ALTERNATIVES = [
    # (Name, param overrides)
    ("Aperture [1.0, 3.0] R_d", {"aperture_lo": 1.0, "aperture_hi": 3.0}),
    ("Radial bins: 20",         {"n_bins": 20}),
    ("Radial bins: 80",         {"n_bins": 80}),
    ("Height cut |z|<2 kpc",     {"z_cut_kpc": 2.0}),
    ("Height cut |z|<5 kpc",     {"z_cut_kpc": 5.0}),
]


# =============================================================================
# Prepare centred, face-on disc particles
# =============================================================================
def prepare_particles(cutout_path: Path, z_cut_kpc: float
                       ) -> tuple[np.ndarray, np.ndarray] | None:
    """Load, centre, orient face-on, and height-cut. Return (pos, mass)."""
    with h5py.File(cutout_path, "r") as f:
        pos = f["PartType4/Coordinates"][:]
        mass = f["PartType4/Masses"][:]
        vel = f["PartType4/Velocities"][:]

    mass_phys = mass * MASS_UNIT_MSUN
    pos_kpc = pos * LENGTH_UNIT_KPC

    # Centring
    centering = Centering()
    centre = centering.shrinking_sphere(pos_kpc, mass_phys)
    pos_c = pos_kpc - centre

    # Face-on orientation (inner 10 kpc L)
    r = np.linalg.norm(pos_c, axis=1)
    inner = r < 10.0
    if inner.sum() < 100:
        return None
    L = np.cross(pos_c[inner], vel[inner] * mass_phys[inner, None]).sum(axis=0)
    L_hat = L / np.linalg.norm(L)
    z_hat = np.array([0.0, 0.0, 1.0])
    v = np.cross(L_hat, z_hat)
    s = np.linalg.norm(v)
    c = np.dot(L_hat, z_hat)
    if s < 1e-8:
        R = np.eye(3) if c > 0 else np.diag([1, 1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    pos_face = pos_c @ R.T

    # Height cut
    disc_mask = np.abs(pos_face[:, 2]) < z_cut_kpc
    if disc_mask.sum() < 500:
        return None

    return pos_face[disc_mask], mass_phys[disc_mask]


# =============================================================================
# Run Fourier with given settings
# =============================================================================
def run_with_settings(pos: np.ndarray, mass: np.ndarray,
                       R_d: float, settings: dict) -> float:
    """Run pipeline with given parameter settings, return A_1_literature."""
    profile = compute_fourier(
        pos[:, 0], pos[:, 1], mass,
        method="particles",
        r_in=0.0,
        r_out=15.0,
        n_bins=settings["n_bins"],
        m_max=2,
    )
    r_range = (settings["aperture_lo"] * R_d, settings["aperture_hi"] * R_d)
    gm = global_lopsidedness(profile, scale_length=R_d, r_range=r_range)
    return float(gm.A1_literature)


# =============================================================================
# Process a single galaxy: fiducial + all alternatives
# =============================================================================
def process_galaxy(subhalo_id: int, R_d: float,
                    cutout_dir: Path) -> dict | None:
    """For one galaxy: compute fiducial A_1 + each alternative.

    Returns dict of {alternative_name: A_1} plus 'fiducial'.
    """
    cutout_path = cutout_dir / f"cutout_{subhalo_id}.hdf5"
    if not cutout_path.exists():
        return None

    result = {"subhalo_id": subhalo_id, "R_d_kpc": R_d}

    # ---- Fiducial ----------------------------------------------------------
    prep = prepare_particles(cutout_path, FIDUCIAL["z_cut_kpc"])
    if prep is None:
        return None
    pos, mass = prep

    try:
        A1_fid = run_with_settings(pos, mass, R_d, FIDUCIAL)
    except Exception as e:
        return None
    result["A1_fiducial"] = A1_fid

    # ---- Alternatives ------------------------------------------------------
    for name, override in ALTERNATIVES:
        settings = dict(FIDUCIAL)
        settings.update(override)

        # If z_cut changed, reprepare particles
        if "z_cut_kpc" in override:
            prep2 = prepare_particles(cutout_path, settings["z_cut_kpc"])
            if prep2 is None:
                result[name] = np.nan
                continue
            pos_alt, mass_alt = prep2
        else:
            pos_alt, mass_alt = pos, mass

        try:
            A1_alt = run_with_settings(pos_alt, mass_alt, R_d, settings)
            result[name] = A1_alt
        except Exception:
            result[name] = np.nan

    return result


# =============================================================================
# Aggregate: compute fractional shifts and summary statistics
# =============================================================================
def summarise_results(results: list[dict]) -> pd.DataFrame:
    """Compute median fractional shift for each alternative."""
    df = pd.DataFrame(results)

    rows = []
    for name, _ in ALTERNATIVES:
        if name not in df.columns:
            continue
        fid = df["A1_fiducial"].values
        alt = df[name].values
        valid = ~(np.isnan(fid) | np.isnan(alt)) & (fid > 0)
        if valid.sum() < 5:
            continue
        frac_shift = np.abs(alt[valid] - fid[valid]) / fid[valid]
        rows.append({
            "alternative": name,
            "N_used": int(valid.sum()),
            "median_frac_shift": float(np.median(frac_shift)),
            "mean_frac_shift": float(np.mean(frac_shift)),
            "p16_frac_shift": float(np.percentile(frac_shift, 16)),
            "p84_frac_shift": float(np.percentile(frac_shift, 84)),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================
def plot_systematic(summary: pd.DataFrame, per_galaxy: pd.DataFrame,
                     outpath_pdf: Path, outpath_png: Path) -> None:
    """One-panel Figure 12 showing per-galaxy fractional shifts."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True)

    positions = np.arange(len(summary))
    labels = summary["alternative"].values

    # Box plot per alternative
    data_lists = []
    for name in labels:
        fid = per_galaxy["A1_fiducial"].values
        alt = per_galaxy[name].values
        valid = ~(np.isnan(fid) | np.isnan(alt)) & (fid > 0)
        frac_shift = np.abs(alt[valid] - fid[valid]) / fid[valid]
        data_lists.append(frac_shift * 100)  # convert to %

    bp = ax.boxplot(data_lists, positions=positions, widths=0.6,
                     patch_artist=True, showmeans=True,
                     meanprops={"marker": "D", "markerfacecolor": "white",
                                "markeredgecolor": "black", "markersize": 8},
                     medianprops={"color": "crimson", "linewidth": 2})

    for patch in bp["boxes"]:
        patch.set_facecolor("tab:blue")
        patch.set_alpha(0.45)

    # Add median value labels above each box
    for i, (name, frac) in enumerate(zip(labels, data_lists)):
        med = np.median(frac)
        ax.text(i - 0.28, med, f"{med:.1f}%", ha="right", va="center",
                fontsize=9, color="crimson", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="crimson", alpha=0.85))

    ax.set_xticks(positions)
    ax.set_xticklabels([l.replace("R_d", r"$R_{\rm d}$").replace(" [", "\n[").replace(": ", ":\n")
                         for l in labels], fontsize=8)
    ax.set_ylabel(r"$|\Delta A_1 / A_1|$ (%)")
    ax.set_title(f"Systematic uncertainty budget "
                  f"(N = {len(per_galaxy)} TNG50 galaxies)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

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
    parser.add_argument("--n-galaxies", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="validation/systematics")
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading catalog: {args.catalog}")
    cat = pd.read_csv(args.catalog)
    valid = cat[(cat["A1_bootstrap_mean"] > 0)
                & (cat["R_d_kpc"] > 0.5)
                & (cat["n_disk_particles"] > 500)].copy()
    print(f"Valid galaxies: {len(valid)}")

    n_take = min(args.n_galaxies, len(valid))
    sample = valid.sample(n=n_take, random_state=args.seed).reset_index(drop=True)
    print(f"Selected {n_take} galaxies for systematic analysis.\n")

    results = []
    cutout_dir = Path(args.cutout_dir)
    t0 = time.time()
    for i, row in sample.iterrows():
        result = process_galaxy(
            subhalo_id=int(row["subhalo_id"]),
            R_d=float(row["R_d_kpc"]),
            cutout_dir=cutout_dir,
        )
        if result is None:
            print(f"  [{i+1:3d}/{n_take}] SKIP subhalo {int(row['subhalo_id'])}")
            continue
        results.append(result)
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (n_take - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1:3d}/{n_take}] subhalo {int(row['subhalo_id']):7d}  "
              f"A_1_fid = {result.get('A1_fiducial', 0):.4f}  "
              f"({elapsed:.1f}s, ETA {eta:.0f}s)")

    runtime = time.time() - t0
    print(f"\nCompleted {len(results)} galaxies in {runtime:.1f}s "
          f"({runtime/60:.1f} min).")

    # Save per-galaxy CSV
    per_galaxy = pd.DataFrame(results)
    per_galaxy.to_csv(outdir / "per_galaxy_systematics.csv", index=False)

    # Aggregate summary
    summary = summarise_results(results)
    summary.to_csv(outdir / "systematic_table.csv", index=False)

    # Print summary
    print()
    print("=" * 70)
    print("SYSTEMATIC UNCERTAINTY BUDGET (median |ΔA_1/A_1|)")
    print("=" * 70)
    for _, row in summary.iterrows():
        print(f"  {row['alternative']:35s}  {100 * row['median_frac_shift']:5.2f}%  "
              f"(16-84%: [{100 * row['p16_frac_shift']:.1f}, "
              f"{100 * row['p84_frac_shift']:.1f}]%)")

    # Combined (quadrature)
    combined = np.sqrt(sum(row["median_frac_shift"]**2
                            for _, row in summary.iterrows()))
    print(f"  {'Combined (quadrature)':35s}  {100 * combined:5.2f}%")

    # Plot
    plot_systematic(summary, per_galaxy,
                     outdir / "fig12_systematic_uncertainties.pdf",
                     outdir / "fig12_systematic_uncertainties.png")

    # Report
    with open(outdir / "systematic_summary.md", "w", encoding="utf-8") as f:
        f.write("# Systematic Uncertainty Budget — Phase 3.2 #15\n\n")
        f.write(f"Sample: {len(per_galaxy)} random TNG50 galaxies, "
                f"processed with the fiducial pipeline and 5 alternative "
                f"settings.\n\n")
        f.write("| Alternative | N used | Median |ΔA_1/A_1| | 16-84% range |\n")
        f.write("|---|---|---|---|\n")
        for _, row in summary.iterrows():
            f.write(f"| {row['alternative']} | {row['N_used']} | "
                    f"{100 * row['median_frac_shift']:.2f}% | "
                    f"[{100 * row['p16_frac_shift']:.1f}, "
                    f"{100 * row['p84_frac_shift']:.1f}]% |\n")
        f.write(f"| **Combined (quadrature)** | -- | "
                f"**{100 * combined:.2f}%** | -- |\n\n")

    print(f"\nAll outputs in: {outdir}")


if __name__ == "__main__":
    main()
