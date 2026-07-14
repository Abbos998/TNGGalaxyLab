"""
analyze_rotation_curve.py
=========================

Phase 2.4 — Rotation-curve analysis of the 915-galaxy TNG50 catalogue.

The pipeline records V_max (peak circular velocity) and R_V_max for each
galaxy, but the reference paper draft does not use these dynamical
tracers in the analysis.  This script closes that gap by producing
three publication-quality panels:

    Panel A:  <A_1> vs V_max         (lopsidedness vs mass)
    Panel B:  <A_2> vs V_max         (bar/spiral strength vs rotation)
    Panel C:  Q_disc vs stellar mass (disc-stability proxy)

Where Q_disc = V_max^2 / (G * M_star / R_V_max)  — a dimensionless
stability proxy that combines rotational support (numerator) against
self-gravity (denominator).  Q >> 1 indicates a rotation-dominated,
stable disc; Q ~ 1 indicates a disc close to marginal stability;
Q < 1 marks systems where self-gravity dominates the local dynamical
balance (typically strong-bar or merger candidates).

Usage
-----
    python analyze_rotation_curve.py

Outputs (written to batch_tng50/paper_figures/):
    fig9_rotation_analysis.pdf, .png
    rotation_analysis_stats.md
    rotation_analysis.csv

Author: Abbos Omonov et al. (2026)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# Configuration
# =============================================================================
G_KPC_KMS2_PER_MSUN = 4.302e-6   # Newton's G in [kpc (km/s)^2 / Msun]

# Mass bins (log_10(M_star / Msun))
V_MAX_BINS = np.array([50, 100, 130, 160, 200, 260])   # km/s edges


# =============================================================================
# Load catalogue
# =============================================================================
def load_catalog(path: str) -> pd.DataFrame:
    """Load TNG50 catalogue; keep only rows with valid rotation-curve data."""
    cat = pd.read_csv(path)
    n_all = len(cat)
    # Filter rows with sensible V_max and R_V_max
    cat = cat[(cat["v_max_kms"] > 30)          # exclude ultra-slow rotators
              & (cat["v_max_kms"] < 500)
              & (cat["R_v_max_kpc"] > 0.5)
              & (cat["R_v_max_kpc"] < 50)
              & (cat["M_star_Msun"] > 1e9)
              & (cat["A1_bootstrap_mean"] > 0)
              & (cat["A2_bootstrap_mean"] > 0)]
    print(f"Loaded {n_all} rows, {len(cat)} pass filter.")
    return cat.reset_index(drop=True)


# =============================================================================
# Compute stability parameter
# =============================================================================
def compute_stability_Q(cat: pd.DataFrame) -> np.ndarray:
    """Compute Q = V_max^2 * R_V_max / (G * M_star).

    Physical interpretation
    -----------------------
    * The numerator V_max^2 * R_V_max is proportional to the kinetic
      energy per unit mass, on the scale where the rotation curve
      peaks.
    * The denominator G * M_star (with R_V_max implicitly in the
      Keplerian form) is the potential energy per unit mass from
      the stellar component alone.
    * Q > 1 → rotation dominates over stellar self-gravity
      (stable, halo-supported disc).
    * Q ~ 1 → marginal stability; bars and lopsidedness expected.
    * Q < 1 → strong self-gravity, prone to global instabilities.
    """
    v_max = cat["v_max_kms"].values          # km/s
    R_v = cat["R_v_max_kpc"].values          # kpc
    M_star = cat["M_star_Msun"].values       # Msun
    Q = (v_max ** 2 * R_v) / (G_KPC_KMS2_PER_MSUN * M_star)
    return Q


# =============================================================================
# Running median in bins
# =============================================================================
def running_median(x: np.ndarray, y: np.ndarray, bins: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return bin-centres, medians, 16%, 84% percentile of y within bins of x.

    Empty bins are dropped.
    """
    centres, med, p16, p84 = [], [], [], []
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() < 5:
            continue
        centres.append(0.5 * (bins[i] + bins[i + 1]))
        y_bin = y[mask]
        med.append(np.median(y_bin))
        p16.append(np.percentile(y_bin, 16))
        p84.append(np.percentile(y_bin, 84))
    return (np.array(centres), np.array(med),
            np.array(p16), np.array(p84))


# =============================================================================
# Plotting: 3-panel Figure 9
# =============================================================================
def plot_rotation_analysis(cat: pd.DataFrame, Q: np.ndarray,
                            outpath_pdf: Path,
                            outpath_png: Path) -> dict:
    """Produce the 3-panel Figure 9 and return summary statistics."""
    v_max = cat["v_max_kms"].values
    A1 = cat["A1_bootstrap_mean"].values
    A2 = cat["A2_bootstrap_mean"].values
    logM = np.log10(cat["M_star_Msun"].values)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    # ------------------------------------------------------------------------
    # Panel A: A_1 vs V_max
    # ------------------------------------------------------------------------
    ax = axes[0]
    ax.scatter(v_max, A1, c=logM, cmap="viridis", s=20, alpha=0.55,
               edgecolor="none", rasterized=True)

    xc, m, p16, p84 = running_median(v_max, A1, V_MAX_BINS)
    ax.plot(xc, m, "-", color="crimson", lw=2.5, marker="D", markersize=9,
            markerfacecolor="white", markeredgecolor="crimson",
            markeredgewidth=1.5, zorder=5, label="Running median")
    ax.fill_between(xc, p16, p84, color="crimson", alpha=0.18,
                     label="16-84 percentile")
    ax.axhline(0.1, ls=":", color="grey", lw=1)
    ax.text(240, 0.11, r"$A_1 = 0.1$", color="grey", fontsize=8, ha="right")

    ax.set_xlabel(r"$V_{\rm max}$ [km s$^{-1}$]")
    ax.set_ylabel(r"$\langle A_1 \rangle$")
    ax.set_title("Panel A: Lopsidedness vs. rotation", fontsize=11)
    ax.set_xlim(30, 260)
    ax.set_ylim(0, 0.45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(vmin=logM.min(),
                                                    vmax=logM.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85,
                         label=r"$\log_{10}(M_\star / M_\odot)$")

    # ------------------------------------------------------------------------
    # Panel B: A_2 vs V_max
    # ------------------------------------------------------------------------
    ax = axes[1]
    ax.scatter(v_max, A2, c=logM, cmap="viridis", s=20, alpha=0.55,
               edgecolor="none", rasterized=True)

    xc, m, p16, p84 = running_median(v_max, A2, V_MAX_BINS)
    ax.plot(xc, m, "-", color="darkblue", lw=2.5, marker="D", markersize=9,
            markerfacecolor="white", markeredgecolor="darkblue",
            markeredgewidth=1.5, zorder=5, label="Running median")
    ax.fill_between(xc, p16, p84, color="darkblue", alpha=0.18,
                     label="16-84 percentile")
    ax.axhline(0.2, ls=":", color="grey", lw=1)
    ax.text(240, 0.21, r"$A_2 = 0.2$ (bar)", color="grey", fontsize=8,
             ha="right")

    ax.set_xlabel(r"$V_{\rm max}$ [km s$^{-1}$]")
    ax.set_ylabel(r"$\langle A_2 \rangle$")
    ax.set_title("Panel B: Bar strength vs. rotation", fontsize=11)
    ax.set_xlim(30, 260)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------------------
    # Panel C: Q_disc vs stellar mass
    # ------------------------------------------------------------------------
    ax = axes[2]
    # Colour by A_1
    sc = ax.scatter(logM, Q, c=A1, cmap="magma_r", s=20, alpha=0.7,
                     edgecolor="none", vmin=0, vmax=0.3, rasterized=True)

    # Show median trend in mass bins
    logM_bins = np.array([9.4, 9.7, 10.0, 10.3, 10.6, 11.0])
    xc, m, p16, p84 = running_median(logM, Q, logM_bins)
    ax.plot(xc, m, "-", color="black", lw=2.5, marker="D", markersize=9,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.5, zorder=5, label="Running median")
    ax.fill_between(xc, p16, p84, color="black", alpha=0.15,
                     label="16-84 percentile")

    ax.axhline(1.0, ls="--", color="red", lw=1.5, alpha=0.8, zorder=4)
    ax.text(10.9, 1.05, r"$Q = 1$", bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="grey", alpha=0.85), color="red", fontsize=8, ha="right")

    ax.set_xlabel(r"$\log_{10}(M_\star / M_\odot)$")
    ax.set_ylabel(r"$Q_{\rm disc} = V_{\rm max}^{2}\,R_{V_{\rm max}} / (G M_\star)$")
    ax.set_yscale("log")
    ax.set_title("Panel C: Disc stability proxy", fontsize=11)
    ax.set_xlim(9.4, 11.0)
    ax.set_ylim(0.1, 50)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    cbar2 = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85,
                          label=r"$\langle A_1 \rangle$")

    fig.savefig(outpath_pdf, bbox_inches="tight")
    fig.savefig(outpath_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure written:\n  {outpath_pdf}\n  {outpath_png}")

    # --------------------------------------------------------------------------
    # Return summary statistics
    # --------------------------------------------------------------------------
    return summarise(cat, Q)


# =============================================================================
# Summary statistics (Spearman + bin medians)
# =============================================================================
def summarise(cat: pd.DataFrame, Q: np.ndarray) -> dict:
    """Compute Spearman ranks and bin medians for the paper."""
    v_max = cat["v_max_kms"].values
    A1 = cat["A1_bootstrap_mean"].values
    A2 = cat["A2_bootstrap_mean"].values
    logM = np.log10(cat["M_star_Msun"].values)

    # Spearman correlations
    rho_A1_vmax, p_A1_vmax = stats.spearmanr(A1, v_max)
    rho_A2_vmax, p_A2_vmax = stats.spearmanr(A2, v_max)
    rho_Q_M, p_Q_M = stats.spearmanr(Q, logM)
    rho_Q_A1, p_Q_A1 = stats.spearmanr(Q, A1)

    # Fraction with Q < 1 by mass bin
    dwarf = logM < 10.0
    mid = (logM >= 10.0) & (logM < 10.5)
    massive = logM >= 10.5
    frac_Q1_dwarf = float(np.mean(Q[dwarf] < 1.0)) if dwarf.sum() else np.nan
    frac_Q1_mid = float(np.mean(Q[mid] < 1.0)) if mid.sum() else np.nan
    frac_Q1_massive = float(np.mean(Q[massive] < 1.0)) if massive.sum() else np.nan

    return {
        "N_analyzed": len(cat),
        "rho_A1_Vmax": rho_A1_vmax,
        "p_A1_Vmax": p_A1_vmax,
        "rho_A2_Vmax": rho_A2_vmax,
        "p_A2_Vmax": p_A2_vmax,
        "rho_Q_logM": rho_Q_M,
        "p_Q_logM": p_Q_M,
        "rho_Q_A1": rho_Q_A1,
        "p_Q_A1": p_Q_A1,
        "Q_median": float(np.median(Q)),
        "Q_mean": float(np.mean(Q)),
        "frac_Q_lt_1_dwarf": frac_Q1_dwarf,
        "frac_Q_lt_1_intermediate": frac_Q1_mid,
        "frac_Q_lt_1_massive": frac_Q1_massive,
    }


# =============================================================================
# Text report
# =============================================================================
def write_report(stats_dict: dict, outpath: Path) -> None:
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("# Rotation Curve Analysis — Phase 2.4\n\n")
        f.write(f"**Sample**: {stats_dict['N_analyzed']} galaxies with "
                "valid V_max and R_V_max\n\n")

        f.write("## Spearman rank correlations\n\n")
        f.write("| Relation | ρ (Spearman) | p-value |\n")
        f.write("|---|---|---|\n")
        f.write(f"| A_1 vs V_max | {stats_dict['rho_A1_Vmax']:+.3f} | "
                f"{stats_dict['p_A1_Vmax']:.2e} |\n")
        f.write(f"| A_2 vs V_max | {stats_dict['rho_A2_Vmax']:+.3f} | "
                f"{stats_dict['p_A2_Vmax']:.2e} |\n")
        f.write(f"| Q vs log_10(M_*) | {stats_dict['rho_Q_logM']:+.3f} | "
                f"{stats_dict['p_Q_logM']:.2e} |\n")
        f.write(f"| Q vs A_1 | {stats_dict['rho_Q_A1']:+.3f} | "
                f"{stats_dict['p_Q_A1']:.2e} |\n\n")

        f.write("## Q_disc statistics\n\n")
        f.write(f"* Median Q = {stats_dict['Q_median']:.2f}\n")
        f.write(f"* Mean Q   = {stats_dict['Q_mean']:.2f}\n\n")

        f.write("## Fraction with Q < 1 (potentially unstable discs)\n\n")
        f.write(f"* Dwarf (log_M < 10.0):        "
                f"{100 * stats_dict['frac_Q_lt_1_dwarf']:.1f}%\n")
        f.write(f"* Intermediate (10.0-10.5):    "
                f"{100 * stats_dict['frac_Q_lt_1_intermediate']:.1f}%\n")
        f.write(f"* Massive (log_M >= 10.5):     "
                f"{100 * stats_dict['frac_Q_lt_1_massive']:.1f}%\n")

    print(f"Report written to: {outpath}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="batch_tng50/batch_output/catalog.csv")
    parser.add_argument("--output", default="batch_tng50/paper_figures")
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading catalog: {args.catalog}")
    cat = load_catalog(args.catalog)

    print(f"\nComputing Q_disc for {len(cat)} galaxies...")
    Q = compute_stability_Q(cat)
    cat["Q_disc"] = Q

    print("\nGenerating Figure 9 (rotation analysis)...")
    stats_dict = plot_rotation_analysis(
        cat, Q,
        outdir / "fig9_rotation_analysis.pdf",
        outdir / "fig9_rotation_analysis.png",
    )

    # Print summary to console
    print()
    print("=" * 70)
    print("ROTATION ANALYSIS SUMMARY")
    print("=" * 70)
    for k, v in stats_dict.items():
        if isinstance(v, float):
            if "p_" in k:
                print(f"  {k:35s} = {v:.3e}")
            else:
                print(f"  {k:35s} = {v:+.4f}")
        else:
            print(f"  {k:35s} = {v}")

    # Write CSV + report
    cat.to_csv(outdir / "rotation_analysis.csv", index=False)
    write_report(stats_dict, outdir / "rotation_analysis_stats.md")

    print(f"\nAll outputs in: {outdir}")


if __name__ == "__main__":
    main()
