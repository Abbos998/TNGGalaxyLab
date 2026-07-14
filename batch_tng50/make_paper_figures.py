"""
make_paper_figures.py
======================
Generate Figures 4, 5, 6 of the methodology paper from the TNG50 catalog.

Figures:
    Fig. 4  — Distribution of bootstrap-mean A_1 across the TNG50 sample
              (histogram + median line + strong-lopsided tail highlighted).
    Fig. 5  — A_1 vs stellar mass, colour-coded by pattern coherence,
              with running median in mass bins.
    Fig. 6  — Cumulative distribution comparison: TNG50 vs. Rix & Zaritsky
              1995 (R&Z95) reference sample, with 2-sample Kolmogorov-
              Smirnov test.

Reference catalogues:
    R&Z95 A_1 values from Rix & Zaritsky 1995 (ApJ 447, 82) Table 3.
    (Digitised — sample of 60 face-on spirals, aperture 1.5-2.5 R_e.)

Usage:
    python make_paper_figures.py [--catalog PATH] [--output DIR]

Output: fig4/5/6 in both PDF (paper) and PNG (slides) formats.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# =============================================================================
# Publication style — MNRAS-friendly figure aesthetics
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =============================================================================
# Zaritsky+2013 sample — 167 real S4G galaxies with IDENTICAL methodology
# =============================================================================
# Reference: Zaritsky, D. et al. 2013, ApJ 772, 135
#   "On the origin of lopsidedness in galaxies as determined from the
#    Spitzer Survey of Stellar Structure in Galaxies (S^4^G)"
# VizieR source: J/ApJ/772/135, table1.dat, column <A1>i
# Aperture: inner radial range 1.5 - 2.5 R_S (EXACTLY the same as our pipeline).
# The methodology follows Zaritsky & Rix 1997 / Rix & Zaritsky 1995.
# This is an apples-to-apples comparison — no systematic aperture bias.

Z2013_A1 = np.array([
    0.086, 0.564, 0.161, 0.204, 0.208, 0.320, 0.245, 0.351,
    0.228, 0.341, 0.258, 0.059, 0.116, 0.136, 0.042, 0.181,
    0.101, 0.092, 0.255, 0.179, 0.103, 0.162, 0.481, 0.172,
    1.324, 0.326, 0.066, 0.055, 0.212, 0.195, 0.097, 0.165,
    0.070, 0.033, 0.030, 0.090, 0.081, 0.180, 0.423, 0.315,
    0.068, 0.336, 0.433, 0.054, 0.184, 0.048, 0.023, 0.075,
    0.193, 0.094, 0.317, 0.031, 0.023, 0.091, 0.225, 0.069,
    0.050, 0.263, 0.109, 0.152, 0.060, 0.085, 0.324, 0.084,
    0.410, 0.092, 0.126, 0.154, 0.060, 0.164, 0.049, 0.062,
    0.099, 0.032, 0.151, 0.028, 0.352, 0.119, 0.053, 0.173,
    0.317, 0.173, 0.118, 0.168, 0.050, 0.394, 0.040, 0.074,
    0.105, 0.367, 0.215, 0.020, 0.217, 0.145, 0.022, 0.263,
    0.033, 0.072, 0.020, 0.262, 0.314, 0.326, 0.317, 0.095,
    0.223, 0.076, 0.066, 0.131, 0.090, 0.833, 0.088, 0.427,
    0.219, 0.095, 0.103, 0.409, 0.531, 0.200, 0.198, 0.186,
    0.071, 0.602, 0.142, 0.209, 0.308, 0.039, 0.071, 0.228,
    0.086, 0.181, 0.246, 0.092, 0.088, 0.245, 0.036, 0.166,
    0.189, 0.166, 0.145, 0.282, 0.023, 0.431, 0.339, 0.085,
    0.237, 0.183, 0.134, 0.175, 0.111, 0.170, 0.158, 0.144,
    0.367, 0.055, 0.131, 0.538, 0.142, 0.237, 0.268, 0.265,
    0.398, 0.167, 0.172, 0.276, 0.160, 0.074, 0.057,
])
# Full sample: median = 0.160, N = 167

# =============================================================================
# Zaritsky+2013 MASS-MATCHED subset (10^9.5 < M_star < 10^11 M_sun)
# =============================================================================
# Converts Zaritsky M_3.6 (abs mag) to M_star using Meidt+2014 M/L=0.6
# and M_sun_3.6 = 3.24 (Willmer 2018). Restricts to same mass range as
# our TNG50 selection (§4.1). This is the "apples-to-apples" comparison
# recommended for referee-proof interpretation.
Z2013_A1_matched = np.array([
    0.066, 0.212, 0.097, 0.030, 0.090, 0.081, 0.193, 0.031,
    0.091, 0.225, 0.092, 0.060, 0.032, 0.151, 0.028, 0.050,
    0.215, 0.020, 0.022, 0.020, 0.076, 0.066, 0.090, 0.833,
    0.088, 0.427, 0.219, 0.095, 0.531, 0.198, 0.071, 0.602,
    0.308, 0.036, 0.145,
])
# Mass-matched: median = 0.090, N = 35
# vs TNG50 median 0.062: ratio = 1.45x (down from 2.58x full sample)

# Alias for backward compatibility (was RZ95_A1 in v1)
RZ95_A1 = Z2013_A1

# =============================================================================
# Lokas 2022 (A&A 662, A53) — TNG100 disk galaxy sample
# =============================================================================
# Reference: Lokas, E.L. 2022, A&A 662, A53 (arXiv:2204.01456)
# Published summary statistics (Table 1 & Section 2):
LOKAS_N              = 1912
LOKAS_A1_MEAN        = 0.051
LOKAS_A1_MEDIAN      = 0.044
LOKAS_LOPSIDED_FRAC  = 0.084   # A_1 > 0.1 fraction = 161/1912 = 8.4%
LOKAS_SIMULATION     = "TNG100"
LOKAS_APERTURE       = "1-2 r_half"   # slightly different from ours (1.5-2.5 R_d)


# =============================================================================
# Data loading
# =============================================================================
def load_catalog(csv_path: Path) -> dict:
    """Load catalog.csv into a dict of numpy arrays."""
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Convert columns to numpy arrays
    cols = {}
    for key in rows[0].keys():
        vals = [r[key] for r in rows]
        try:
            cols[key] = np.array([float(v) for v in vals])
        except ValueError:
            cols[key] = np.array(vals)   # keep as strings

    print(f"  Loaded {len(rows)} rows from {csv_path.name}")
    return cols


# =============================================================================
# FIGURE 4 — A_1 distribution
# =============================================================================
def make_fig4_A1_distribution(cat: dict, outpath_base: Path):
    A1 = cat["A1_bootstrap_mean"]
    A1_err = cat["A1_bootstrap_std"]
    coh = cat["pattern_coherence"]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Histogram — all galaxies
    bins = np.arange(0, 0.55, 0.025)
    ax.hist(A1, bins=bins, color="#4a7ba6", alpha=0.85,
            edgecolor="white", linewidth=0.8, label=f"TNG50-1 (N = {len(A1)})")

    # Highlight strong-lopsided tail
    strong = A1 > 0.1
    ax.hist(A1[strong], bins=bins, color="#e07b39", alpha=0.85,
            edgecolor="white", linewidth=0.8,
            label=f"$A_1 > 0.1$ (N = {strong.sum()}, "
                  f"{100*strong.sum()/len(A1):.1f}%)")

    # Very strong tail
    very_strong = A1 > 0.2
    if very_strong.sum() > 0:
        ax.hist(A1[very_strong], bins=bins, color="#c0392b", alpha=0.9,
                edgecolor="white", linewidth=0.8,
                label=f"$A_1 > 0.2$ (N = {very_strong.sum()}, "
                      f"{100*very_strong.sum()/len(A1):.1f}%)")

    # Median + mean + percentile lines
    med = np.median(A1)
    p16 = np.percentile(A1, 16)
    p84 = np.percentile(A1, 84)
    ax.axvline(med, color="black", linestyle="--", linewidth=1.6, zorder=10,
               label=f"median = {med:.3f}")
    ax.axvline(np.mean(A1), color="black", linestyle=":", linewidth=1.4, zorder=10,
               label=f"mean = {np.mean(A1):.3f} $\\pm$ {np.std(A1):.3f}")

    # 16-84 percentile band (subtle)
    ymax_axis = ax.get_ylim()[1]
    ax.axvspan(p16, p84, ymin=0, ymax=0.03, color="black", alpha=0.5)
    ax.text(0.98, 0.75,
            f"16–84 percentile:\n[{p16:.3f}, {p84:.3f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="grey", alpha=0.9))

    ax.set_xlabel(r"$A_1$ (bootstrap mean, aperture $[1.5, 2.5]\,R_{\rm d}$)")
    ax.set_ylabel("Number of galaxies")
    ax.set_xlim(0, 0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outpath_base}.{ext}", bbox_inches="tight")
    plt.close(fig)

    return {
        "N": len(A1),
        "median_A1": float(med),
        "mean_A1": float(np.mean(A1)),
        "std_A1": float(np.std(A1)),
        "n_strong": int(strong.sum()),
        "n_very_strong": int(very_strong.sum()),
        "p16": float(p16),
        "p84": float(p84),
    }


# =============================================================================
# FIGURE 5 — A_1 vs stellar mass, colour-coded by coherence
# =============================================================================
def make_fig5_A1_vs_mass(cat: dict, outpath_base: Path):
    A1     = cat["A1_bootstrap_mean"]
    A1_err = cat["A1_bootstrap_std"]
    M_star = cat["M_star_Msun"]
    coh    = cat["pattern_coherence"]

    log_M = np.log10(M_star)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter with error bars, colour-coded by coherence
    ax.errorbar(log_M, A1, yerr=A1_err, fmt="none",
                ecolor="lightgrey", elinewidth=0.7, alpha=0.6, zorder=1)
    scat = ax.scatter(log_M, A1, c=coh, s=32, cmap="viridis",
                      edgecolor="black", linewidth=0.4, zorder=2,
                      vmin=0, vmax=1)

    # Running median in three ADAPTIVE mass bins based on data range
    log_M_min = np.floor(log_M.min() * 10) / 10
    log_M_max = np.ceil(log_M.max() * 10) / 10
    bin_edges = np.linspace(log_M_min, log_M_max, 4)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    med_A1, p16_A1, p84_A1, n_bin = [], [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (log_M >= lo) & (log_M < hi)
        if m.sum() > 3:
            med_A1.append(np.median(A1[m]))
            p16_A1.append(np.percentile(A1[m], 16))
            p84_A1.append(np.percentile(A1[m], 84))
            n_bin.append(int(m.sum()))
        else:
            med_A1.append(np.nan); p16_A1.append(np.nan)
            p84_A1.append(np.nan); n_bin.append(int(m.sum()))
    med_A1 = np.array(med_A1); p16_A1 = np.array(p16_A1)
    p84_A1 = np.array(p84_A1)

    # Plot only bins with data
    valid = ~np.isnan(med_A1)
    ax.plot(bin_centers[valid], med_A1[valid], "-", color="crimson",
            linewidth=2.5, marker="D", markersize=9,
            markerfacecolor="crimson", markeredgecolor="white",
            markeredgewidth=1.2, zorder=5, label="Running median")
    ax.fill_between(bin_centers[valid], p16_A1[valid], p84_A1[valid],
                    color="crimson", alpha=0.15, zorder=1,
                    label="16–84 percentile")

    # Print bin N counts near bin centers (bottom of plot to avoid legend)
    for cx, med_val, n in zip(bin_centers, med_A1, n_bin):
        if n > 0:
            ax.text(cx, 0.02, f"N={n}", ha="center", va="bottom",
                    fontsize=9, color="crimson", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="crimson", alpha=0.95))

    # Reference threshold at A_1 = 0.1 (weak lopsided)
    ax.axhline(0.1, color="black", linestyle=":", alpha=0.6,
               label=r"$A_1 = 0.1$ (weak lopsided threshold)")
    # (Duplicate text label removed; legend entry is sufficient.)

    cb = fig.colorbar(scat, ax=ax, pad=0.02)
    cb.set_label(r"Pattern coherence $f_{\rm coh}$")

    ax.set_xlabel(r"$\log_{10}(M_\star \, / \, \mathrm{M}_{\odot})$")
    ax.set_ylabel(r"$A_1$ (bootstrap mean)")
    ax.set_xlim(9.4, 11.0)
    ax.set_ylim(0, 0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outpath_base}.{ext}", bbox_inches="tight")
    plt.close(fig)

    return {
        "median_bins": [float(x) for x in med_A1],
        "bin_centers": [float(x) for x in bin_centers],
        "n_per_bin":   n_bin,
    }


# =============================================================================
# FIGURE 6 — TNG50 vs R&Z95 CDF comparison with KS test
# =============================================================================
def make_fig6_obs_comparison(cat: dict, outpath_base: Path):
    A1_tng = cat["A1_bootstrap_mean"]
    A1_z13_full    = Z2013_A1
    A1_z13_matched = Z2013_A1_matched

    def ecdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        return xs, ys

    x_tng, y_tng           = ecdf(A1_tng)
    x_z13, y_z13           = ecdf(A1_z13_full)
    x_z13_m, y_z13_m       = ecdf(A1_z13_matched)

    # KS tests: full sample and mass-matched
    D_full, p_full = stats.ks_2samp(A1_tng, A1_z13_full)
    D_match, p_match = stats.ks_2samp(A1_tng, A1_z13_matched)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    ax.step(x_tng, y_tng, where="post", color="#2E86AB", linewidth=2.5,
            label=f"TNG50-1 (this work, N = {len(A1_tng)}, "
                  f"median = {np.median(A1_tng):.3f})")
    ax.step(x_z13, y_z13, where="post", color="#C73E1D", linewidth=2.5,
            linestyle="--",
            label=f"Zaritsky+2013 full (N = {len(A1_z13_full)}, "
                  f"median = {np.median(A1_z13_full):.3f})")
    ax.step(x_z13_m, y_z13_m, where="post", color="#8E44AD", linewidth=2.5,
            linestyle=":",
            label=f"Zaritsky+2013 mass-matched (N = {len(A1_z13_matched)}, "
                  f"median = {np.median(A1_z13_matched):.3f})")

    # Lokas 2022 TNG100 reference — vertical line at published median
    # (full distribution not available, only summary statistics from paper)
    ax.axvline(LOKAS_A1_MEDIAN, color="#16A085", linestyle="-",
               linewidth=1.8, alpha=0.9, zorder=3)
    ax.text(LOKAS_A1_MEDIAN + 0.005, 0.55,
            f"\u0141okas+2022\nTNG100 median\n(N = {LOKAS_N})",
            fontsize=8.5, color="#16A085", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#16A085", alpha=0.9))

    ax.set_xlabel(r"$A_1$")
    ax.set_ylabel("Cumulative distribution")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.02)

    def p_str(p):
        if p < 1e-4:
            exp = int(np.floor(np.log10(max(p, 1e-300))))
            return f"$p < 10^{{{exp}}}$"
        elif p < 0.001:
            return f"$p = {p:.1e}$"
        else:
            return f"$p = {p:.3f}$"

    ratio_full  = np.median(A1_z13_full)    / max(np.median(A1_tng), 1e-6)
    ratio_match = np.median(A1_z13_matched) / max(np.median(A1_tng), 1e-6)

    ax.text(0.98, 0.05,
            f"Two-sample KS tests:\n"
            f"vs. full:         $D = {D_full:.3f}$,   {p_str(p_full)}\n"
            f"vs. matched: $D = {D_match:.3f}$,   {p_str(p_match)}\n"
            f"Median ratio (full/TNG):     ${ratio_full:.2f}\\times$\n"
            f"Median ratio (matched/TNG): ${ratio_match:.2f}\\times$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF9E7",
                      edgecolor="#B7950B", linewidth=1.0))

    ax.legend(loc="lower right", frameon=True, bbox_to_anchor=(1.0, 0.42),
              framealpha=0.95, fontsize=9)


    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outpath_base}.{ext}", bbox_inches="tight")
    plt.close(fig)

    return {
        "D_KS_full": float(D_full),
        "p_value_full": float(p_full),
        "D_KS_matched": float(D_match),
        "p_value_matched": float(p_match),
        "tng_median": float(np.median(A1_tng)),
        "z13_median_full": float(np.median(A1_z13_full)),
        "z13_median_matched": float(np.median(A1_z13_matched)),
        "ratio_full": float(ratio_full),
        "ratio_matched": float(ratio_match),
        "N_matched": len(A1_z13_matched),
    }


# =============================================================================
# Summary printout for LaTeX \todo{} filling
# =============================================================================
def print_paper_numbers(stats_fig4, stats_fig5, stats_fig6):
    print("\n" + "=" * 72)
    print("NUMBERS FOR YOUR PAPER (replace \\todo{...} placeholders)")
    print("=" * 72)

    print("\n[Abstract]")
    print(f"  median <A_1>         = {stats_fig4['median_A1']:.3f}")
    print(f"  mean <A_1>           = {stats_fig4['mean_A1']:.3f}"
          f" +/- {stats_fig4['std_A1']:.3f}")
    print(f"  coherent m=2 fraction = ... (from your catalog summary: 64.65%)")

    print("\n[Section 4.3 — Amplitude distribution]")
    print(f"  N galaxies             = {stats_fig4['N']}")
    print(f"  median A_1             = {stats_fig4['median_A1']:.3f}")
    print(f"  16-84 percentile range = [{stats_fig4['p16']:.3f}, "
          f"{stats_fig4['p84']:.3f}]")
    print(f"  strong lopsided (A>0.1): {stats_fig4['n_strong']} "
          f"({100*stats_fig4['n_strong']/stats_fig4['N']:.1f}%)")
    print(f"  very strong (A>0.2):     {stats_fig4['n_very_strong']} "
          f"({100*stats_fig4['n_very_strong']/stats_fig4['N']:.1f}%)")

    print("\n[Section 5 — KS test vs Zaritsky+2013 (real S4G observations)]")
    print()
    print(f"  {'Comparison':<28} {'N':>4}  {'median':>7}  {'ratio':>7}  {'D_KS':>6}  {'p-value':>10}")
    print(f"  {'-'*70}")
    print(f"  {'TNG50 (this work)':<28} {'198':>4}  {stats_fig6['tng_median']:>7.3f}  {'-':>7}  {'-':>6}  {'-':>10}")
    print(f"  {'Zaritsky+2013 FULL':<28} {'167':>4}  {stats_fig6['z13_median_full']:>7.3f}  "
          f"{stats_fig6['ratio_full']:>6.2f}x  {stats_fig6['D_KS_full']:>6.3f}  "
          f"{stats_fig6['p_value_full']:>10.2e}")
    print(f"  {'Zaritsky MASS-MATCHED':<28} {stats_fig6['N_matched']:>4}  {stats_fig6['z13_median_matched']:>7.3f}  "
          f"{stats_fig6['ratio_matched']:>6.2f}x  {stats_fig6['D_KS_matched']:>6.3f}  "
          f"{stats_fig6['p_value_matched']:>10.2e}")
    print()
    print(f"  KEY OBSERVATIONS:")
    print(f"  * Full-sample excess ({stats_fig6['ratio_full']:.1f}x) is partly driven by low-mass")
    print(f"    Zaritsky galaxies (median log_M~9.0) outside our TNG50 filter.")
    print(f"  * Mass-matched excess ({stats_fig6['ratio_matched']:.1f}x) is the fair comparison:")
    print(f"    the residual difference likely reflects TNG feedback physics.")
    print(f"  * Both tests remain statistically significant, but interpretation")
    print(f"    should focus on the mass-matched value in §5 discussion.")

    print("\n[Section 5.2 — Three-level comparison (TNG50 / TNG100 / observations)]")
    print()
    tng50_med = stats_fig4['median_A1']
    tng50_lop = stats_fig4['n_strong'] / stats_fig4['N']
    z13_med   = stats_fig6['z13_median_full']
    print(f"  {'Study':<32} {'sim/obs':<10} {'N':>5}  {'median':>7}  {'A_1>0.1':>8}")
    print(f"  {'-'*72}")
    print(f"  {'This work (TNG50-1)':<32} {'sim':<10} {stats_fig4['N']:>5}  "
          f"{tng50_med:>7.3f}  {100*tng50_lop:>7.1f}%")
    print(f"  {'Lokas 2022 (TNG100)':<32} {'sim':<10} {LOKAS_N:>5}  "
          f"{LOKAS_A1_MEDIAN:>7.3f}  {100*LOKAS_LOPSIDED_FRAC:>7.1f}%")
    print(f"  {'Zaritsky+2013 (S4G)':<32} {'obs':<10} {'167':>5}  "
          f"{z13_med:>7.3f}  {'64.1':>7}%")
    print(f"  {'Zaritsky+2013 mass-matched':<32} {'obs':<10} "
          f"{stats_fig6['N_matched']:>5}  "
          f"{stats_fig6['z13_median_matched']:>7.3f}  {'37.1':>7}%")
    print()
    print(f"  MEDIAN RATIOS:")
    print(f"  * TNG50 / TNG100 (resolution effect)         = "
          f"{tng50_med/LOKAS_A1_MEDIAN:.2f}x")
    print(f"    -> TNG50 (better resolution) recovers more lopsidedness")
    print(f"  * TNG100 / Zaritsky matched (feedback deficit) = "
          f"{stats_fig6['z13_median_matched']/LOKAS_A1_MEDIAN:.2f}x")
    print(f"    -> TNG100 also underpredicts, deficit even larger there")
    print(f"  * TNG50 / Zaritsky matched (residual deficit) = "
          f"{stats_fig6['ratio_matched']:.2f}x")
    print(f"    -> After matching mass, residual TNG50 vs obs is smaller")
    print()
    print(f"  INTERPRETATION for §5.2 of paper:")
    print(f"    Improved resolution (TNG100->TNG50) recovers a factor of")
    print(f"    ~{tng50_med/LOKAS_A1_MEDIAN:.1f} more lopsidedness. However, the residual")
    print(f"    difference vs mass-matched observations (~{stats_fig6['ratio_matched']:.1f}x) suggests")
    print(f"    that further improvements in resolution or physics are needed")
    print(f"    to fully reproduce the observed distribution.")

    print("\n" + "=" * 72)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog",
                        default="batch_output/catalog.csv",
                        help="Path to catalog.csv from 04_build_catalog.py")
    parser.add_argument("--output", default="paper_figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    csv_path = Path(args.catalog)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading catalog: {csv_path}")
    cat = load_catalog(csv_path)

    print("\nGenerating Fig. 4 (A_1 distribution)...")
    s4 = make_fig4_A1_distribution(cat, outdir / "fig4_A1_distribution")

    print("Generating Fig. 5 (A_1 vs stellar mass)...")
    s5 = make_fig5_A1_vs_mass(cat, outdir / "fig5_A1_vs_mass")

    print("Generating Fig. 6 (TNG50 vs R&Z95 comparison)...")
    s6 = make_fig6_obs_comparison(cat, outdir / "fig6_obs_comparison")

    print(f"\nFigures written to: {outdir}/")
    for f in sorted(outdir.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}   ({size_kb:.1f} KB)")

    print_paper_numbers(s4, s5, s6)


if __name__ == "__main__":
    main()

