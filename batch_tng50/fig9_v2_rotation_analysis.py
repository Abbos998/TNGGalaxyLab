#!/usr/bin/env python3
"""
fig9_v2_rotation_analysis.py
=============================
Three-panel rotation-curve analysis extended to include TNG100-1
alongside TNG50-1. Reveals whether the resolution difference in
lopsidedness is tracked by dynamical properties.

Panel A: A_1 vs V_max               (lopsidedness vs rotation)
Panel B: A_2 vs V_max               (bar strength vs rotation)
Panel C: Q_dyn vs stellar mass      (dynamical mass ratio proxy)

Q_dyn is defined as V_max^2 * R_Vmax / (G * M_star), which measures
the dynamical-to-stellar mass ratio within R_Vmax. It is NOT the
Toomre Q or the bar-stability parameter. This convention matches
the original paper's definition.

TNG50 uses filled markers, TNG100 uses open markers, both with
running medians.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TNG50_CATALOG  = "batch_output/catalog.csv"
TNG100_CATALOG = "batch_output_tng100/catalog.csv"
TNG50_DT       = "batch_output/disc_quality.csv"
TNG100_DT      = "batch_output_tng100/disc_quality.csv"

G_CGS  = 6.6743e-8            # cm^3 g^-1 s^-2
MSUN_G = 1.989e33             # g / M_sun
KPC_CM = 3.0857e21            # cm / kpc
KMS_CMS = 1e5                 # (cm/s) / (km/s)


def load_merged(cat_path, dq_path):
    cat = pd.read_csv(cat_path)
    dq  = pd.read_csv(dq_path)[["subhalo_id", "DT_ratio", "kappa_rot"]]
    return cat.merge(dq, on="subhalo_id")


def science_filter(df):
    return (df["DT_ratio"] > 0.5) & \
           (df["n_disk_particles"] > 10000) & \
           (df["A1_bootstrap_mean"] < 0.5)


def Q_dyn(vmax_kms, r_vmax_kpc, mstar_msun):
    """V_max^2 * R_Vmax / (G * M_star), dimensionless."""
    v_cgs = vmax_kms * KMS_CMS
    r_cgs = r_vmax_kpc * KPC_CM
    m_cgs = mstar_msun * MSUN_G
    return (v_cgs ** 2 * r_cgs) / (G_CGS * m_cgs)


def running_median(x, y, edges):
    xc, med, p16, p84 = [], [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (x >= lo) & (x < hi) & np.isfinite(y)
        if m.sum() < 5:
            continue
        xc.append(0.5 * (lo + hi))
        med.append(np.median(y[m]))
        p16.append(np.percentile(y[m], 16))
        p84.append(np.percentile(y[m], 84))
    return np.array(xc), np.array(med), np.array(p16), np.array(p84)


def plot_panel(ax, x50, y50, x100, y100, edges,
               xlabel, ylabel, title,
               yline=None, yline_label=None,
               ylog=False):
    # Scatter (background)
    ax.scatter(x100, y100, s=4, c="#f4a261", alpha=0.20,
               edgecolors="none", zorder=1, rasterized=True)
    ax.scatter(x50, y50, s=6, c="#2166ac", alpha=0.35,
               edgecolors="none", zorder=2, rasterized=True)

    # Running medians
    c100, m100, p16_100, p84_100 = running_median(x100, y100, edges)
    c50,  m50,  p16_50,  p84_50  = running_median(x50,  y50,  edges)

    ax.fill_between(c100, p16_100, p84_100, color="#d97706",
                    alpha=0.18, zorder=3)
    ax.plot(c100, m100, "-", color="#7c2d12", lw=2.4, marker="s",
            ms=6, mfc="none", mew=1.6, zorder=5,
            label=f"TNG100-1 (N={len(x100)})")

    ax.fill_between(c50, p16_50, p84_50, color="#2166ac",
                    alpha=0.18, zorder=4)
    ax.plot(c50, m50, "-", color="#08306b", lw=2.4, marker="o",
            ms=6, zorder=6,
            label=f"TNG50-1 (N={len(x50)})")

    if yline is not None:
        ax.axhline(yline, color="grey", ls=":", lw=1.0, alpha=0.7, zorder=1)
        if yline_label:
            ax.text(0.02, yline * 1.05, yline_label, color="grey",
                    fontsize=8.5, transform=ax.get_yaxis_transform())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.03, 0.97, "(" + title.split(":")[0].split()[-1].lower() + ")", transform=ax.transAxes, fontsize=11, ha="left", va="top", bbox=dict(fc="white", ec="none", pad=1.0), zorder=10)
    ax.legend(loc="upper right", fontsize=8.6, framealpha=0.92)
    ax.grid(True, alpha=0.3, linewidth=0.4, zorder=0)
    if ylog:
        ax.set_yscale("log")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="figures_v2")
    p.add_argument("--filtered", action="store_true",
                   help="Apply science filter to both simulations")
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cat50  = load_merged(TNG50_CATALOG,  TNG50_DT)
    cat100 = load_merged(TNG100_CATALOG, TNG100_DT)

    if args.filtered:
        cat50  = cat50 [science_filter(cat50)].copy()
        cat100 = cat100[science_filter(cat100)].copy()
        suffix = "_filtered"
        title_suffix = " (science-filtered)"
    else:
        suffix = ""
        title_suffix = ""

    # Drop rows with missing V_max / R_Vmax
    for df in (cat50, cat100):
        df.dropna(subset=["v_max_kms", "R_v_max_kpc",
                          "A1_bootstrap_mean", "A2_bootstrap_mean",
                          "M_star_Msun"], inplace=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # Panel A: A_1 vs V_max
    plot_panel(axes[0],
               cat50 ["v_max_kms"].values, cat50 ["A1_bootstrap_mean"].values,
               cat100["v_max_kms"].values, cat100["A1_bootstrap_mean"].values,
               edges=np.arange(40, 260, 20),
               xlabel=r"$V_{\rm max}$ [km s$^{-1}$]",
               ylabel=r"$\langle A_1 \rangle$",
               title=f"Panel A: lopsidedness vs. rotation{title_suffix}",
               yline=0.1, yline_label=r"$A_1=0.1$")
    axes[0].set_ylim(0, None)

    # Panel B: A_2 vs V_max
    plot_panel(axes[1],
               cat50 ["v_max_kms"].values, cat50 ["A2_bootstrap_mean"].values,
               cat100["v_max_kms"].values, cat100["A2_bootstrap_mean"].values,
               edges=np.arange(40, 260, 20),
               xlabel=r"$V_{\rm max}$ [km s$^{-1}$]",
               ylabel=r"$\langle A_2 \rangle$",
               title=f"Panel B: bar strength vs. rotation{title_suffix}",
               yline=0.2, yline_label=r"$A_2=0.2$ (bar)")
    axes[1].set_ylim(0, None)

    # Panel C: Q_dyn vs stellar mass
    logM50  = np.log10(cat50 ["M_star_Msun"].values)
    logM100 = np.log10(cat100["M_star_Msun"].values)
    Qd50  = Q_dyn(cat50 ["v_max_kms"].values, cat50 ["R_v_max_kpc"].values,
                  cat50 ["M_star_Msun"].values)
    Qd100 = Q_dyn(cat100["v_max_kms"].values, cat100["R_v_max_kpc"].values,
                  cat100["M_star_Msun"].values)
    plot_panel(axes[2],
               logM50, Qd50, logM100, Qd100,
               edges=np.linspace(9.5, 11.0, 8),
               xlabel=r"$\log_{10}(M_\star / M_\odot)$",
               ylabel=r"$Q_{\rm dyn} \equiv V_{\rm max}^2 R_{V_{\rm max}} / (G M_\star)$",
               title=f"Panel C: dynamical mass proxy{title_suffix}",
               yline=1.0, yline_label=r"$Q_{\rm dyn}=1$",
               ylog=True)
    axes[2].set_ylim(0.1, 50)

    plt.tight_layout()
    fig.savefig(outdir / f"fig9_v2_rotation_analysis{suffix}.pdf",
                bbox_inches="tight")
    fig.savefig(outdir / f"fig9_v2_rotation_analysis{suffix}.png",
                dpi=150, bbox_inches="tight")
    print(f"Saved: {outdir}/fig9_v2_rotation_analysis{suffix}.pdf/.png")

    print("\n=== Fig 9 v2 summary ===")
    print(f"TNG50  N={len(cat50)}, V_max median={cat50['v_max_kms'].median():.1f} km/s")
    print(f"TNG100 N={len(cat100)}, V_max median={cat100['v_max_kms'].median():.1f} km/s")
    print(f"TNG50  Q_dyn median = {np.median(Qd50):.2f}")
    print(f"TNG100 Q_dyn median = {np.median(Qd100):.2f}")


if __name__ == "__main__":
    main()
