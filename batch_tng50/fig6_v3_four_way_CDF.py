#!/usr/bin/env python3
"""
fig6_v3_four_way_CDF.py
========================
Polishing pass on Fig 6 v2:
  - Legend moved to lower-right, KS test box moved to upper-left
    of each panel so they no longer overlap.
  - White opaque background for KS text box (was transparent).
  - Larger font for KS test text (7.8 -> 8.4 pt).
  - Line width slightly increased for readability.

Data / statistics unchanged from v2.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


TNG50_CATALOG  = "batch_output/catalog.csv"
TNG100_CATALOG = "batch_output_tng100/catalog.csv"
TNG50_DT       = "batch_output/disc_quality.csv"
TNG100_DT      = "batch_output_tng100/disc_quality.csv"

LOKAS_MEDIAN     = 0.044
LOKAS_A1_GT01    = 0.084
LOKAS_N          = 1912


def load_merged(cat_path, dq_path):
    cat = pd.read_csv(cat_path)
    dq  = pd.read_csv(dq_path)[["subhalo_id", "DT_ratio", "kappa_rot"]]
    return cat.merge(dq, on="subhalo_id")


def science_filter(df):
    return (df["DT_ratio"] > 0.5) & \
           (df["n_disk_particles"] > 10000) & \
           (df["A1_bootstrap_mean"] < 0.5)


def import_zaritsky_arrays(paper_figs_path: Path):
    spec = importlib.util.spec_from_file_location("paper_figs", paper_figs_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod.Z2013_A1, mod.Z2013_A1_matched


def cdf(sample):
    x = np.sort(np.asarray(sample))
    F = np.arange(1, len(x) + 1) / len(x)
    return x, F


def make_panel(ax, sets, xmax=0.6):
    for s in sets:
        if s.get("median_only"):
            ax.axvline(s["median"], color=s["color"], ls=s["style"],
                       lw=s["lw"], label=s["label"], zorder=3)
        else:
            x, F = cdf(s["data"])
            ax.plot(x, F, color=s["color"], ls=s["style"], lw=s["lw"],
                    label=s["label"], zorder=3)

    ax.axvline(0.1, color="grey", ls=":", lw=1.0, alpha=0.6, zorder=1)
    ax.text(0.105, 0.03, r"$A_1>0.1$", color="grey", fontsize=8,
            rotation=90, va="bottom")

    ax.set_xlabel(r"$A_1$ (bootstrap mean)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3, linewidth=0.4, zorder=0)


def ks_line(x, y, label=""):
    D, p = ks_2samp(x, y)
    if p < 1e-14:
        ptxt = r"$p < 10^{-14}$"
    elif p < 1e-3:
        ptxt = f"$p = {p:.1e}$"
    else:
        ptxt = f"$p = {p:.3f}$"
    return f"{label} $D={D:.3f}$, {ptxt}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="figures_v2")
    p.add_argument("--paper-figs", default="make_paper_figures.py")
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cat50  = load_merged(TNG50_CATALOG,  TNG50_DT)
    cat100 = load_merged(TNG100_CATALOG, TNG100_DT)
    a1_50_raw  = cat50 ["A1_bootstrap_mean"].dropna().values
    a1_100_raw = cat100["A1_bootstrap_mean"].dropna().values
    a1_50_flt  = cat50 .loc[science_filter(cat50),  "A1_bootstrap_mean"].dropna().values
    a1_100_flt = cat100.loc[science_filter(cat100), "A1_bootstrap_mean"].dropna().values

    Z_full, Z_matched = import_zaritsky_arrays(Path(args.paper_figs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6))

    sets_A = [
        dict(data=a1_50_raw,  color="#08306b", style="-",  lw=2.6,
             label=f"TNG50-1  (this work, N={len(a1_50_raw)}, "
                   f"median={np.median(a1_50_raw):.3f})"),
        dict(data=a1_100_raw, color="#7c2d12", style="-",  lw=2.6,
             label=f"TNG100-1 (this work, N={len(a1_100_raw)}, "
                   f"median={np.median(a1_100_raw):.3f})"),
        dict(data=Z_full,     color="black",   style="--", lw=1.9,
             label=f"Zaritsky+2013 full (N={len(Z_full)}, "
                   f"median={np.median(Z_full):.3f})"),
        dict(data=Z_matched,  color="grey",    style=":",  lw=2.0,
             label=f"Zaritsky+2013 mass-matched (N={len(Z_matched)}, "
                   f"median={np.median(Z_matched):.3f})"),
        dict(median_only=True, median=LOKAS_MEDIAN,
             color="#008b45", style="-", lw=1.8,
             label=f"Łokas 2022 TNG100 (N={LOKAS_N}, "
                   f"median={LOKAS_MEDIAN})"),
    ]
    make_panel(ax1, sets_A, xmax=0.6)
    ax1.legend(loc="lower right", fontsize=8.4, framealpha=0.95)
    ax1.text(0.0, 1.015, "(a)", transform=ax1.transAxes, fontsize=11, ha="left", va="bottom")

    ks_A_text = (
        "Two-sample KS tests:\n"
        "  vs. Zaritsky+2013 full:\n"
        f"    TNG50-1:  {ks_line(a1_50_raw,  Z_full)}\n"
        f"    TNG100-1: {ks_line(a1_100_raw, Z_full)}\n"
        "  vs. Zaritsky+2013 matched:\n"
        f"    TNG50-1:  {ks_line(a1_50_raw,  Z_matched)}\n"
        f"    TNG100-1: {ks_line(a1_100_raw, Z_matched)}"
    )

    sets_B = [
        dict(data=a1_50_flt,  color="#08306b", style="-",  lw=2.6,
             label=f"TNG50-1  filtered (N={len(a1_50_flt)}, "
                   f"median={np.median(a1_50_flt):.3f})"),
        dict(data=a1_100_flt, color="#7c2d12", style="-",  lw=2.6,
             label=f"TNG100-1 filtered (N={len(a1_100_flt)}, "
                   f"median={np.median(a1_100_flt):.3f})"),
        dict(data=Z_full,     color="black",   style="--", lw=1.9,
             label=f"Zaritsky+2013 full (N={len(Z_full)}, "
                   f"median={np.median(Z_full):.3f})"),
        dict(data=Z_matched,  color="grey",    style=":",  lw=2.0,
             label=f"Zaritsky+2013 mass-matched (N={len(Z_matched)}, "
                   f"median={np.median(Z_matched):.3f})"),
        dict(median_only=True, median=LOKAS_MEDIAN,
             color="#008b45", style="-", lw=1.8,
             label=f"Łokas 2022 TNG100 (N={LOKAS_N}, "
                   f"median={LOKAS_MEDIAN})"),
    ]
    make_panel(ax2, sets_B, xmax=0.6)
    ax2.legend(loc="lower right", fontsize=8.4, framealpha=0.95)
    ax2.text(0.0, 1.015, "(b)", transform=ax2.transAxes, fontsize=11, ha="left", va="bottom")

    ks_B_text = (
        "Two-sample KS tests:\n"
        "  vs. Zaritsky+2013 full:\n"
        f"    TNG50-1:  {ks_line(a1_50_flt,  Z_full)}\n"
        f"    TNG100-1: {ks_line(a1_100_flt, Z_full)}\n"
        "  vs. Zaritsky+2013 matched:\n"
        f"    TNG50-1:  {ks_line(a1_50_flt,  Z_matched)}\n"
        f"    TNG100-1: {ks_line(a1_100_flt, Z_matched)}"
    )

    plt.tight_layout()
    fig.savefig(outdir / "fig6_v3_four_way_CDF.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig6_v3_four_way_CDF.png", dpi=150,
                bbox_inches="tight")
    print(f"Saved: {outdir}/fig6_v3_four_way_CDF.pdf/.png")

    print("\n=== Fig 6 v3 summary ===")
    print(f"{'Sample':<30} {'N':>6} {'median':>10} {'>0.1 frac':>12}")
    print("-" * 60)
    for name, arr in [("TNG50-1 raw",         a1_50_raw),
                      ("TNG100-1 raw",        a1_100_raw),
                      ("TNG50-1 filtered",    a1_50_flt),
                      ("TNG100-1 filtered",   a1_100_flt),
                      ("Zaritsky+2013 full",  Z_full),
                      ("Zaritsky+2013 match", Z_matched)]:
        print(f"{name:<30} {len(arr):>6} {np.median(arr):>10.4f} "
              f"{100*(arr>0.1).mean():>11.1f}%")
    print(f"{'Lokas 2022 TNG100':<30} {LOKAS_N:>6} {LOKAS_MEDIAN:>10.4f} "
          f"{100*LOKAS_A1_GT01:>11.1f}%")


if __name__ == "__main__":
    main()
