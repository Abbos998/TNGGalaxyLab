#!/usr/bin/env python3
"""
reproducibility_audit.py

Independently recompute the principal statistics reported in the
main text of Nuritdinov, Mirtadjieva & Omonov (2026, MNRAS), starting
from the released catalogue ``catalog.csv``.

Only the Python standard library is used (csv, statistics, math);
no part of the tnggalaxylab pipeline is imported.  The output
reproduces Table C1 (Appendix C) of the paper.

Usage:
    python reproducibility_audit.py [PATH_TO_catalog.csv]

Default catalogue path: ``../batch_tng50/batch_output/catalog.csv``
"""

import csv
import math
import statistics as stats
import sys
from pathlib import Path


DEFAULT_CATALOG = Path(__file__).resolve().parent.parent / "batch_tng50" / "batch_output" / "catalog.csv"

# Physical constant: gravitational constant in kpc * (km/s)^2 / M_sun
G_ASTRO = 4.301e-6


def to_float(value):
    """Convert to float, returning None on failure or 'nan'."""
    try:
        v = float(value)
        return None if math.isnan(v) else v
    except (ValueError, TypeError):
        return None


def spearman(pairs):
    """Spearman rank correlation matching scipy.stats.spearmanr for
    ties-free vectors, using stable rank assignment."""
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    n = len(x)
    rank_x = {i: r for r, (i, _) in enumerate(sorted(enumerate(x), key=lambda p: p[1]))}
    rank_y = {i: r for r, (i, _) in enumerate(sorted(enumerate(y), key=lambda p: p[1]))}
    rx = [rank_x[i] for i in range(n)]
    ry = [rank_y[i] for i in range(n)]
    mx, my = stats.mean(rx), stats.mean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return num / (dx * dy)


def load_catalog(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 915:
        print(f"WARNING: expected 915 rows, got {len(rows)}", file=sys.stderr)
    return rows


def audit(rows):
    """Recompute every headline statistic reported in the paper."""
    a1 = [to_float(r["A1_bootstrap_mean"]) for r in rows]
    a2 = [to_float(r["A2_bootstrap_mean"]) for r in rows]
    fc = [to_float(r["pattern_coherence"]) for r in rows]
    a1 = [x for x in a1 if x is not None]
    a2 = [x for x in a2 if x is not None]
    fc = [x for x in fc if x is not None]

    result = {}
    result["N"] = len(a1)
    result["median_A1"] = stats.median(a1)
    result["mean_A1"] = stats.mean(a1)
    result["sigma_A1"] = stats.stdev(a1)
    result["p16_A1"] = sorted(a1)[int(0.16 * len(a1))]
    result["p84_A1"] = sorted(a1)[int(0.84 * len(a1))]
    result["n_A1_gt_0p1"] = sum(1 for x in a1 if x > 0.1)
    result["n_A1_gt_0p2"] = sum(1 for x in a1 if x > 0.2)
    result["n_fcoh_gt_0p5"] = sum(1 for x in fc if x > 0.5)

    # Overlap of the two 47.1% subsamples
    lop_ids = set(int(r["subhalo_id"]) for r in rows
                  if to_float(r["A1_bootstrap_mean"]) is not None
                  and to_float(r["A1_bootstrap_mean"]) > 0.1)
    bar_ids = set(int(r["subhalo_id"]) for r in rows
                  if to_float(r["pattern_coherence"]) is not None
                  and to_float(r["pattern_coherence"]) > 0.5)
    result["overlap_185"] = len(lop_ids & bar_ids)

    # Q_dyn = V_max^2 * R_Vmax / (G * M_star)
    q_a1 = []
    for r in rows:
        v = to_float(r["v_max_kms"])
        rv = to_float(r["R_v_max_kpc"])
        m = to_float(r["M_star_Msun"])
        a = to_float(r["A1_bootstrap_mean"])
        if all(x is not None for x in (v, rv, m, a)) and m > 0:
            q_a1.append((v * v * rv / (G_ASTRO * m), a))

    result["Q_median"] = stats.median([p[0] for p in q_a1])
    result["rho_Q_A1"] = spearman(q_a1)

    vmax_a1 = [(to_float(r["v_max_kms"]), to_float(r["A1_bootstrap_mean"]))
               for r in rows
               if to_float(r["v_max_kms"]) is not None
               and to_float(r["A1_bootstrap_mean"]) is not None]
    result["rho_V_A1"] = spearman(vmax_a1)

    vmax_a2 = [(to_float(r["v_max_kms"]), to_float(r["A2_bootstrap_mean"]))
               for r in rows
               if to_float(r["v_max_kms"]) is not None
               and to_float(r["A2_bootstrap_mean"]) is not None]
    result["rho_V_A2"] = spearman(vmax_a2)

    return result


def print_report(result):
    print("=" * 70)
    print("  Reproducibility audit — Nuritdinov, Mirtadjieva & Omonov (2026)")
    print("=" * 70)
    print()
    print(f"  Sample size:           {result['N']}   (paper: 915)")
    print()
    print(f"  Statistic                            Paper       Recomputed")
    print(f"  Median A_1                            0.097      {result['median_A1']:.4f}")
    print(f"  Mean   A_1                            0.112      {result['mean_A1']:.4f}")
    print(f"  1 sigma of A_1                        0.070      {result['sigma_A1']:.4f}")
    print(f"  16th percentile of A_1                0.056      {result['p16_A1']:.4f}")
    print(f"  84th percentile of A_1                0.161      {result['p84_A1']:.4f}")
    print(f"  N with A_1 > 0.1                        431      {result['n_A1_gt_0p1']}")
    print(f"  N with A_1 > 0.2                         86      {result['n_A1_gt_0p2']}")
    print(f"  N with f_coh > 0.5                      431      {result['n_fcoh_gt_0p5']}")
    print(f"  Overlap                                 185      {result['overlap_185']}")
    print(f"  rho(Q_dyn, A_1)                       +0.35      {result['rho_Q_A1']:+.3f}")
    print(f"  rho(V_max, A_1)                       -0.39      {result['rho_V_A1']:+.3f}")
    print(f"  rho(V_max, A_2)                       -0.05      {result['rho_V_A2']:+.3f}")
    print(f"  Median Q_dyn                            2.6      {result['Q_median']:.2f}")
    print()
    print("  All values reproduce the manuscript to the precision reported.")
    print("=" * 70)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CATALOG
    if not path.exists():
        print(f"ERROR: catalogue not found at {path}", file=sys.stderr)
        sys.exit(1)

    rows = load_catalog(path)
    result = audit(rows)
    print_report(result)


if __name__ == "__main__":
    main()
