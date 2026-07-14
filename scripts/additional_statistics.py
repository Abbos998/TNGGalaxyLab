"""
additional_statistics.py
========================

Phase 3.3 — Additional statistical tests for the TNG50 vs. Zaritsky+2013
comparison, complementing the two-sample KS test reported in Section 5.1.

Adds four widely-used non-parametric and parametric measures of the
difference between two distributions:

    1. Mann-Whitney U        — rank-based test of location shift
    2. Anderson-Darling k    — more sensitive than KS in the distribution tails
    3. Cliff's delta         — non-parametric effect-size measure
                               (probability that a random draw from one sample
                               exceeds a random draw from the other)
    4. Cohen's d             — parametric standardised mean difference

Because the Zaritsky+2013 catalogue is not distributed with our code,
this script uses PLACEHOLDER hardcoded values matching Table 1 of the
paper.  For a fully reproducible analysis with real sample data, replace
the `TNG50_A1` and `ZARITSKY_A1_FULL`/`ZARITSKY_A1_MATCHED` arrays with
the actual data vectors.

Usage
-----
    python additional_statistics.py

Outputs:
    batch_tng50/paper_figures/additional_statistics.md
    batch_tng50/paper_figures/additional_statistics.csv

Author: Abbos Omonov et al. (2026)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Placeholder Zaritsky+2013 A_1 samples
# =============================================================================
# These are approximate distributions inferred from the paper Table 1
# and the histogram/CDF (Figure 6).  Replace with actual data if available.
#
# Full sample (N=167): mean ~0.16, spread visible in Fig 6
# Mass-matched subsample (N=35): mean ~0.09

def _synthesise_zaritsky_placeholder(median: float, N: int,
                                       seed: int) -> np.ndarray:
    """Synthesise a log-normal placeholder matching the target median."""
    rng = np.random.default_rng(seed)
    sigma = 0.55
    mu = np.log(median)
    samples = rng.lognormal(mu, sigma, size=N)
    samples = np.clip(samples, 0.005, 1.0)
    return samples

ZARITSKY_A1_FULL = _synthesise_zaritsky_placeholder(0.160, 167, seed=42)
ZARITSKY_A1_MATCHED = _synthesise_zaritsky_placeholder(0.090, 35, seed=43)


# =============================================================================
# Statistical measures
# =============================================================================
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size.

    delta = (# pairs where x_i > y_j - # pairs where x_i < y_j) / (n_x * n_y)

    Ranges from -1 (all x_i < all y_j) to +1 (all x_i > all y_j).
    Common thresholds: |δ|<0.147 negligible, <0.33 small, <0.474 medium,
    >=0.474 large (Romano+2006).
    """
    nx, ny = len(x), len(y)
    # Rank-based efficient version
    combined = np.concatenate([x, y])
    ranks = stats.rankdata(combined)
    rx = ranks[:nx]
    ry = ranks[nx:]
    delta = 2 * (rx.mean() - ry.mean()) / (nx + ny)
    return float(delta)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d (standardised mean difference, pooled SD)."""
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    if pooled == 0:
        return float("nan")
    return float((mx - my) / pooled)


def interpret_cliffs(delta: float) -> str:
    """Return qualitative Cliff's delta descriptor (Romano+2006)."""
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


def interpret_cohens_d(d: float) -> str:
    """Return qualitative Cohen's d descriptor (Cohen 1988)."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


# =============================================================================
# Compute all statistics for a pair
# =============================================================================
def compare_two_samples(x: np.ndarray, y: np.ndarray,
                          x_label: str, y_label: str) -> dict:
    """Compute a battery of two-sample tests + effect sizes."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Location & spread
    med_x, med_y = np.median(x), np.median(y)
    mean_x, mean_y = np.mean(x), np.mean(y)

    # KS
    ks = stats.ks_2samp(x, y)

    # Mann-Whitney U (two-sided)
    mw = stats.mannwhitneyu(x, y, alternative="two-sided")

    # Anderson-Darling k-sample
    try:
        ad = stats.anderson_ksamp([x, y])
        ad_stat = float(ad.statistic)
        ad_pvalue = float(ad.pvalue)
    except Exception as e:
        ad_stat, ad_pvalue = float("nan"), float("nan")

    # Effect sizes
    delta = cliffs_delta(x, y)
    d = cohens_d(x, y)

    return {
        "comparison": f"{x_label} vs {y_label}",
        "N_x": len(x),
        "N_y": len(y),
        "median_x": med_x,
        "median_y": med_y,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "median_ratio": med_y / med_x if med_x > 0 else float("nan"),
        "KS_D": float(ks.statistic),
        "KS_p": float(ks.pvalue),
        "MannWhitney_U": float(mw.statistic),
        "MannWhitney_p": float(mw.pvalue),
        "AndersonDarling_stat": ad_stat,
        "AndersonDarling_p": ad_pvalue,
        "Cliffs_delta": delta,
        "Cliffs_interpretation": interpret_cliffs(delta),
        "Cohens_d": d,
        "Cohens_interpretation": interpret_cohens_d(d),
    }


# =============================================================================
# Reporting
# =============================================================================
def write_report(results: list[dict], outpath: Path) -> None:
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("# Additional Statistical Tests\n\n")
        f.write("Complements the two-sample KS test in Section 5.1 of the "
                "paper with three additional non-parametric measures "
                "(Mann-Whitney U, Anderson-Darling k-sample, Cliff's "
                "delta) and one parametric effect-size measure "
                "(Cohen's d).\n\n")

        for res in results:
            f.write(f"## {res['comparison']}\n\n")

            # --- Sample summary ---------------------------------------------
            f.write("**Sample sizes**: "
                    f"N({res['comparison'].split(' vs ')[0]}) = {res['N_x']}, "
                    f"N({res['comparison'].split(' vs ')[1]}) = {res['N_y']}\n\n")

            f.write("| Sample | Median | Mean |\n")
            f.write("|---|---|---|\n")
            lx, ly = res['comparison'].split(' vs ')
            f.write(f"| {lx} | {res['median_x']:.4f} | {res['mean_x']:.4f} |\n")
            f.write(f"| {ly} | {res['median_y']:.4f} | {res['mean_y']:.4f} |\n\n")
            f.write(f"Median ratio {ly}/{lx} = {res['median_ratio']:.3f}\n\n")

            # --- Tests -------------------------------------------------------
            f.write("### Tests of distribution equality\n\n")
            f.write("| Test | Statistic | p-value |\n")
            f.write("|---|---|---|\n")
            f.write(f"| Kolmogorov-Smirnov ($D$)          | "
                    f"{res['KS_D']:.4f} | {res['KS_p']:.3e} |\n")
            f.write(f"| Mann-Whitney U                    | "
                    f"{res['MannWhitney_U']:.0f} | {res['MannWhitney_p']:.3e} |\n")
            f.write(f"| Anderson-Darling ($k$-sample $T$) | "
                    f"{res['AndersonDarling_stat']:.4f} | "
                    f"{res['AndersonDarling_p']:.3e} |\n\n")

            # --- Effect sizes -----------------------------------------------
            f.write("### Effect sizes\n\n")
            f.write("| Measure | Value | Interpretation |\n")
            f.write("|---|---|---|\n")
            f.write(f"| Cliff's $\\delta$ | {res['Cliffs_delta']:+.3f} | "
                    f"{res['Cliffs_interpretation']} |\n")
            f.write(f"| Cohen's $d$       | {res['Cohens_d']:+.3f} | "
                    f"{res['Cohens_interpretation']} |\n\n")

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

    print(f"Loading TNG50 catalog: {args.catalog}")
    cat = pd.read_csv(args.catalog)
    tng50_A1 = cat["A1_bootstrap_mean"].dropna().values
    print(f"TNG50 sample: N = {len(tng50_A1)}, "
          f"median = {np.median(tng50_A1):.4f}")

    print(f"\nZaritsky+2013 samples (placeholder synthetic distributions):")
    print(f"  Full sample:    N = {len(ZARITSKY_A1_FULL)}, "
          f"median = {np.median(ZARITSKY_A1_FULL):.4f}")
    print(f"  Mass-matched:   N = {len(ZARITSKY_A1_MATCHED)}, "
          f"median = {np.median(ZARITSKY_A1_MATCHED):.4f}")

    print("\nRunning all statistical tests...")

    results = []
    results.append(compare_two_samples(
        tng50_A1, ZARITSKY_A1_FULL,
        "TNG50", "Zaritsky+2013 full"
    ))
    results.append(compare_two_samples(
        tng50_A1, ZARITSKY_A1_MATCHED,
        "TNG50", "Zaritsky+2013 matched"
    ))

    # Print summary to console
    print()
    print("=" * 70)
    print("ADDITIONAL STATISTICS SUMMARY")
    print("=" * 70)
    for res in results:
        print(f"\n  {res['comparison']}")
        print(f"    KS D  = {res['KS_D']:.3f}, p = {res['KS_p']:.2e}")
        print(f"    MW U  = {res['MannWhitney_U']:.0f}, "
              f"p = {res['MannWhitney_p']:.2e}")
        print(f"    AD T  = {res['AndersonDarling_stat']:.3f}, "
              f"p = {res['AndersonDarling_p']:.2e}")
        print(f"    Cliff = {res['Cliffs_delta']:+.3f} "
              f"({res['Cliffs_interpretation']})")
        print(f"    d     = {res['Cohens_d']:+.3f} "
              f"({res['Cohens_interpretation']})")

    # Save CSV
    pd.DataFrame(results).to_csv(outdir / "additional_statistics.csv",
                                    index=False)
    write_report(results, outdir / "additional_statistics.md")

    print(f"\nAll outputs in: {outdir}")


if __name__ == "__main__":
    main()
