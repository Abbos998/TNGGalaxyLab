"""
validate_synthetic_extended.py
==============================

Extended systematic-effect validation of the TNGGalaxyLab Fourier pipeline.

This script complements the basic amplitude-recovery validation in
`validate_synthetic.py` by testing three additional systematic effects
that are relevant to real cosmological analyses:

    1. INCLINATION       —  disc tilted by 0°, 15°, 30°, 45°, 60°
    2. CENTERING OFFSET  —  centre shifted by 0.1, 0.5, 1.0, 2.0 kpc
    3. PARTICLE LOSS     —  10%, 30%, 50% of particles removed at random

Each test uses a synthetic *lopsided* disc with ε₁ = 0.30 as the reference,
so the analytic answer is always A₁ = 0.30.  We measure the recovered A₁
after applying the perturbation, and report the fractional bias
    ΔA₁ / A₁ = (A₁_recovered - 0.30) / 0.30

For each systematic level we run N_REPLICAS realisations to obtain
uncertainties.  Results are written to CSV and a 3-panel publication
figure (`fig7_extended_validation.pdf/.png`) suitable for inclusion in
Section 3.5 of the paper.

Usage
-----
    python validate_synthetic_extended.py \
        --output validation/extended \
        --n-particles 100000 \
        --replicas 5

Author: Abbos Omonov et al. (2026)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tnggalaxylab.fourier import (
    make_lopsided_disk,
    compute_fourier,
    global_lopsidedness,
)

# =============================================================================
# Configuration
# =============================================================================
REFERENCE_EPSILON_1 = 0.30   # injected lopsidedness for reference galaxy
REFERENCE_R_D = 3.0          # scale length [kpc]

INCLINATION_ANGLES_DEG = (0.0, 15.0, 30.0, 45.0, 60.0)
CENTER_OFFSETS_KPC = (0.0, 0.1, 0.5, 1.0, 2.0)
PARTICLE_LOSS_FRAC = (0.0, 0.10, 0.30, 0.50)

DEFAULT_N_PARTICLES = 100_000
DEFAULT_REPLICAS = 5


# =============================================================================
# Utility: run the pipeline on (x, y, mass) → recovered A_1
# =============================================================================
def recover_A1(x: np.ndarray, y: np.ndarray, mass: np.ndarray,
               R_d: float = REFERENCE_R_D) -> float:
    """Run the TNGGalaxyLab pipeline on positions, return literature-aperture A_1."""
    profile = compute_fourier(
        x, y, mass,
        method="particles",
        r_in=0.0,
        r_out=15.0,
        n_bins=40,
        m_max=2,
    )
    gm = global_lopsidedness(profile, scale_length=R_d)
    return float(gm.A1_literature)


# =============================================================================
# TEST 1 — INCLINATION
# =============================================================================
def apply_inclination(x: np.ndarray, y: np.ndarray,
                      inclination_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Tilt a face-on disc by a given inclination angle.

    The disc is originally in the x-y plane.  Inclination is applied by
    rotating about the x-axis by angle i:
        y'  = y cos(i)         (z is uninteresting — we project back to 2D)
        x'  = x                (unchanged along the tilt axis)

    This mimics the projection of an inclined disc as it would appear
    to a distant observer looking down the z-axis.  The face-on
    reference plane is recovered when i = 0°.
    """
    i_rad = np.deg2rad(inclination_deg)
    return x, y * np.cos(i_rad)


def run_inclination_test(n_particles: int, replicas: int) -> list[dict]:
    """Test how recovered A_1 depends on disc inclination."""
    print("\n" + "=" * 70)
    print("TEST 1: INCLINATION")
    print("=" * 70)

    rows = []
    for inc_deg in INCLINATION_ANGLES_DEG:
        recovered = []
        for r in range(replicas):
            gal = make_lopsided_disk(
                n_particles=n_particles,
                R_d=REFERENCE_R_D,
                epsilon_1=REFERENCE_EPSILON_1,
                phi_1=0.0,
                seed=1000 + int(inc_deg) * 100 + r,
            )
            x, y = apply_inclination(gal.x, gal.y, inc_deg)
            A1 = recover_A1(x, y, gal.mass, R_d=REFERENCE_R_D)
            recovered.append(A1)

            print(f"  inc = {inc_deg:5.1f}°  rep {r+1}/{replicas}  A_1 = {A1:.4f}")

        arr = np.array(recovered)
        rows.append({
            "test": "inclination",
            "level": inc_deg,
            "level_units": "degrees",
            "A1_mean": float(arr.mean()),
            "A1_std": float(arr.std(ddof=1)),
            "A1_bias": float(arr.mean() - REFERENCE_EPSILON_1),
            "A1_frac_bias": float((arr.mean() - REFERENCE_EPSILON_1) / REFERENCE_EPSILON_1),
            "n_replicas": replicas,
        })
    return rows


# =============================================================================
# TEST 2 — CENTERING OFFSET
# =============================================================================
def run_centering_test(n_particles: int, replicas: int) -> list[dict]:
    """Test how recovered A_1 depends on centring error.

    We shift the whole particle distribution by a fixed offset
    (dx, dy) = (offset, 0), simulating a mis-centred galaxy where
    the pipeline's assumed origin does not coincide with the true
    disc centre.  This directly probes the S1 (centring) systematic.
    """
    print("\n" + "=" * 70)
    print("TEST 2: CENTERING OFFSET")
    print("=" * 70)

    rows = []
    for offset in CENTER_OFFSETS_KPC:
        recovered = []
        for r in range(replicas):
            gal = make_lopsided_disk(
                n_particles=n_particles,
                R_d=REFERENCE_R_D,
                epsilon_1=REFERENCE_EPSILON_1,
                phi_1=0.0,
                seed=2000 + int(offset * 10) * 100 + r,
            )
            # Apply centring offset along x
            x_shifted = gal.x + offset
            y_shifted = gal.y

            A1 = recover_A1(x_shifted, y_shifted, gal.mass, R_d=REFERENCE_R_D)
            recovered.append(A1)

            print(f"  offset = {offset:5.2f} kpc  rep {r+1}/{replicas}  A_1 = {A1:.4f}")

        arr = np.array(recovered)
        rows.append({
            "test": "centering",
            "level": offset,
            "level_units": "kpc",
            "A1_mean": float(arr.mean()),
            "A1_std": float(arr.std(ddof=1)),
            "A1_bias": float(arr.mean() - REFERENCE_EPSILON_1),
            "A1_frac_bias": float((arr.mean() - REFERENCE_EPSILON_1) / REFERENCE_EPSILON_1),
            "n_replicas": replicas,
        })
    return rows


# =============================================================================
# TEST 3 — PARTICLE LOSS
# =============================================================================
def run_particle_loss_test(n_particles: int, replicas: int) -> list[dict]:
    """Test how recovered A_1 depends on random particle loss.

    A random subset of particles is deleted, simulating a cosmological
    scenario where some tracers are lost (e.g. mass cut, star-formation
    history selection, or numerical artefact).  The lopsidedness signal
    should be preserved on average, but shot-noise scatter increases.
    """
    print("\n" + "=" * 70)
    print("TEST 3: PARTICLE LOSS")
    print("=" * 70)

    rows = []
    for frac_lost in PARTICLE_LOSS_FRAC:
        recovered = []
        for r in range(replicas):
            gal = make_lopsided_disk(
                n_particles=n_particles,
                R_d=REFERENCE_R_D,
                epsilon_1=REFERENCE_EPSILON_1,
                phi_1=0.0,
                seed=3000 + int(frac_lost * 100) * 100 + r,
            )
            n_keep = int(n_particles * (1 - frac_lost))
            rng = np.random.default_rng(3000 + int(frac_lost * 100) * 100 + r + 7)
            idx = rng.choice(n_particles, size=n_keep, replace=False)
            x_keep = gal.x[idx]
            y_keep = gal.y[idx]
            mass_keep = gal.mass[idx]

            A1 = recover_A1(x_keep, y_keep, mass_keep, R_d=REFERENCE_R_D)
            recovered.append(A1)

            print(f"  loss = {frac_lost:4.0%}  N_kept = {n_keep:6d}  "
                  f"rep {r+1}/{replicas}  A_1 = {A1:.4f}")

        arr = np.array(recovered)
        rows.append({
            "test": "particle_loss",
            "level": frac_lost,
            "level_units": "fraction",
            "A1_mean": float(arr.mean()),
            "A1_std": float(arr.std(ddof=1)),
            "A1_bias": float(arr.mean() - REFERENCE_EPSILON_1),
            "A1_frac_bias": float((arr.mean() - REFERENCE_EPSILON_1) / REFERENCE_EPSILON_1),
            "n_replicas": replicas,
        })
    return rows


# =============================================================================
# Plotting — 3-panel Fig 7
# =============================================================================
def plot_extended_validation(all_rows: list[dict], outpath_pdf: Path,
                              outpath_png: Path) -> None:
    """Produce the 3-panel Figure 7 for the paper."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    panels = [
        ("inclination", "Inclination angle [degrees]",
         INCLINATION_ANGLES_DEG, "Panel A: Inclination", "tab:blue"),
        ("centering", r"Centring offset [kpc]",
         CENTER_OFFSETS_KPC, "Panel B: Centring offset", "tab:orange"),
        ("particle_loss", "Fraction of particles removed",
         PARTICLE_LOSS_FRAC, "Panel C: Particle loss", "tab:green"),
    ]

    for ax, (test_name, xlabel, xvalues, title, color) in zip(axes, panels):
        rows = [r for r in all_rows if r["test"] == test_name]
        levels = np.array([r["level"] for r in rows])
        means = np.array([r["A1_mean"] for r in rows])
        stds = np.array([r["A1_std"] for r in rows])

        ax.errorbar(levels, means, yerr=stds, marker="o", markersize=7,
                    linewidth=1.5, capsize=4, color=color, ecolor="0.4",
                    zorder=3, label=r"Recovered $\langle A_1 \rangle$")

        # Reference line: injected value
        ax.axhline(REFERENCE_EPSILON_1, ls="--", color="0.4", lw=1,
                   zorder=1, label=r"Analytic $\varepsilon_1 = 0.30$")

        # ±10% shaded region for reference
        ax.axhspan(REFERENCE_EPSILON_1 * 0.9,
                    REFERENCE_EPSILON_1 * 1.1,
                    color="0.85", alpha=0.5, zorder=0,
                    label=r"$\pm 10\%$ band")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Recovered $A_1$")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

        # Auto Y-limits per panel: data range + 15% padding
        y_min = max(0.0, min(0.20, (means - stds).min() - 0.02))
        y_max = (means + stds).max() * 1.10
        ax.set_ylim(y_min, y_max)

        # Panel B: use log scale if effect is very large (>2× injected)
        if test_name == "centering" and means.max() > 2 * REFERENCE_EPSILON_1:
            ax.set_yscale("log")
            ax.set_ylim(0.25, means.max() * 1.20)
            ax.set_ylabel(r"Recovered $A_1$ (log scale)")

        if test_name == "particle_loss":
            ax.set_xlim(-0.03, 0.55)

        ax.legend(loc="upper left" if test_name == "centering"
                  else "upper right" if test_name == "inclination"
                  else "lower left", fontsize=8)

    fig.savefig(outpath_pdf, bbox_inches="tight")
    fig.savefig(outpath_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure written to:\n  {outpath_pdf}\n  {outpath_png}")


# =============================================================================
# CSV writer
# =============================================================================
def write_csv(rows: list[dict], outpath: Path) -> None:
    fieldnames = ["test", "level", "level_units",
                   "A1_mean", "A1_std", "A1_bias", "A1_frac_bias",
                   "n_replicas"]
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"CSV written to: {outpath}")


# =============================================================================
# Text report — human-readable summary
# =============================================================================
def write_report(rows: list[dict], outpath: Path,
                  n_particles: int, replicas: int,
                  runtime_s: float) -> None:
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("# Extended Validation Report — TNGGalaxyLab\n\n")
        f.write(f"**Reference disc**: lopsided, ε₁ = {REFERENCE_EPSILON_1}, "
                f"R_d = {REFERENCE_R_D} kpc\n")
        f.write(f"**N particles**: {n_particles:,}\n")
        f.write(f"**Replicas per level**: {replicas}\n")
        f.write(f"**Total wall time**: {runtime_s:.1f} s "
                f"({runtime_s/60:.1f} min)\n\n")

        for test_name, description in [
            ("inclination", "Test 1 — Inclination"),
            ("centering",   "Test 2 — Centring offset"),
            ("particle_loss", "Test 3 — Particle loss"),
        ]:
            f.write(f"## {description}\n\n")
            f.write("| Level | A_1 mean | A_1 std | |ΔA_1/A_1| |\n")
            f.write("|---|---|---|---|\n")
            for r in rows:
                if r["test"] != test_name:
                    continue
                u = r["level_units"]
                lvl = f"{r['level']}" if u != "fraction" else f"{r['level']*100:.0f}%"
                f.write(f"| {lvl} {u if u != 'fraction' else ''} | "
                        f"{r['A1_mean']:.4f} | {r['A1_std']:.4f} | "
                        f"{100 * abs(r['A1_frac_bias']):.2f}% |\n")
            f.write("\n")

        # Key summary bullets
        f.write("## Key findings\n\n")
        inc_60 = next((r for r in rows if r["test"] == "inclination"
                        and r["level"] == 60.0), None)
        offset_10 = next((r for r in rows if r["test"] == "centering"
                          and r["level"] == 1.0), None)
        loss_30 = next((r for r in rows if r["test"] == "particle_loss"
                        and r["level"] == 0.30), None)

        if inc_60:
            f.write(f"* **Inclination**: at 60° tilt, "
                    f"|ΔA₁/A₁| = {100 * abs(inc_60['A1_frac_bias']):.1f}%\n")
        if offset_10:
            f.write(f"* **Centring**: 1 kpc off-centre "
                    f"→ |ΔA₁/A₁| = {100 * abs(offset_10['A1_frac_bias']):.1f}%\n")
        if loss_30:
            f.write(f"* **Particle loss**: 30% loss → "
                    f"|ΔA₁/A₁| = {100 * abs(loss_30['A1_frac_bias']):.1f}%, "
                    f"σ = {loss_30['A1_std']:.4f}\n")

    print(f"Report written to: {outpath}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extended systematic-effect validation for TNGGalaxyLab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", default="validation/extended",
                        help="Output directory")
    parser.add_argument("--n-particles", type=int, default=DEFAULT_N_PARTICLES,
                        help="Particle count per synthetic galaxy")
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS,
                        help="Random realisations per systematic level")
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Extended validation")
    print(f"  Reference: lopsided disc, ε₁ = {REFERENCE_EPSILON_1}, "
          f"R_d = {REFERENCE_R_D} kpc")
    print(f"  N particles: {args.n_particles:,}")
    print(f"  Replicas per level: {args.replicas}")
    print(f"  Total runs: "
          f"{args.replicas * (len(INCLINATION_ANGLES_DEG) + len(CENTER_OFFSETS_KPC) + len(PARTICLE_LOSS_FRAC))}")

    t0 = time.time()
    all_rows = []
    all_rows.extend(run_inclination_test(args.n_particles, args.replicas))
    all_rows.extend(run_centering_test(args.n_particles, args.replicas))
    all_rows.extend(run_particle_loss_test(args.n_particles, args.replicas))
    runtime = time.time() - t0

    print(f"\nAll tests complete in {runtime:.1f} s ({runtime/60:.1f} min).")

    write_csv(all_rows, outdir / "extended_validation_results.csv")
    write_report(all_rows, outdir / "EXTENDED_VALIDATION_REPORT.md",
                  args.n_particles, args.replicas, runtime)
    plot_extended_validation(all_rows,
                              outdir / "fig7_extended_validation.pdf",
                              outdir / "fig7_extended_validation.png")

    print(f"\n{'=' * 70}")
    print("EXTENDED VALIDATION SUMMARY")
    print("=" * 70)
    for test_name in ("inclination", "centering", "particle_loss"):
        print(f"\n  {test_name.upper()}")
        for r in all_rows:
            if r["test"] != test_name:
                continue
            u = r["level_units"]
            lvl = f"{r['level']}" if u != "fraction" else f"{r['level']*100:.0f}%"
            print(f"    level={lvl:>6s} {u if u != 'fraction' else '':10s}  "
                  f"A_1 = {r['A1_mean']:.4f} ± {r['A1_std']:.4f}  "
                  f"({100 * r['A1_frac_bias']:+.1f}%)")


if __name__ == "__main__":
    main()
