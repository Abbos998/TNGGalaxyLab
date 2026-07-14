"""
03_process_batch.py
====================
Run the Stage 4 Fourier pipeline on every downloaded cutout.
Writes per-subhalo JSON results to <processed_dir>/<subhalo_id>.json.

Resumable — skips subhalos that already have a result JSON.

Usage:
    python 03_process_batch.py [--config batch_config.yaml]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

# Make the parent's tnggalaxylab importable (script lives in batch_tng50/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnggalaxylab.fourier import (
    compute_fourier,
    global_lopsidedness,
    compute_pattern_diagnostics,
    bootstrap_fourier,
    rotation_curve_tracer,
)
from tnggalaxylab.core.center import Centering

# Import the load_tng_stars / centring / alignment helpers from the demo script
# (they're generic enough to reuse)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_tng50_subhalo import (
    load_tng_stars, find_center_power2003, align_disk_face_on,
    fit_scale_length_and_sigma,
)


def process_single(cutout_path: Path, cfg: dict) -> dict:
    """Run Stage 4 pipeline on one cutout.  Returns a dict of results."""
    # 1. Load + convert units
    data = load_tng_stars(str(cutout_path))
    pos, vel, mass = data["pos"], data["vel"], data["mass"]

    # 2. Centre + face-on
    center = find_center_power2003(pos, data["potential"], mass)
    pos -= center
    vel -= np.average(vel, axis=0, weights=mass)
    pos_rot, vel_rot, _ = align_disk_face_on(pos, vel, mass, r_align=10.0)

    # 3. Select disk particles
    x, y, z = pos_rot[:, 0], pos_rot[:, 1], pos_rot[:, 2]
    R_cyl = np.sqrt(x**2 + y**2)
    disk_mask = (np.abs(z) < 3.0) & (R_cyl < 30.0)
    x_d, y_d, m_d = x[disk_mask], y[disk_mask], mass[disk_mask]

    if disk_mask.sum() < 500:
        return {
            "n_disk_particles": int(disk_mask.sum()),
            "error": "insufficient_disk_particles",
        }

    # 4. Fourier + R_d + Sigma(R)
    r_out = float(cfg["r_out_kpc"])
    n_bins = int(cfg["n_bins"])
    m_max = int(cfg["m_max"])
    profile = compute_fourier(
        x_d, y_d, m_d, method="particles",
        r_in=0.0, r_out=r_out, n_bins=n_bins, m_max=m_max,
    )
    R_d, sigma_R = fit_scale_length_and_sigma(
        np.column_stack([x_d, y_d]), m_d, profile.r_bins,
        z_cut=3.0, pos_full=pos_rot[disk_mask],
    )

    # 5. Global lopsidedness (both R&Z95 lit-avg and Jog with Sigma)
    gm_default = global_lopsidedness(profile, scale_length=R_d)
    gm_jog     = global_lopsidedness(profile, scale_length=R_d,
                                       surface_density=sigma_R)

    # 6. Pattern diagnostics
    diag = compute_pattern_diagnostics(profile, bar_threshold=0.2)

    # 7. Bootstrap (uses cfg n_bootstrap)
    boot = bootstrap_fourier(
        x_d, y_d, m_d, method="particles",
        n_bootstrap=int(cfg["n_bootstrap"]), seed=42,
        r_in=0.0, r_out=r_out, n_bins=n_bins, m_max=m_max,
        scale_length=R_d,
    )

    # 8. Rotation curve — tracer only (physical answer)
    pos_disk = pos_rot[disk_mask]; vel_disk = vel_rot[disk_mask]
    rc = rotation_curve_tracer(
        pos_disk[:, 0], pos_disk[:, 1], pos_disk[:, 2],
        vel_disk[:, 0], vel_disk[:, 1], vel_disk[:, 2],
        m_d, r_in=0.5, r_out=min(8*R_d, 20.0),
        n_bins=25, z_max=3.0,
    )
    v_max_finite = np.isfinite(rc.v_circ) & (rc.v_circ > 0)
    v_max     = float(np.nanmax(rc.v_circ[v_max_finite])) if v_max_finite.any() else np.nan
    R_v_max   = float(rc.r_bins[v_max_finite][np.argmax(rc.v_circ[v_max_finite])]) \
                    if v_max_finite.any() else np.nan

    # 9. Assemble results
    return {
        "simulation":         data["simulation"],
        "snapshot":           int(data["snapshot"]),
        "redshift":           float(data["redshift"]),
        "n_stellar_particles": int(len(data["pos"])),
        "n_disk_particles":   int(disk_mask.sum()),
        "M_star_Msun":        float(data["mass"].sum()),
        "R_d_kpc":            float(R_d),
        "center_kpc":         [float(c) for c in center],
        "aperture_kpc":       [float(gm_default.r_range_kpc[0]),
                               float(gm_default.r_range_kpc[1])],
        # Fourier amplitudes (bootstrap means and stds)
        "A1_literature":      float(gm_default.A1_literature),
        "A2_literature":      float(gm_default.A2_literature),
        "A1_integral_area":   float(gm_default.A1_integral),
        "A1_integral_jog":    float(gm_jog.A1_integral),
        "A2_integral_area":   float(gm_default.A2_integral),
        "A2_integral_jog":    float(gm_jog.A2_integral),
        "A1_bootstrap_mean":  float(boot["global_A1_mean"]),
        "A1_bootstrap_std":   float(boot["global_A1_std"]),
        "A2_bootstrap_mean":  float(boot["global_A2_mean"]),
        "A2_bootstrap_std":   float(boot["global_A2_std"]),
        # Pattern diagnostics
        "dominant_mode":      int(diag.dominant_mode),
        "bar_length_kpc":     float(diag.bar_length),
        "bar_angle_deg":      float(np.degrees(diag.bar_angle))
                              if not np.isnan(diag.bar_angle) else None,
        "pattern_coherence":  float(diag.pattern_coherence)
                              if not np.isnan(diag.pattern_coherence) else None,
        "phase_scatter_m2_deg": float(np.degrees(diag.phase_scatter_m2))
                              if not np.isnan(diag.phase_scatter_m2) else None,
        "bar_length_bootstrap_mean": float(boot["bar_length_mean"])
                              if boot["bar_length_mean"] is not None else None,
        "bar_length_bootstrap_std":  float(boot["bar_length_std"])
                              if boot["bar_length_std"] is not None else None,
        # Kinematics
        "v_max_kms":          v_max,
        "R_v_max_kpc":        R_v_max,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="batch_config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    cutouts_dir   = Path(cfg["cutouts_dir"])
    processed_dir = Path(cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    cutouts = sorted(cutouts_dir.glob("cutout_*.hdf5"))
    print(f"Processing {len(cutouts)} cutouts...")
    print(f"  Cutouts:   {cutouts_dir}")
    print(f"  Results:   {processed_dir}")
    print(f"  Bootstrap iterations: {cfg['n_bootstrap']}\n")

    n_ok = n_skip = n_fail = 0
    t_start = time.perf_counter()

    for i, cpath in enumerate(cutouts, 1):
        sid = int(cpath.stem.split("_")[-1])
        result_path = processed_dir / f"{sid}.json"

        if result_path.exists():
            n_skip += 1
            continue

        try:
            t0 = time.perf_counter()
            result = process_single(cpath, cfg)
            result["subhalo_id"] = sid
            result["_processing_time_s"] = time.perf_counter() - t0

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

            if "error" in result:
                n_fail += 1
                print(f"  [{i:>4d}/{len(cutouts)}] subhalo {sid}: "
                      f"skipped ({result['error']})")
            else:
                n_ok += 1
                print(f"  [{i:>4d}/{len(cutouts)}] subhalo {sid}: "
                      f"OK  A1={result['A1_bootstrap_mean']:.3f}±{result['A1_bootstrap_std']:.3f}  "
                      f"R_d={result['R_d_kpc']:.2f} kpc  "
                      f"({result['_processing_time_s']:.1f}s)")
        except Exception as e:
            n_fail += 1
            print(f"  [{i:>4d}/{len(cutouts)}] subhalo {sid}: FAIL — {e}")
            traceback.print_exc(limit=1)

    dt = time.perf_counter() - t_start
    print(f"\n[Done in {dt:.1f}s]")
    print(f"  Processed:  {n_ok}")
    print(f"  Skipped:    {n_skip}")
    print(f"  Failed:     {n_fail}")


if __name__ == "__main__":
    main()
