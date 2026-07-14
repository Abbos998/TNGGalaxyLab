"""
04_build_catalog.py
====================
Aggregate all per-subhalo JSON results into a single catalog:
  - FITS binary table  (for astronomical downstream tools)
  - CSV                 (for spreadsheets, easy inspection)

Usage:
    python 04_build_catalog.py [--config batch_config.yaml]
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# FITS support via astropy (already in your requirements.txt)
try:
    from astropy.table import Table
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("Warning: astropy not available → skipping FITS output", file=sys.stderr)


# Ordered column schema (order matters for FITS/CSV consistency)
CATALOG_COLUMNS = [
    ("subhalo_id",                "int64",   "Subhalo ID in the simulation"),
    ("simulation",                "str",     "Simulation name"),
    ("snapshot",                  "int32",   "Snapshot number"),
    ("redshift",                  "float64", "Cosmological redshift"),
    ("n_stellar_particles",       "int64",   "Number of stellar particles (post-wind cut)"),
    ("n_disk_particles",          "int64",   "Particles in |z|<3 kpc, R<30 kpc"),
    ("M_star_Msun",               "float64", "Total stellar mass [M_sun]"),
    ("R_d_kpc",                   "float64", "Exponential disk scale length [kpc]"),
    ("aperture_R_in_kpc",         "float64", "1.5 R_d apertura inner edge"),
    ("aperture_R_out_kpc",        "float64", "2.5 R_d apertura outer edge"),
    ("A1_literature",             "float64", "A_1 (R&Z95 literature average)"),
    ("A2_literature",             "float64", "A_2 (R&Z95 literature average)"),
    ("A1_integral_area",          "float64", "A_1 area-weighted integral"),
    ("A1_integral_jog",           "float64", "A_1 Jog-2002 Sigma-weighted integral"),
    ("A2_integral_area",          "float64", "A_2 area-weighted integral"),
    ("A2_integral_jog",           "float64", "A_2 Jog-2002 Sigma-weighted integral"),
    ("A1_bootstrap_mean",         "float64", "A_1 bootstrap mean"),
    ("A1_bootstrap_std",          "float64", "A_1 bootstrap 1-sigma"),
    ("A2_bootstrap_mean",         "float64", "A_2 bootstrap mean"),
    ("A2_bootstrap_std",          "float64", "A_2 bootstrap 1-sigma"),
    ("dominant_mode",             "int32",   "Fourier mode with highest disk-avg A"),
    ("bar_length_kpc",            "float64", "Aguerri+2005 phase-coherent bar length"),
    ("bar_angle_deg",             "float64", "Bar phase angle [deg]"),
    ("pattern_coherence",         "float64", "Fraction of coherent m=2 bins [0..1]"),
    ("phase_scatter_m2_deg",      "float64", "Circular sigma of m=2 phase [deg]"),
    ("bar_length_bootstrap_mean", "float64", "Bar length bootstrap mean [kpc]"),
    ("bar_length_bootstrap_std",  "float64", "Bar length bootstrap 1-sigma [kpc]"),
    ("v_max_kms",                 "float64", "Peak circular velocity (tracer method) [km/s]"),
    ("R_v_max_kpc",               "float64", "Radius of v_max [kpc]"),
]


def _sanitize(value, dtype):
    """Convert None / NaN to a FITS-safe value based on dtype."""
    if value is None:
        return np.nan if dtype.startswith("float") else -1
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="batch_config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["processed_dir"])
    catalog_base  = Path(cfg["catalog_file"])
    catalog_base.parent.mkdir(parents=True, exist_ok=True)

    result_files = sorted(processed_dir.glob("*.json"))
    if not result_files:
        print(f"ERROR: no results in {processed_dir}", file=sys.stderr)
        sys.exit(1)

    # Load and filter (drop rows with errors)
    rows = []
    n_err = 0
    for rf in result_files:
        with open(rf) as f:
            d = json.load(f)
        if "error" in d:
            n_err += 1
            continue
        rows.append(d)

    print(f"Loaded {len(rows)} clean results (skipped {n_err} with errors)")

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    csv_path = catalog_base.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([c[0] for c in CATALOG_COLUMNS])
        for r in rows:
            row_out = []
            for col_name, dtype, _ in CATALOG_COLUMNS:
                v = r.get(col_name)
                v = _sanitize(v, dtype)
                if dtype.startswith("float") and isinstance(v, float):
                    row_out.append(f"{v:.6g}")
                else:
                    row_out.append(v)
            writer.writerow(row_out)
    print(f"[✓] CSV catalog: {csv_path}")

    # ------------------------------------------------------------------
    # FITS output (astropy Table)
    # ------------------------------------------------------------------
    if HAS_ASTROPY:
        columns = {}
        for col_name, dtype, _ in CATALOG_COLUMNS:
            data = [_sanitize(r.get(col_name), dtype) for r in rows]
            if dtype == "str":
                columns[col_name] = np.array(data, dtype="U64")
            elif dtype.startswith("int"):
                columns[col_name] = np.array(data, dtype=dtype)
            else:
                columns[col_name] = np.array(data, dtype="float64")

        tbl = Table(columns)
        # Attach unit / description metadata
        for col_name, _, description in CATALOG_COLUMNS:
            tbl[col_name].description = description

        # Header metadata
        tbl.meta["SIMULATION"] = rows[0]["simulation"]
        tbl.meta["SNAPSHOT"]   = rows[0]["snapshot"]
        tbl.meta["REDSHIFT"]   = rows[0]["redshift"]
        tbl.meta["N_ROWS"]     = len(rows)
        tbl.meta["PIPELINE"]   = "TNGGalaxyLab Stage 4"
        tbl.meta["A_NORM"]     = "R&Z95"
        tbl.meta["BAR_ALGO"]   = "Aguerri+2005 phase-coherent"
        tbl.meta["N_BOOT"]     = int(cfg["n_bootstrap"])

        fits_path = catalog_base.with_suffix(".fits")
        tbl.write(fits_path, overwrite=True)
        print(f"[✓] FITS catalog: {fits_path}")

    # ------------------------------------------------------------------
    # Quick summary statistics
    # ------------------------------------------------------------------
    A1_arr = np.array([r["A1_bootstrap_mean"] for r in rows])
    A2_arr = np.array([r["A2_bootstrap_mean"] for r in rows])
    coh_arr = np.array([r["pattern_coherence"] or 0 for r in rows])
    Ms_arr  = np.array([r["M_star_Msun"] for r in rows])

    print("\n=== Catalog summary ===")
    print(f"  N galaxies:         {len(rows)}")
    print(f"  M_star median:      {np.median(Ms_arr):.2e} M_sun")
    print(f"  A_1 median:         {np.median(A1_arr):.3f}")
    print(f"  A_1 mean:           {np.mean(A1_arr):.3f} ± {np.std(A1_arr):.3f}")
    print(f"  A_2 median:         {np.median(A2_arr):.3f}")
    print(f"  A_1 > 0.2 fraction: {(A1_arr > 0.2).sum() / len(A1_arr):.2%}  "
          f"(strong lopsided)")
    print(f"  Coherent m=2 bars:  {(coh_arr > 0.5).sum() / len(coh_arr):.2%}  "
          f"(coherence > 0.5)")


if __name__ == "__main__":
    main()
