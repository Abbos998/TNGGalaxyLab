"""
compute_disc_quality.py
========================
Compute kinematic disc-to-total ratio (D/T) for each subhalo,
using stellar circularity parameter eps = J_z / J_circ(E).

D/T = fraction of stellar mass with |eps| > 0.7 (disc-dominated orbits).

This is the STANDARD kinematic disc-quality criterion used by
Vogelsberger+2014, Genel+2015, Lokas 2022, Varela-Lavin+2023.

Runs on both TNG100 (6717 galaxies) and TNG50 (915 galaxies) using
local snapshot data — no HTTP API needed.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "/home/tnguser")
import illustris_python as il


def compute_dt_ratio(basepath: str, snap: int, sid: int) -> dict:
    """Return kinematic D/T and related quality metrics for one subhalo."""
    # Load stellar particles with kinematic fields
    fields = ["Coordinates", "Velocities", "Masses",
              "GFM_StellarFormationTime", "Potential"]
    try:
        stars = il.snapshot.loadSubhalo(basepath, snap, sid, 'stars',
                                         fields=fields)
    except Exception as e:
        return dict(n_star=0, DT_ratio=np.nan, kappa_rot=np.nan,
                    error=f"load: {e}")
    if not isinstance(stars, dict) or stars.get("count", 0) == 0:
        return dict(n_star=0, DT_ratio=np.nan, kappa_rot=np.nan,
                    error="no stars")

    # Wind particle filter (GFM_StellarFormationTime < 0 = wind)
    mask = stars["GFM_StellarFormationTime"] > 0
    if mask.sum() < 100:
        return dict(n_star=int(mask.sum()), DT_ratio=np.nan,
                    kappa_rot=np.nan, error="too few stars")

    pos  = stars["Coordinates"][mask]     # ckpc/h
    vel  = stars["Velocities"][mask]      # sqrt(a) km/s
    mass = stars["Masses"][mask]          # 10^10 Msun/h
    pot  = stars["Potential"][mask]       # (km/s)^2 / a

    # Load header for scale factor + h
    hdr = il.groupcat.loadHeader(basepath, snap)
    h = hdr["HubbleParam"]
    a = hdr["Time"]

    # Convert to physical units
    pos_phys  = pos * a / h              # kpc
    vel_phys  = vel * np.sqrt(a)          # km/s peculiar
    mass_phys = mass * 1e10 / h           # Msun

    # Center on most bound (min potential)
    idx_center = np.argmin(pot)
    pos_phys -= pos_phys[idx_center]
    # Periodic-boundary unwrap (physical kpc, snapshot 99, a = 1)
    _L = next((b for k, b in {"TNG50-1": 35000.0, "TNG100-1": 75000.0,
                               "TNG300-1": 205000.0}.items() if k in basepath), None)
    if _L is not None:
        _L = _L / 0.6774
        pos_phys = (pos_phys + 0.5 * _L) % _L - 0.5 * _L
    # Bulk velocity: mass-weighted median (robust)
    v_bulk = np.array([np.average(vel_phys[:, i], weights=mass_phys)
                       for i in range(3)])
    vel_phys -= v_bulk

    # Compute angular momentum vector of inner disc (R < 10 kpc)
    r3d = np.linalg.norm(pos_phys, axis=1)
    inner = (r3d > 1.0) & (r3d < 10.0)
    if inner.sum() < 50:
        return dict(n_star=int(mask.sum()), DT_ratio=np.nan,
                    kappa_rot=np.nan, error="no inner stars")

    L_particle = np.cross(pos_phys[inner], vel_phys[inner]) \
                 * mass_phys[inner, None]
    L_total = L_particle.sum(axis=0)
    L_norm = np.linalg.norm(L_total)
    if L_norm < 1e-6:
        return dict(n_star=int(mask.sum()), DT_ratio=np.nan,
                    kappa_rot=np.nan, error="no angular momentum")
    L_hat = L_total / L_norm

    # Align z-axis with L_hat
    def _rotmat_from_z(z_new):
        z_old = np.array([0, 0, 1.0])
        axis = np.cross(z_old, z_new)
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return np.eye(3)
        axis /= n
        c = np.dot(z_old, z_new); s = np.sqrt(1 - c*c)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + s*K + (1-c)*(K @ K)

    R = _rotmat_from_z(L_hat).T
    pos_r = pos_phys @ R.T
    vel_r = vel_phys @ R.T

    # For all stars: compute eps = j_z / j_circ(E)
    x, y, z = pos_r[:, 0], pos_r[:, 1], pos_r[:, 2]
    vx, vy, vz = vel_r[:, 0], vel_r[:, 1], vel_r[:, 2]
    R_cyl = np.sqrt(x*x + y*y)
    j_z = x*vy - y*vx                     # angular momentum along L_hat
    # Kinetic + potential energy per unit mass
    E_kin = 0.5 * (vx*vx + vy*vy + vz*vz)
    E_tot = E_kin + pot                    # per unit mass, (km/s)^2

    # j_circ(E) = R_circ(E) * v_circ(E), computed empirically:
    # For each star, find the maximum j_z of stars with energy <= this star's E,
    # which approximates the circular orbit angular momentum at that binding energy.
    idx_sort = np.argsort(E_tot)
    j_sorted = j_z[idx_sort]
    # Running maximum from most-bound to least-bound
    j_circ_sorted = np.maximum.accumulate(np.abs(j_sorted))
    j_circ = np.empty_like(j_z)
    j_circ[idx_sort] = j_circ_sorted
    j_circ = np.where(j_circ > 1e-6, j_circ, 1e-6)
    eps = j_z / j_circ

    # D/T ratio: mass fraction with eps > 0.7 (disc-supported)
    disc_mask = eps > 0.7
    DT = mass_phys[disc_mask].sum() / mass_phys.sum()

    # kappa_rot = fraction of kinetic energy in ordered rotation
    E_rot = 0.5 * mass_phys * (j_z / (R_cyl + 1e-6)) ** 2
    E_kin_tot = (mass_phys * E_kin).sum()
    kappa_rot = E_rot.sum() / E_kin_tot if E_kin_tot > 0 else np.nan

    return dict(n_star=int(mask.sum()), DT_ratio=float(DT),
                kappa_rot=float(kappa_rot), error="")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: <output_dir>/disc_quality.csv)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sim = cfg["simulation"]
    snap = int(cfg["snapshot"])
    basepath = f"/home/tnguser/sims.TNG/{sim}/output"

    ids_file = Path(cfg["output_dir"]) / "subhalo_ids.txt"
    sids = []
    with open(ids_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sids.append(int(line.split()[0]))
    print(f"Loaded {len(sids)} subhalo IDs from {ids_file}")

    out = args.output or f"{cfg['output_dir']}/disc_quality.csv"
    print(f"Writing to {out}")

    t0 = time.time()
    rows = []
    for i, sid in enumerate(sids, 1):
        r = compute_dt_ratio(basepath, snap, sid)
        r["subhalo_id"] = sid
        rows.append(r)
        if i % 200 == 0:
            dt = time.time() - t0
            rate = i / dt
            eta = (len(sids) - i) / rate / 60
            n_ok = sum(1 for x in rows if not x.get("error"))
            print(f"  {i}/{len(sids)}  ok={n_ok}  {rate:.1f}/s  ETA {eta:.1f} min")

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\n[DONE] Wrote {len(df)} rows to {out}")
    print(f"  D/T > 0.5: {(df['DT_ratio'] > 0.5).sum()}")
    print(f"  D/T > 0.4: {(df['DT_ratio'] > 0.4).sum()}")
    print(f"  Errors: {df['error'].astype(bool).sum()}")


if __name__ == "__main__":
    main()
