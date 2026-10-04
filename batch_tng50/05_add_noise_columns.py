"""
05_add_noise_columns.py -- shot-noise columns for an existing catalogue
No bootstrap re-run; same centring / alignment / disc cut as 03.
 (a) JSON mode (default): <processed_dir>/*.json + cutouts  -> keys added to JSON
 (b) catalogue mode: --from-catalog catalog.csv --source il  -> noise_columns.csv
--merge writes catalog_v2.csv / catalog_filtered_v2.csv (originals untouched).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tnggalaxylab.fourier import compute_fourier
from tnggalaxylab.fourier.noise import (
    noise_floor_profile, noise_corrected_global, bootstrap_bias_corrected)
from analyze_tng50_subhalo import (
    load_tng_stars, find_center_power2003, align_disk_face_on)

BOX_CKPC_H = {"TNG50-1": 35000.0, "TNG100-1": 75000.0, "TNG300-1": 205000.0}


def load_il(basepath, sid, snap=99, h=0.6774, box=None):
    import illustris_python as il
    flds = ["Coordinates", "Velocities", "Masses", "GFM_StellarFormationTime", "Potential"]
    s = il.snapshot.loadSubhalo(basepath, snap, int(sid), "stars", fields=flds)
    z = float(il.groupcat.loadHeader(basepath, snap)["Redshift"])
    a = 1.0 / (1.0 + z)
    m = s["GFM_StellarFormationTime"] > 0
    if box is not None and m.any():
        _c = s["Coordinates"].astype(np.float64)
        _ref = _c[m][np.argmin(s["Potential"][m])]
        s["Coordinates"] = (_c - _ref + 0.5 * box) % box - 0.5 * box + _ref
    return dict(pos=s["Coordinates"][m].astype(np.float64) * a / h,
                vel=s["Velocities"][m].astype(np.float64) * np.sqrt(a),
                mass=s["Masses"][m].astype(np.float64) * 1e10 / h,
                potential=s["Potential"][m].astype(np.float64), a=a, h=h)


def add_noise(data, res, cfg, apertures, box_kpc):
    pos, vel, mass = data["pos"], data["vel"], data["mass"]
    box_wrap = int(np.any(pos.max(axis=0) - pos.min(axis=0) > 0.5 * box_kpc))
    center = find_center_power2003(pos, data["potential"], mass)
    pos = pos - center
    vel = vel - np.average(vel, axis=0, weights=mass)
    pos_rot, _, _ = align_disk_face_on(pos, vel, mass, r_align=10.0)
    x, y, z = pos_rot[:, 0], pos_rot[:, 1], pos_rot[:, 2]
    d = (np.abs(z) < 3.0) & (np.hypot(x, y) < 30.0)
    x_d, y_d, m_d = x[d], y[d], mass[d]
    fk = dict(method="particles", r_in=0.0, r_out=float(cfg["r_out_kpc"]),
              n_bins=int(cfg["n_bins"]), m_max=int(cfg["m_max"]))
    prof = compute_fourier(x_d, y_d, m_d, **fk)
    noise = noise_floor_profile(x_d, y_d, m_d, n_shuffle=5, seed=42, **fk)
    Rd = float(res["R_d_kpc"])
    nc = noise_corrected_global(x_d, y_d, m_d, scale_length=Rd,
                                profile=prof, noise_profile=noise, **fk)
    out = {"A1_noise": nc["A1_noise"], "A2_noise": nc["A2_noise"],
           "A1_corr": nc["A1_corr"], "A2_corr": nc["A2_corr"],
           "neff_aperture": nc["neff_aperture"], "A1_lit_recheck": nc["A1"],
           "aperture_R_in_kpc": 1.5 * Rd, "aperture_R_out_kpc": 2.5 * Rd,
           "A1_bootstrap_bc": bootstrap_bias_corrected(
               float(res.get("A1_literature", np.nan)), float(res.get("A1_bootstrap_mean", np.nan))),
           "A2_bootstrap_bc": bootstrap_bias_corrected(
               float(res.get("A2_literature", np.nan)), float(res.get("A2_bootstrap_mean", np.nan))),
           "box_wrap": box_wrap}
    for lo, hi in apertures:
        f = noise_corrected_global(x_d, y_d, m_d, r_range=(lo, hi),
                                   profile=prof, noise_profile=noise, **fk)
        tag = f"{lo:g}_{hi:g}kpc"
        out[f"A1_lit_{tag}"] = f["A1"]
        out[f"A1_noise_{tag}"] = f["A1_noise"]
        out[f"A1_corr_{tag}"] = f["A1_corr"]
    return out


def report(i, n, sid, res, out):
    a = float(res["A1_literature"])
    flag = "  <-- RECHECK FARQI" if abs(out["A1_lit_recheck"] - a) > 1e-3 * max(a, 1e-3) else ""
    flag += "  <-- BOX CHEGARASI" if out["box_wrap"] else ""
    print(f"[{i:>5d}/{n}] {sid}: A1={a:.4f} recheck={out['A1_lit_recheck']:.4f} "
          f"noise={out['A1_noise']:.4f} corr={out['A1_corr']:.4f}{flag}", flush=True)


def merge(catalog, noise_df):
    for src in [catalog, catalog.with_name(catalog.stem + "_filtered.csv")]:
        if not src.exists():
            continue
        cat = pd.read_csv(src)
        new = [c for c in noise_df.columns if c != "subhalo_id"]
        cat = cat.drop(columns=[c for c in new if c in cat.columns])
        out = cat.merge(noise_df, on="subhalo_id", how="left")
        dst = src.with_name(src.stem + "_v2.csv")
        out.to_csv(dst, index=False)
        print(f"merge: {dst}  ({len(out)} qator, {out.A1_corr.notna().sum()} ta A1_corr)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="batch_config.yaml")
    ap.add_argument("--from-catalog", default=None)
    ap.add_argument("--source", choices=["cutout", "il"], default="cutout")
    ap.add_argument("--sim", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    apertures = cfg.get("fixed_apertures_kpc", [[3.5, 6.0], [5.0, 8.0]])
    sim = args.sim or ("TNG100-1" if "tng100" in args.config.lower() else "TNG50-1")
    box_kpc = BOX_CKPC_H.get(sim, 35000.0) / 0.6774
    basepath = f"/home/tnguser/sims.TNG/{sim}/output"
    print(f"sim={sim}  box={box_kpc:.0f} kpc  apertures={apertures}", flush=True)
    t0 = time.perf_counter()

    if args.from_catalog:
        catalog = Path(args.from_catalog)
        out_csv = catalog.with_name("noise_columns.csv")
        cat = pd.read_csv(catalog)
        done = set(pd.read_csv(out_csv).subhalo_id) if out_csv.exists() else set()
        rows = cat[~cat.subhalo_id.isin(done)]
        if args.limit:
            rows = rows.head(args.limit)
        print(f"{len(rows)} galaktika qoldi ({len(done)} tayyor) -> {out_csv}", flush=True)
        for i, (_, r) in enumerate(rows.iterrows(), 1):
            sid = int(r.subhalo_id)
            try:
                data = (load_il(basepath, sid, box=BOX_CKPC_H.get(sim)) if args.source == "il" else
                        load_tng_stars(str(Path(cfg["cutouts_dir"]) / f"cutout_{sid}.hdf5")))
                out = add_noise(data, r, cfg, apertures, box_kpc)
                report(i, len(rows), sid, r, out)
                pd.DataFrame([{"subhalo_id": sid, **out}]).to_csv(
                    out_csv, mode="a", header=not out_csv.exists(), index=False)
            except Exception as e:
                print(f"[{i:>5d}/{len(rows)}] {sid}: XATO {type(e).__name__}: {e}", flush=True)
        if args.merge and out_csv.exists():
            merge(catalog, pd.read_csv(out_csv))
    else:
        processed_dir, cutouts_dir = Path(cfg["processed_dir"]), Path(cfg["cutouts_dir"])
        files = sorted(processed_dir.glob("*.json"))
        if args.limit:
            files = files[:args.limit]
        print(f"{len(files)} JSON ({processed_dir})", flush=True)
        recs = []
        keep = ("A1_noise", "A2_noise", "A1_corr", "A2_corr", "A1_lit_", "A1_bootstrap_bc",
                "A2_bootstrap_bc", "neff_aperture", "aperture_R_", "box_wrap")
        for i, jp in enumerate(files, 1):
            res = json.loads(jp.read_text())
            if "error" in res:
                continue
            sid = int(res.get("subhalo_id", jp.stem))
            if "box_wrap" not in res:
                try:
                    data = (load_il(basepath, sid, box=BOX_CKPC_H.get(sim)) if args.source == "il" else
                            load_tng_stars(str(cutouts_dir / f"cutout_{sid}.hdf5")))
                    out = add_noise(data, res, cfg, apertures, box_kpc)
                    report(i, len(files), sid, res, out)
                    res.update(out)
                    jp.write_text(json.dumps(res, indent=2))
                except Exception as e:
                    print(f"[{i:>5d}/{len(files)}] {sid}: XATO {type(e).__name__}: {e}", flush=True)
                    continue
            recs.append({"subhalo_id": sid, **{k: res[k] for k in res if k.startswith(keep)}})
        if args.merge and recs:
            merge(processed_dir.parent / "catalog.csv", pd.DataFrame(recs))
    print(f"\ntayyor ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
