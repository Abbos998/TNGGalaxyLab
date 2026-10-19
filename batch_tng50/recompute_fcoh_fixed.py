"""Recompute m=2 diagnostics (bar_angle, L_bar, f_coh, phase_scatter_m2) with
the period-pi phase fix. No bootstrap; compares with the v16 catalogue."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from tnggalaxylab.fourier import compute_fourier, compute_pattern_diagnostics
from analyze_tng50_subhalo import load_tng_stars, find_center_power2003, align_disk_face_on


def wrap_pi(d):
    return (d + np.pi / 2) % np.pi - np.pi / 2


def m2_diagnostics_fixed(profile, bar_threshold=0.2, coherence_tol=np.pi / 8,
                         r_min=0.0, r_max=None):
    if r_max is None:
        r_max = profile.r_out
    r = profile.r_bins
    mask = (r >= r_min) & (r <= r_max)
    r_sel = r[mask]; A2 = profile.A(2)[mask]; P2 = profile.Phi(2)[mask]
    bar_angle = (float(0.5 * np.angle(np.sum(A2 / A2.sum() * np.exp(2j * P2))))
                 if A2.sum() > 0 else np.nan)
    bar_length = 0.0
    if (A2 > bar_threshold).any() and np.isfinite(bar_angle):
        k = int(np.argmax(A2)); ref = float(P2[k])
        for i in range(k, len(r_sel)):
            if A2[i] <= bar_threshold or abs(wrap_pi(P2[i] - ref)) > coherence_tol:
                break
            bar_length = float(r_sel[i])
    if np.isfinite(bar_angle) and len(P2) > 0:
        s = A2 > bar_threshold
        f_coh = float(np.mean(np.abs(wrap_pi(P2[s] - bar_angle)) < coherence_tol)) if s.any() else 0.0
    else:
        f_coh = np.nan
    if len(P2):
        Rb = float(np.clip(np.abs(np.mean(np.exp(2j * P2))), 1e-15, 1.0))
        scatter = 0.5 * np.sqrt(max(-2.0 * np.log(Rb), 0.0))
    else:
        scatter = np.nan
    return dict(bar_angle=bar_angle, bar_length=bar_length,
                pattern_coherence=f_coh, phase_scatter_m2=scatter)


def profile_for(cutout, r_out=15.0, n_bins=40, m_max=4):
    data = load_tng_stars(str(cutout))
    pos, vel, mass = data["pos"], data["vel"], data["mass"]
    center = find_center_power2003(pos, data["potential"], mass)
    pos = pos - center
    vel = vel - np.average(vel, axis=0, weights=mass)
    pos_rot, _, _ = align_disk_face_on(pos, vel, mass, r_align=10.0)
    x, y, z = pos_rot[:, 0], pos_rot[:, 1], pos_rot[:, 2]
    disk = (np.abs(z) < 3.0) & (np.hypot(x, y) < 30.0)
    if disk.sum() < 500:
        return None
    return compute_fourier(x[disk], y[disk], mass[disk], method="particles",
                           r_in=0.0, r_out=r_out, n_bins=n_bins, m_max=m_max)


def summary(cat, col, label):
    f = cat[col]; A = cat["A1_bootstrap_mean"]
    out = [f"--- {label} ({col}), N = {len(cat)} ---"]
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = f > thr
        out.append(f"  f_coh > {thr:.1f}: {m.sum():5d}  ({100*m.mean():5.1f}%)"
                   f"   median A1_boot = {A[m].median():.3f}"
                   f"   median A1_lit = {cat.loc[m, 'A1_literature'].median():.3f}")
    m5 = f > 0.5
    out.append(f"  overlap A1_boot>0.1 & f_coh>0.5 : {int(((A > 0.1) & m5).sum())}")
    big = A > 0.2
    out.append(f"  A1_boot>0.2: {int(big.sum())}, of which f_coh>0.5: {int((big & m5).sum())}")
    if "A1_corr" in cat:
        bc = cat["A1_corr"] > 0.2
        out.append(f"  A1_corr>0.2: {int(bc.sum())}, of which f_coh>0.5: {int((bc & m5).sum())}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutouts", required=True); ap.add_argument("--catalog", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r_out", type=float, default=15.0)
    ap.add_argument("--n_bins", type=int, default=40); ap.add_argument("--m_max", type=int, default=4)
    a = ap.parse_args()
    cat = pd.read_csv(a.catalog); cdir = Path(a.cutouts); rows = []; t0 = time.perf_counter()
    for k, sid in enumerate(cat["subhalo_id"].astype(int), 1):
        cp = cdir / f"cutout_{sid}.hdf5"; rec = dict(subhalo_id=sid)
        if not cp.exists():
            rec["status"] = "missing_cutout"; rows.append(rec); continue
        try:
            prof = profile_for(cp, a.r_out, a.n_bins, a.m_max)
            if prof is None:
                rec["status"] = "few_disc_particles"
            else:
                old = compute_pattern_diagnostics(prof, bar_threshold=0.2)
                new = m2_diagnostics_fixed(prof, bar_threshold=0.2)
                rec.update(status="ok", fcoh_old_repro=old.pattern_coherence,
                           Lbar_old_repro=old.bar_length,
                           pattern_coherence=new["pattern_coherence"],
                           bar_length_kpc=new["bar_length"],
                           bar_angle_deg=np.degrees(new["bar_angle"]),
                           phase_scatter_m2_deg=np.degrees(new["phase_scatter_m2"]))
        except Exception as e:
            rec["status"] = f"fail: {e}"
        rows.append(rec)
        if k % 25 == 0 or k == len(cat):
            print(f"  [{k:5d}/{len(cat)}]  {(time.perf_counter()-t0)/k:.2f} s/galaxy", flush=True)

    new = pd.DataFrame(rows).set_index("subhalo_id")
    for c in ["fcoh_old_repro", "Lbar_old_repro", "pattern_coherence",
              "bar_length_kpc", "bar_angle_deg", "phase_scatter_m2_deg"]:
        if c not in new:
            new[c] = np.nan
    ok = new["status"] == "ok"
    print(f"\nok: {ok.sum()}  other: {(~ok).sum()}")
    print(new.loc[~ok, "status"].value_counts().to_string())
    out = cat.copy().set_index("subhalo_id")
    for c in ["bar_length_kpc", "bar_angle_deg", "pattern_coherence", "phase_scatter_m2_deg"]:
        out[c + "_v16"] = out[c]
        out.loc[new.index[ok], c] = new.loc[ok, c]
    chk = new.loc[ok, ["fcoh_old_repro", "Lbar_old_repro"]].join(
        cat.set_index("subhalo_id")[["pattern_coherence", "bar_length_kpc"]])
    d_f = (chk["fcoh_old_repro"] - chk["pattern_coherence"]).abs()
    d_L = (chk["Lbar_old_repro"] - chk["bar_length_kpc"]).abs()
    L = [f"Catalogue: {a.catalog}",
         "Reproduction of v16 values (should be ~0 except PBC-unwrapped galaxies):",
         f"  |d f_coh| > 1e-6 : {int((d_f > 1e-6).sum())} galaxies",
         f"  |d L_bar| > 1e-6 : {int((d_L > 1e-6).sum())} galaxies", ""]
    out = out.reset_index()
    L += summary(out, "pattern_coherence_v16", "v16 (old)")
    L += [""] + summary(out, "pattern_coherence", "fixed")
    ch = (out["pattern_coherence"] > 0.5) != (out["pattern_coherence_v16"] > 0.5)
    L += ["", f"Galaxies whose f_coh>0.5 flag changed: {int(ch.sum())}"
              f"  (gained: {int((ch & (out['pattern_coherence'] > 0.5)).sum())},"
              f" lost: {int((ch & (out['pattern_coherence'] <= 0.5)).sum())})"]
    so = int(np.isclose(out["bar_length_kpc_v16"], 14.8125).sum())
    sn = int(np.isclose(out["bar_length_kpc"], 14.8125).sum())
    L += [f"L_bar saturated at 14.8125 kpc: old {so}/{len(out)} ({100*so/len(out):.1f}%),"
          f" new {sn}/{len(out)} ({100*sn/len(out):.1f}%)"]
    if 668581 in set(out["subhalo_id"]):
        r = out.loc[out["subhalo_id"] == 668581].iloc[0]
        L.append(f"subhalo 668581: f_coh old {r['pattern_coherence_v16']:.2f} -> new {r['pattern_coherence']:.2f}")
    out.to_csv(a.out, index=False)
    txt = "\n".join(L); print("\n" + txt)
    Path(str(a.out).replace(".csv", "_summary.txt")).write_text(txt + "\n")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
