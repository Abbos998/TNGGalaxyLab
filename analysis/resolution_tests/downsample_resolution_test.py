#!/usr/bin/env python
"""
downsample_resolution_test.py  --  TNGGalaxyLab, Paper I (Section 5.3)

TNG50-1 yulduz zarralarini TNG100-1 massa rezolyutsiyasigacha tasodifiy
siyraklashtiradi (f = m_b(TNG50)/m_b(TNG100) = 8.5e4/1.4e6 ~ 1/16, massa 1/f ga
ko'paytiriladi) va maqolaning 2-bo'limidagi pipeline'ni (shrinking-sphere ->
L bo'yicha face-on -> |z|<3, R<30 -> 40 bin [0,15] kpc -> [1.5,2.5] R_d) bir xil
qilib ham to'liq, ham siyraklashtirilgan to'plamga qo'llaydi.

Har galaktika uchun yoziladi:
  A1, A2            -- apertura o'rtachasi (bootstrapsiz, catalog A1_literature bilan taqqoslanadi)
  A1_noise          -- fazalar tasodifiy aralashtirilgandagi A1 (sof shot-noise tagligi)
  A1_corr           -- sqrt(A1^2 - A1_noise^2)
  centre_offset_kpc -- siyrak to'plam markazining to'liq to'plam markazidan siljishi
  neff_ap_median    -- aperturadagi binlarda effektiv zarra soni

TNG JupyterLab (asosiy usul), terminal yoki notebook katagida "!" bilan:
  python downsample_resolution_test.py --catalog <TNG50 catalog.csv> --out ds_tng50.csv --limit 20
  # TNG100 ni shu kod bilan siyraklashtirmasdan o'lchash (taqqoslash uchun):
  python downsample_resolution_test.py --basepath /home/tnguser/sims.TNG/TNG100-1/output \
      --box 75000 --f-ds 1.0 --catalog <TNG100 catalog.csv> --out full_tng100.csv

Lokal cutoutlar bilan (PowerShell):
  python scripts\\downsample_resolution_test.py `
      --cutout-dir <HDF5 cutoutlar papkasi> `
      --catalog batch_tng50\\batch_output\\catalog.csv `
      --out batch_tng50\\batch_output\\downsample_test.csv `
      --limit 20            # avval sinov uchun; keyin olib tashlang
Natija CSV'ga qatorma-qator yoziladi; uzilsa, qayta ishga tushirish davom ettiradi.
"""
import argparse, glob, os, re, time
import numpy as np
import pandas as pd
import h5py

H, A_SCALE, BOX = 0.6774, 1.0, 35000.0          # TNG50-1, snapshot 99 (z=0), ckpc/h
F_DS = 8.5e4 / 1.4e6                            # ~0.0607
RC0, RC_MIN, SHRINK, NMIN_SPHERE = 30.0, 0.5, 0.8, 128
R_ORIENT, ZCUT, RCUT, NDISC_MIN = 10.0, 3.0, 30.0, 500
NBINS, ROUT, NBIN_MIN = 40, 15.0, 100
AP_IN, AP_OUT = 1.5, 2.5
AP_KPC = None


def ap_bounds(Rd):
    return AP_KPC if AP_KPC is not None else (AP_IN * Rd, AP_OUT * Rd)


def load_stars(path):
    with h5py.File(path, "r") as f:
        g = f["PartType4"]
        ok = g["GFM_StellarFormationTime"][:] > 0          # wind zarralarini chiqarish
        pos = g["Coordinates"][:][ok].astype(np.float64)
        vel = g["Velocities"][:][ok].astype(np.float64)
        mass = g["Masses"][:][ok].astype(np.float64) * 1e10 / H
        pot = g["Potential"][:][ok] if "Potential" in g else None
    ref = pos[0]
    pos = (pos - ref + BOX / 2) % BOX - BOX / 2 + ref       # davriy chegarani ochish
    return pos * A_SCALE / H, vel * np.sqrt(A_SCALE), mass, pot


def load_stars_il(basepath, sid, snap=99):
    """TNG JupyterLab: snapshotdan to'g'ridan-to'g'ri o'qish (cutout kerak emas)."""
    import illustris_python as il
    flds = ["Coordinates", "Velocities", "Masses", "GFM_StellarFormationTime", "Potential"]
    try:
        g = il.snapshot.loadSubhalo(basepath, snap, int(sid), "stars", fields=flds)
    except Exception:
        g = il.snapshot.loadSubhalo(basepath, snap, int(sid), "stars", fields=flds[:-1])
    ok = g["GFM_StellarFormationTime"] > 0
    pos = g["Coordinates"][ok].astype(np.float64)
    vel = g["Velocities"][ok].astype(np.float64)
    mass = g["Masses"][ok].astype(np.float64) * 1e10 / H
    pot = g["Potential"][ok] if "Potential" in g else None
    ref = pos[0]
    pos = (pos - ref + BOX / 2) % BOX - BOX / 2 + ref
    return pos * A_SCALE / H, vel * np.sqrt(A_SCALE), mass, pot


def shrinking_sphere(pos, mass, pot):
    c = pos[np.argmin(pot)] if pot is not None else np.average(pos, axis=0, weights=mass)
    rc = RC0
    while True:
        sel = np.linalg.norm(pos - c, axis=1) < rc
        if sel.sum() < NMIN_SPHERE:
            break
        c = np.average(pos[sel], axis=0, weights=mass[sel])
        if rc < RC_MIN:
            break
        rc *= SHRINK
    return c


def face_on(pos, vel, mass):
    s = np.linalg.norm(pos, axis=1) < R_ORIENT
    v0 = np.average(vel[s], axis=0, weights=mass[s])
    L = np.sum(mass[s, None] * np.cross(pos[s], vel[s] - v0), axis=0)
    z = L / np.linalg.norm(L)
    t = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(t, z); x /= np.linalg.norm(x)
    return pos @ np.vstack([x, np.cross(z, x), z]).T


def fourier_profile(x, y, m, mmax=2):
    R, phi = np.hypot(x, y), np.arctan2(y, x)
    edges = np.linspace(0.0, ROUT, NBINS + 1)
    idx = np.digitize(R, edges) - 1
    A = np.full((mmax + 1, NBINS), np.nan)
    neff = np.zeros(NBINS)
    for k in range(NBINS):
        s = idx == k
        if s.sum() < NBIN_MIN:
            continue                                        # unresolved bin
        mk, pk = m[s], phi[s]
        c0 = mk.sum()
        neff[k] = c0 ** 2 / np.sum(mk ** 2)
        for mm in range(1, mmax + 1):
            A[mm, k] = 2.0 * np.abs(np.sum(mk * np.exp(-1j * mm * pk))) / c0
    return 0.5 * (edges[1:] + edges[:-1]), A, neff


def aperture_mean(cen, prof, Rd):
    lo, hi = ap_bounds(Rd)
    s = (cen >= lo) & (cen <= hi) & np.isfinite(prof)
    return float(np.mean(prof[s])) if s.any() else np.nan


def measure(pos, vel, mass, pot, Rd, rng, c_ref=None):
    c = shrinking_sphere(pos, mass, pot)
    out = dict(n_star=len(mass), centre=c,
               centre_offset_kpc=float(np.linalg.norm(c - c_ref)) if c_ref is not None else 0.0,
               n_disc=0, A1=np.nan, A2=np.nan, A1_noise=np.nan, A1_corr=np.nan,
               n_ap_bins_resolved=0, neff_ap_median=0.0)
    p = face_on(pos - c, vel, mass)
    disc = (np.abs(p[:, 2]) < ZCUT) & (np.hypot(p[:, 0], p[:, 1]) < RCUT)
    out["n_disc"] = int(disc.sum())
    if out["n_disc"] < NDISC_MIN:
        return out
    x, y, m = p[disc, 0], p[disc, 1], mass[disc]
    cen, A, neff = fourier_profile(x, y, m)
    out["A1"], out["A2"] = aperture_mean(cen, A[1], Rd), aperture_mean(cen, A[2], Rd)
    R, phr = np.hypot(x, y), rng.uniform(-np.pi, np.pi, len(x))   # signal = 0 holati
    _, Ash, _ = fourier_profile(R * np.cos(phr), R * np.sin(phr), m, mmax=1)
    out["A1_noise"] = aperture_mean(cen, Ash[1], Rd)
    if np.isfinite(out["A1"]) and np.isfinite(out["A1_noise"]):
        out["A1_corr"] = float(np.sqrt(max(out["A1"] ** 2 - out["A1_noise"] ** 2, 0.0)))
    lo, hi = ap_bounds(Rd)
    ap = (cen >= lo) & (cen <= hi)
    out["n_ap_bins_resolved"] = int(np.sum(ap & np.isfinite(A[1])))
    good = neff[ap][neff[ap] > 0]
    out["neff_ap_median"] = float(np.median(good)) if good.size else 0.0
    return out


def summarize(df, cat):
    if not (df.f_ds < 1.0).any():
        f = df.set_index("subhalo_id")
        lit = cat.loc[f.index, "A1_literature"]
        print(f"\nmedian A1 = {f.A1.median():.4f}, A1_noise = {f.A1_noise.median():.4f}, "
              f"A1_corr = {f.A1_corr.median():.4f}; median(A1/A1_literature) = {np.nanmedian(f.A1/lit):.3f}")
        return
    full = df[df.f_ds == 1.0].set_index("subhalo_id")
    ds = df[df.f_ds < 1.0].groupby("subhalo_id")[
        ["A1", "A1_noise", "A1_corr", "centre_offset_kpc", "n_disc"]].median()
    lit = cat.loc[full.index, "A1_literature"]
    print("\n=== SANITY: qayta implementatsiya vs catalog (to'liq rezolyutsiya) ===")
    print(f"  median(A1_full / A1_literature) = {np.nanmedian(full.A1 / lit):.3f}   (~1.00 bo'lishi kerak)")
    print(f"  Spearman rho = {full.A1.corr(lit, method='spearman'):.3f}")
    print("\n=== NATIJA ===")
    print(f"  N (to'liq / siyrak, n_disc>=500): {full.A1.notna().sum()} / {ds.A1.notna().sum()}")
    for col in ["A1", "A1_noise", "A1_corr"]:
        print(f"  median {col:9s}: full = {full[col].median():.4f}   downsampled = {ds[col].median():.4f}")
    j = full.index.intersection(ds.dropna(subset=["A1"]).index)
    print(f"  juft median(A1_ds / A1_full) = {np.nanmedian(ds.loc[j, 'A1'] / full.loc[j, 'A1']):.3f}")
    print(f"  median markaz siljishi (ds): {ds.centre_offset_kpc.median():.3f} kpc")
    print("  Taqqoslash uchun: TNG100 catalog'idagi A1_literature medianasini oling (bootstrap 0.225 emas).")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--source", choices=["il", "hdf5"], default="il",
                   help="il = TNG JupyterLab snapshot (illustris_python); hdf5 = lokal cutoutlar")
    a.add_argument("--basepath", default="/home/tnguser/sims.TNG/TNG50-1/output")
    a.add_argument("--cutout-dir", default=None)
    a.add_argument("--box", type=float, default=35000.0, help="ckpc/h; TNG50-1 = 35000, TNG100-1 = 75000")
    a.add_argument("--ap-kpc", type=float, nargs=2, default=None)
    a.add_argument("--f-ds", type=float, default=F_DS, help="1.0 = siyraklashtirishsiz")
    a.add_argument("--catalog", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n-real", type=int, default=3, help="har galaktika uchun siyraklashtirish soni")
    a.add_argument("--ids-file", default=None, help="ixtiyoriy: subhalo ID ro'yxati (masalan D/T>0.5)")
    a.add_argument("--limit", type=int, default=None)
    args = a.parse_args()
    global BOX, AP_KPC
    BOX = args.box
    AP_KPC = tuple(args.ap_kpc) if args.ap_kpc else None
    if AP_KPC:
        print(f"Fizik apertura: {AP_KPC[0]}-{AP_KPC[1]} kpc")

    cat = pd.read_csv(args.catalog).set_index("subhalo_id")
    if args.source == "hdf5":
        files = {int(re.findall(r"\d+", os.path.splitext(os.path.basename(p))[0])[-1]): p
                 for p in glob.glob(os.path.join(args.cutout_dir, "*.hdf5"))}
        ids = [i for i in cat.index if i in files]
        loader = lambda sid: load_stars(files[sid])
    else:
        files = {}
        ids = list(cat.index)
        loader = lambda sid: load_stars_il(args.basepath, sid)
    if args.ids_file:
        keep = {int(l) for l in open(args.ids_file) if l.strip()}
        ids = [i for i in ids if i in keep]
    if args.limit:
        ids = ids[:args.limit]
    print(f"{len(ids)} galaktika (catalog: {len(cat)}, manba: {args.source})")

    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(zip(prev.subhalo_id, prev.seed))
    t0 = time.time()
    for n, sid in enumerate(ids, 1):
        nseeds = args.n_real + 1 if args.f_ds < 1 else 1
        if all((sid, s) in done for s in range(nseeds)):
            continue
        Rd = float(cat.loc[sid, "R_d_kpc"])          # apertura qat'iy: faqat zarra soni effekti
        pos, vel, mass, pot = loader(sid)
        rows = []
        ref = measure(pos, vel, mass, pot, Rd, np.random.default_rng([sid, 0]))
        rows.append(dict(subhalo_id=sid, f_ds=1.0, seed=0, **ref))
        for s in (range(1, args.n_real + 1) if args.f_ds < 1 else []):
            rng = np.random.default_rng([sid, s])
            k = rng.random(len(mass)) < args.f_ds
            r = measure(pos[k], vel[k], mass[k] / args.f_ds, pot[k] if pot is not None else None,
                        Rd, rng, c_ref=ref["centre"])
            rows.append(dict(subhalo_id=sid, f_ds=round(args.f_ds, 4), seed=s, **r))
        df = pd.DataFrame(rows).drop(columns="centre")
        df.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        if n % 10 == 0:
            print(f"  {n}/{len(ids)}  {time.time() - t0:.0f} s")

    if not os.path.exists(args.out):
        print("Natija yo'q: cutout papkasi yoki ID mosligini tekshiring.")
        return
    summarize(pd.read_csv(args.out), cat)


if __name__ == "__main__":
    main()
