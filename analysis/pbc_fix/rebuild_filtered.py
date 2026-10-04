"""Rebuild catalog_filtered_v2.csv from the PBC-patched catalog_v2 + disc_quality."""
import pandas as pd

def sel(cat, dq, col):
    m = cat.merge(dq[["subhalo_id", "DT_ratio"]], on="subhalo_id", how="left")
    return set(m[(m.DT_ratio > 0.5) & (m.n_disk_particles > 1e4) & (m[col] < 0.5)].subhalo_id)

for d in ["batch_output", "batch_output_tng100"]:
    cat0 = pd.read_csv(f"{d}/catalog_v2_prePBC.csv")
    dq0 = pd.read_csv(f"{d}/disc_quality_prePBC.csv").drop_duplicates("subhalo_id")
    old = set(pd.read_csv(f"{d}/catalog_filtered_v2_prePBC.csv").subhalo_id)
    col = None
    for c in ["A1_bootstrap_mean", "A1_literature"]:
        s = sel(cat0, dq0, c)
        ok = s == old
        print(f"{d}: mezon A1={c:18s} -> {len(s)} (eski {len(old)})  {'ANIQ MOS' if ok else 'farq ' + str(len(s ^ old))}")
        if ok and col is None:
            col = c
    if col is None:
        print(f"  {d}: hech bir mezon eski namunani aniq bermadi -- filtrlangan fayl o'zgartirilmadi")
        continue
    cat = pd.read_csv(f"{d}/catalog_v2.csv")
    dq = pd.read_csv(f"{d}/disc_quality.csv").drop_duplicates("subhalo_id")
    new = sel(cat, dq, col)
    print(f"  yangi: {len(new)}  qo'shildi: {sorted(new - old)}  chiqdi: {sorted(old - new)}")
    cat[cat.subhalo_id.isin(new)].to_csv(f"{d}/catalog_filtered_v2.csv", index=False)
    print(f"  yozildi: {d}/catalog_filtered_v2.csv")
