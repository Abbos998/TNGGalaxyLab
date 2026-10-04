"""Overwrite catalogue rows of the PBC-fixed galaxies with the re-processed JSON values.
Usage: python patch_catalog.py <batch_output_dir> <ids.txt>"""
import json, sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd

d, ids = Path(sys.argv[1]), [int(l) for l in open(sys.argv[2]) if l.strip()]
for name in ("catalog_v2.csv", "catalog_filtered_v2.csv"):
    src = d / name
    bak = d / name.replace(".csv", "_prePBC.csv")
    if not bak.exists():
        shutil.copy(src, bak)
    cat = pd.read_csv(bak)
    if "pbc_fixed" not in cat:
        cat["pbc_fixed"] = 0
    n = 0
    for sid in ids:
        jp = d / "processed" / f"{sid}.json"
        if not jp.exists() or sid not in set(cat.subhalo_id):
            continue
        res = json.loads(jp.read_text())
        if "box_wrap" not in res:
            print(f"  {sid}: JSON da shovqin ustunlari yo'q (05 ishlamagan)"); continue
        i = cat.index[cat.subhalo_id == sid][0]
        old = cat.at[i, "A1_literature"]
        for c in cat.columns:
            if c in res and not isinstance(res[c], (list, dict)):
                cat.at[i, c] = np.nan if res[c] is None else res[c]
        cat.at[i, "pbc_fixed"] = 1
        n += 1
        if name == "catalog_v2.csv":
            print(f"  {sid}: A1_lit {old:.4f} -> {cat.at[i,'A1_literature']:.4f}   "
                  f"A1_corr -> {cat.at[i,'A1_corr']:.4f}")
    cat.to_csv(src, index=False)
    print(f"{src}: {n} qator yangilandi (zaxira: {bak.name})")
