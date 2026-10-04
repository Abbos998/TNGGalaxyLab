"""Usage: python dt_subset.py <SIM> <ids.txt> <disc_quality.csv>"""
import importlib.util, shutil, sys
import pandas as pd
spec = importlib.util.spec_from_file_location("cdq", "compute_disc_quality.py")
cdq = importlib.util.module_from_spec(spec); spec.loader.exec_module(cdq)
sim, ids_file, dq_path = sys.argv[1:4]
bp = f"/home/tnguser/sims.TNG/{sim}/output"
ids = [int(l) for l in open(ids_file) if l.strip()]
bak = dq_path.replace(".csv", "_prePBC.csv")
try:
    open(bak).close()
except FileNotFoundError:
    shutil.copy(dq_path, bak)
dq = pd.read_csv(bak)
for sid in ids:
    r = cdq.compute_dt_ratio(bp, 99, sid)
    m = dq.subhalo_id == sid
    old = dq.loc[m, ["DT_ratio", "kappa_rot"]].values.ravel().tolist()
    for k, v in r.items():
        if k in dq.columns and k != "subhalo_id":
            dq.loc[m, k] = v
    new = dq.loc[m, ["DT_ratio", "kappa_rot"]].values.ravel().tolist()
    print(f"{sid}: D/T, kappa  {old} -> {new}", flush=True)
dq.to_csv(dq_path, index=False)
print(f"yozildi: {dq_path} (zaxira: {bak})")
