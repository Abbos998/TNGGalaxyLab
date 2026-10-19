"""Run recompute_fcoh_fixed.py reading stars straight from the TNG snapshot."""
import sys, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import illustris_python as il

SIM = "TNG50-1"
BASE = f"/home/tnguser/sims.TNG/{SIM}/output"
SNAP = 99
BOX = 35000.0
HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("rf", HERE / "recompute_fcoh_fixed.py")
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)

hdr = il.groupcat.loadHeader(BASE, SNAP)
h, a, z = float(hdr["HubbleParam"]), float(hdr["Time"]), float(hdr["Redshift"])


def load_from_snapshot(path):
    sid = int(Path(path).stem.split("_")[-1])
    s = il.snapshot.loadSubhalo(BASE, SNAP, sid, "stars",
            fields=["Coordinates", "Masses", "Velocities", "Potential",
                    "GFM_StellarFormationTime"])
    keep = s["GFM_StellarFormationTime"] > 0
    coords = s["Coordinates"][keep].astype(np.float64)
    mass = s["Masses"][keep].astype(np.float64)
    vel = s["Velocities"][keep].astype(np.float64)
    pot = s["Potential"][keep].astype(np.float64)
    ref = coords[np.argmin(pot)]
    coords = (coords - ref + 0.5 * BOX) % BOX - 0.5 * BOX + ref
    return dict(pos=coords * a / h, mass=mass * 1e10 / h, vel=vel * np.sqrt(a),
                potential=pot, h=h, a=a, redshift=z, snapshot=SNAP, simulation=SIM)


rf.load_tng_stars = load_from_snapshot

cat_path = sys.argv[sys.argv.index("--catalog") + 1]
ph = HERE / "_snapshot_placeholders"
ph.mkdir(exist_ok=True)
for sid in pd.read_csv(cat_path)["subhalo_id"].astype(int):
    (ph / f"cutout_{sid}.hdf5").touch()
sys.argv += ["--cutouts", str(ph)]
rf.main()
