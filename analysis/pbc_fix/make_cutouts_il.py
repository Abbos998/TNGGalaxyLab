"""Write TNG50 cutouts (format expected by load_tng_stars) for a list of subhalo IDs."""
import sys
from pathlib import Path
import h5py
import illustris_python as il

SIM, SNAP = "TNG50-1", 99
bp = f"/home/tnguser/sims.TNG/{SIM}/output"
out = Path("batch_output/cutouts"); out.mkdir(parents=True, exist_ok=True)
z = float(il.groupcat.loadHeader(bp, SNAP)["Redshift"])
F = ["Coordinates", "Velocities", "Masses", "GFM_StellarFormationTime", "Potential"]
for sid in [int(l) for l in open(sys.argv[1]) if l.strip()]:
    s = il.snapshot.loadSubhalo(bp, SNAP, sid, "stars", fields=F)
    with h5py.File(out / f"cutout_{sid}.hdf5", "w") as f:
        g = f.create_group("PartType4")
        for k in F:
            g.create_dataset(k, data=s[k])
        H = f.create_group("Header")
        H.attrs.update(Redshift=z, Time=1.0 / (1.0 + z), HubbleParam=0.6774,
                       SnapshotNumber=SNAP, SimulationName=SIM, SubhaloID=sid,
                       NumStars=len(s["Coordinates"]))
    print("cutout:", sid, len(s["Coordinates"]), flush=True)
