"""Patch every copy of load_tng_stars (and 05's load_il) with a periodic-boundary unwrap."""
import glob
BOXDEF = 'BOX_CKPC_H = {"TNG50-1": 35000.0, "TNG100-1": 75000.0, "TNG300-1": 205000.0}\n\n\n'
UNWRAP = '''    potential  = potential[stellar_mask]

    # Periodic-boundary unwrap around the potential minimum (ckpc/h).
    # Without it, galaxies near the box edge lose particles to the
    # opposite side of the box and acquire a spurious m=1 signal.
    _box = BOX_CKPC_H.get(str(sim))
    if _box is not None and len(coords) > 0:
        _ref = coords[np.argmin(potential)]
        coords = (coords - _ref + 0.5 * _box) % _box - 0.5 * _box + _ref
'''
for p in glob.glob("../**/*.py", recursive=True):
    s = open(p).read()
    if "def load_tng_stars" not in s or "_box = BOX_CKPC_H" in s:
        continue
    if "    potential  = potential[stellar_mask]\n" not in s:
        print("TOPILMADI (qo'lda tekshiring):", p); continue
    s = s.replace("    potential  = potential[stellar_mask]\n", UNWRAP, 1)
    if "BOX_CKPC_H =" not in s:
        s = s.replace("def load_tng_stars", BOXDEF + "def load_tng_stars", 1)
    open(p, "w").write(s); print("tuzatildi:", p)

p = "05_add_noise_columns.py"
s = open(p).read()
if "_ref = _c[m]" not in s:
    s = s.replace('''    m = s["GFM_StellarFormationTime"] > 0
''', '''    m = s["GFM_StellarFormationTime"] > 0
    if box is not None and m.any():
        _c = s["Coordinates"].astype(np.float64)
        _ref = _c[m][np.argmin(s["Potential"][m])]
        s["Coordinates"] = (_c - _ref + 0.5 * box) % box - 0.5 * box + _ref
''', 1)
    s = s.replace("def load_il(basepath, sid, snap=99, h=0.6774):",
                  "def load_il(basepath, sid, snap=99, h=0.6774, box=None):")
    s = s.replace("load_il(basepath, sid) if", "load_il(basepath, sid, box=BOX_CKPC_H.get(sim)) if")
    open(p, "w").write(s); print("tuzatildi: 05_add_noise_columns.py (load_il)")
else:
    print("05 allaqachon tuzatilgan")
