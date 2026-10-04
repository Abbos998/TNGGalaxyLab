p = "compute_disc_quality.py"
s = open(p).read()
old = "    pos_phys -= pos_phys[idx_center]\n"
new = old + '''    # Periodic-boundary unwrap (physical kpc, snapshot 99, a = 1)
    _L = next((b for k, b in {"TNG50-1": 35000.0, "TNG100-1": 75000.0,
                               "TNG300-1": 205000.0}.items() if k in basepath), None)
    if _L is not None:
        _L = _L / 0.6774
        pos_phys = (pos_phys + 0.5 * _L) % _L - 0.5 * _L
'''
if "Periodic-boundary unwrap" in s:
    print("compute_disc_quality allaqachon tuzatilgan")
elif old not in s:
    print("TOPILMADI: 'pos_phys -= pos_phys[idx_center]' qatorini qo'lda tekshiring")
else:
    open(p, "w").write(s.replace(old, new, 1)); print("tuzatildi: compute_disc_quality.py")
