"""final_analysis.py -- Paper I yakuniy raqamlari (catalog_v2 fayllaridan)"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

P = {"TNG50": "batch_output", "TNG100": "batch_output_tng100"}
STATS = [("A1_bootstrap_mean", "bootstrap mean (eski)"),
         ("A1_literature", "A1_lit (bootstrapsiz)"),
         ("A1_bootstrap_bc", "bootstrap bias-corrected"),
         ("A1_corr", "A1_corr (shovqinsiz)"),
         ("A1_lit_3.5_6kpc", "A1_lit, 3.5-6 kpc"),
         ("A1_corr_3.5_6kpc", "A1_corr, 3.5-6 kpc"),
         ("A1_lit_5_8kpc", "A1_lit, 5-8 kpc"),
         ("A1_corr_5_8kpc", "A1_corr, 5-8 kpc")]


def load(sim, kind):
    c = pd.read_csv(f"{P[sim]}/{'catalog' if kind == 'parent' else 'catalog_filtered'}_v2.csv")
    c["lgM"] = np.log10(c.M_star_Msun)
    return c


cats = {(s, k): load(s, k) for s in P for k in ("parent", "filtered")}

print("=== Sifat nazorati ===")
for (s, k), c in cats.items():
    nw = int(c.box_wrap.fillna(0).sum())
    print(f"{s:6s} {k:8s} N={len(c):5d}  A1_corr bor={c.A1_corr.notna().sum():5d}  box_wrap={nw}"
          f"  noise/A1 median={np.nanmedian(c.A1_noise / c.A1_literature):.2f}"
          f"  Neff_ap median={c.neff_aperture.median():.0f}")
for s in P:
    w = cats[(s, "parent")].query("box_wrap == 1")
    if len(w):
        print(f"  {s} box_wrap: " + ", ".join(
            f"{int(r.subhalo_id)} (A1={r.A1_literature:.2f})" for _, r in w.iterrows()))
o = cats[("TNG50", "parent")].query("subhalo_id == 668581")
if len(o):
    r = o.iloc[0]
    print(f"  668581: A1_lit={r.A1_literature:.3f} A1_corr={r.A1_corr:.3f} box_wrap={r.box_wrap}")

rows = []
print("\n=== TNG50 vs TNG100 (box_wrap chiqarilgan) ===")
for kind in ("parent", "filtered"):
    a = cats[("TNG50", kind)].query("box_wrap != 1")
    b = cats[("TNG100", kind)].query("box_wrap != 1")
    print(f"\n-- {kind} (N50={len(a)}, N100={len(b)}) --")
    print(f"{'statistika':28s} {'TNG50':>8s} {'TNG100':>8s} {'50/100':>7s} {'KS D':>6s} {'KS p':>9s}")
    for col, lab in STATS:
        x, y = a[col].dropna(), b[col].dropna()
        D, p = ks_2samp(x, y)
        m50, m100 = x.median(), y.median()
        print(f"{lab:28s} {m50:8.4f} {m100:8.4f} {m50 / m100:7.3f} {D:6.3f} {p:9.1e}")
        rows.append(dict(sample=kind, statistic=col, N50=len(x), N100=len(y), med50=m50,
                         med100=m100, ratio50_100=m50 / m100, KS_D=D, KS_p=p))
pd.DataFrame(rows).to_csv("results_summary.csv", index=False)

print("\n=== Massa binlari (filtered, box_wrap chiqarilgan): T100/T50 ===")
brow = []
a = cats[("TNG50", "filtered")].query("box_wrap != 1")
b = cats[("TNG100", "filtered")].query("box_wrap != 1")
for lo, hi in zip([10.0, 10.3, 10.6], [10.3, 10.6, 11.0]):
    sa, sb = a[(a.lgM >= lo) & (a.lgM < hi)], b[(b.lgM >= lo) & (b.lgM < hi)]
    line = f"[{lo},{hi}) N50={len(sa):3d} N100={len(sb):3d} |"
    for col in ("A1_literature", "A1_corr", "A1_corr_3.5_6kpc", "A1_corr_5_8kpc"):
        r = sb[col].median() / sa[col].median()
        line += f" {col.replace('A1_', '')}={r:.2f}"
        brow.append(dict(lgM_lo=lo, lgM_hi=hi, statistic=col, N50=len(sa), N100=len(sb),
                         med50=sa[col].median(), med100=sb[col].median(), ratio100_50=r))
    print(line)
pd.DataFrame(brow).to_csv("results_massbins.csv", index=False)
print("\nYozildi: results_summary.csv, results_massbins.csv")
