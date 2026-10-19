"""Fig 7: TNG100/TNG50 farqining dekompozitsiyasi (sarlavhasiz versiya)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9, "legend.fontsize": 7.5,
                     "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True})
R, B = "analysis/resolution_tests/", "batch_tng50/"
m50 = pd.read_csv(B + "batch_output/catalog_v2.csv").set_index("subhalo_id").M_star_Msun
m100 = pd.read_csv(B + "batch_output_tng100/catalog_v2.csv").set_index("subhalo_id").M_star_Msun
a = pd.read_csv(R + "ds_tng50_filtered_all.csv")
b = pd.read_csv(R + "full_tng100_filtered.csv").set_index("subhalo_id")
f = a[a.f_ds == 1].set_index("subhalo_id")
d = a[a.f_ds < 1].groupby("subhalo_id")[["A1", "A1_corr", "neff_ap_median"]].median()
for df, m in [(f, m50), (d, m50), (b, m100)]:
    df["lgM"] = np.log10(m.loc[df.index].values)

def boot_median(x, n=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([np.median(rng.choice(x, len(x))) for _ in range(n)])

def panel(ax, s):
    ax.text(0.04, 0.96, f"({s})", transform=ax.transAxes, fontsize=10, ha="left", va="top")

edges = np.array([9.5, 10.0, 10.3, 10.6, 11.0]); xc = 0.5 * (edges[1:] + edges[:-1])
sty = {"f": dict(c="#1f4e9c", marker="o", label="TNG50-1"),
       "d": dict(c="#6aa6e0", marker="s", ls="--", label="TNG50-1, downsampled"),
       "b": dict(c="#c0392b", marker="D", label="TNG100-1")}
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.55), constrained_layout=True)
for k, col in enumerate(["A1", "A1_corr"]):
    for key, df in [("f", f), ("d", d), ("b", b)]:
        y, lo, hi = [], [], []
        for l, h in zip(edges[:-1], edges[1:]):
            s = df[(df.lgM >= l) & (df.lgM < h)][col].dropna().values
            if len(s) < 5:
                y.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            bm = boot_median(s); y.append(np.median(s))
            lo.append(np.percentile(bm, 16)); hi.append(np.percentile(bm, 84))
        y, lo, hi = map(np.array, (y, lo, hi)); ok = ~np.isnan(y)
        ax[k].errorbar(xc[ok], y[ok], yerr=[y[ok] - lo[ok], hi[ok] - y[ok]],
                       ms=4, capsize=2, lw=1.1, **sty[key])
    ax[k].set_ylim(0, 0.21)
ax[0].set_ylabel(r"median $A_1$")
ax[1].set_ylabel(r"median $A_{1,\rm corr}$")
ax[0].legend(loc="lower left", frameon=False, fontsize=7)

v50 = pd.read_csv(B + "batch_output/catalog_filtered_v2.csv")
v100 = pd.read_csv(B + "batch_output_tng100/catalog_filtered_v2.csv")
for v in (v50, v100):
    v["lgM"] = np.log10(v.M_star_Msun)
e2 = np.array([10.0, 10.3, 10.6, 11.0]); x2 = 0.5 * (e2[1:] + e2[:-1])
for j, (col, lab, c, mk) in enumerate([("A1_corr", r"$[1.5,2.5]\,R_{\rm d}$", "k", "o"),
                                       ("A1_corr_3.5_6kpc", "3.5–6 kpc", "#8e44ad", "^"),
                                       ("A1_corr_5_8kpc", "5–8 kpc", "#27ae60", "v")]):
    r, lo, hi = [], [], []
    for l, h in zip(e2[:-1], e2[1:]):
        s5 = v50[(v50.lgM >= l) & (v50.lgM < h)][col].values
        s1 = v100[(v100.lgM >= l) & (v100.lgM < h)][col].values
        bs = boot_median(s1, seed=1) / boot_median(s5, seed=2)
        r.append(np.median(s1) / np.median(s5))
        lo.append(np.percentile(bs, 16)); hi.append(np.percentile(bs, 84))
    r, lo, hi = map(np.array, (r, lo, hi))
    ax[2].errorbar(x2 + (j - 1) * 0.025, r, yerr=[r - lo, hi - r], c=c, marker=mk,
                   ms=4, capsize=2, lw=1.1, label=lab)
    print(col, np.round(r, 2))
ax[2].axhline(1, c="grey", lw=0.8, ls=":")
ax[2].set_ylim(0.8, 2.8); ax[2].set_xlim(9.9, 11.0)
ax[2].set_ylabel(r"TNG100 / TNG50, median $A_{1,\rm corr}$")
ax[2].legend(loc="upper right", frameon=False)
for x in ax:
    x.set_xlabel(r"$\log_{10}(M_\star/{\rm M}_\odot)$")
ax[0].set_xlim(9.5, 11.0); ax[1].set_xlim(9.5, 11.0)
for k in range(3):
    panel(ax[k], "abc"[k])
fig.savefig("figures_new/fig13_resolution_decomposition.pdf")
fig.savefig("figures_new/fig13_resolution_decomposition.png", dpi=200)
print("saqlandi: figures_new/fig13_resolution_decomposition.pdf")
