import glob
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
B50, B100 = 'batch_tng50/batch_output', 'batch_tng50/batch_output_tng100'

def side(path):
    t = pd.read_csv(path).drop_duplicates('subhalo_id').set_index('subhalo_id')
    return t
dq50, dq100 = side(f'{B50}/disc_quality.csv'), side(f'{B100}/disc_quality.csv')
m50, m100 = side(f'{B50}/catalog.csv').M_star_Msun, side(f'{B100}/catalog.csv').M_star_Msun

def prep(df, m, dq):
    df = df.copy()
    df['lgM'] = np.log10(m.reindex(df.index).values)
    df['kappa'] = dq['kappa_rot'].reindex(df.index).values
    df['DT'] = dq['DT_ratio'].reindex(df.index).values
    return df[(df.lgM >= 10.0) & (df.lgM < 11.0)].dropna(subset=['kappa', 'DT', 'lgM'])

def fit(a, b, col, terms, nboot=300):
    d = pd.concat([a.assign(s=0), b.assign(s=1)])
    d = d[d[col] > 0]
    med = {t: d[t].median() for t in terms}
    def one(dd):
        X = np.column_stack([np.ones(len(dd)), dd.s, dd.lgM - 10.3] + [dd[t] - med[t] for t in terms])
        return np.linalg.lstsq(X, np.log10(dd[col].values), rcond=None)[0]
    beta = one(d)
    bs = np.array([one(pd.concat([g.sample(len(g), replace=True, random_state=int(r))
                                  for _, g in d.groupby('s')]))
                   for r in rng.integers(0, 2**31 - 1, nboot)])
    lo, hi = np.percentile(10 ** bs[:, 1], [16, 84])
    return 10 ** beta[1], lo, hi, beta

sets = {
    'R_d [1.5,2.5]': (['ds_tng50_filtered_all.csv'], 'full_tng100_filtered.csv'),
    '3.5-6.0 kpc':   (sorted(glob.glob('ap3.5-6.0_t50_*.csv')), 'ap3.5-6.0_t100.csv'),
    '5.0-8.0 kpc':   (sorted(glob.glob('ap5.0-8.0_t50_*.csv')), 'ap5.0-8.0_t100.csv'),
}

first = True
for name, (f50, f100) in sets.items():
    a = pd.concat([pd.read_csv(f) for f in f50])
    full = prep(a[a.f_ds == 1].set_index('subhalo_id'), m50, dq50)
    ds = prep(a[a.f_ds < 1].groupby('subhalo_id')[['A1', 'A1_corr']].median(), m50, dq50)
    t100 = prep(pd.read_csv(f100).set_index('subhalo_id'), m100, dq100)

    if first:
        first = False
        print('=== Disk kinematikasi (lgM 10-11, filtrlangan namunalar) ===')
        for lo, hi in zip([10.0, 10.3, 10.6], [10.3, 10.6, 11.0]):
            for nm, df in [('TNG50', full), ('TNG100', t100)]:
                s = df[(df.lgM >= lo) & (df.lgM < hi)]
                print(f'[{lo},{hi}) {nm:6s} N={len(s):4d}  kappa_rot={s.kappa.median():.3f}  D/T={s.DT.median():.3f}')
        print()

    print(f'=== Apertura: {name} ===')
    for label, A, Bn, col in [('A1corr: T100 vs T50full', full, t100, 'A1_corr'),
                              ('A1 raw: T100 vs T50ds  ', ds, t100, 'A1')]:
        for terms in [[], ['kappa'], ['kappa', 'DT']]:
            r, lo, hi, beta = fit(A, Bn, col, terms)
            extra = ''.join(f'  d(logA1)/d{t}={b:+.2f}' for t, b in zip(terms, beta[3:]))
            print(f'  {label}  | massa{"+" + "+".join(terms) if terms else "":12s}: '
                  f'T100/T50 = {r:.2f} [{lo:.2f}, {hi:.2f}]{extra}')
    print()
