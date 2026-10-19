# v17 (2026-10-19)

Phase-period fix in the $m=2$ pattern diagnostic.

## Fix

- `tnggalaxylab/fourier/diagnostics.py` and `.../bootstrap.py`:
  the amplitude-weighted mean $m=2$ phase, the bar-length phase
  coherence test, the coherent-fraction flag $f_{\rm coh}$ and the
  $m=2$ phase scatter all wrapped phase differences with period
  $2\pi$ instead of $\pi$ (the natural period of $\phi_2 =
  \arg(c_2)/2$).  Discs whose bar aligns near $\pm 90^\circ$ were
  systematically mis-flagged.  The fix uses
  $\bar\phi_2 = \tfrac{1}{2}\arg\sum w\, e^{2\mathrm i\phi_2}$ and
  wraps phase differences into $(-\pi/2, \pi/2]$.

## Catalogues

All catalogue columns affected by the bug were recomputed with the
corrected diagnostic and overwrite the released files in place.
The columns are `bar_length_kpc`, `bar_angle_deg`,
`pattern_coherence` (= $f_{\rm coh}$) and `phase_scatter_m2_deg`.
All other columns (amplitudes, bootstrap means, noise floors,
noise-corrected amplitudes, kinematic quantities) are unchanged.

Headline change: the TNG50-1 parent fraction of coherent $m=2$
detections ($f_{\rm coh} > 0.5$) rises from 47.1 to 66.7 per cent.
The TNG100/TNG50 excess in $A_1$, the shot-noise correction and the
mass trend are not affected by the bug and are unchanged.

## New scripts

- `batch_tng50/recompute_fcoh_fixed.py` — reproduces the catalogue
  diagnostic columns from HDF5 cutouts with the corrected algorithm.
- `batch_tng50/recompute_fcoh_from_snapshot.py` — same, but reads
  particles directly from a TNG snapshot through `illustris_python`
  (used to regenerate the TNG50-1 catalogue when the cutouts were
  not retained).
- `batch_tng50/fig7_resolution_notitle.py` — Fig. 7 regeneration
  with the MNRAS-style (a)/(b)/(c) panel labels.

## Figure scripts

- `batch_tng50/fig9_v2_rotation_analysis.py` and
  `batch_tng50/fig6_v3_four_way_CDF.py`: panel titles replaced by
  (a)/(b) labels inside the frames; KS insets in Fig. 6 removed
  (moved to the caption).  Łokas is spelt with the barred L.
- `scripts/validate_synthetic.py` (Fig. D1): the `\%`-mangled Note
  annotation was removed.
- `scripts/parameter_sensitivity.py` (Fig. D3): the y-axis label
  was changed from `[\%]` to `[%]`.
