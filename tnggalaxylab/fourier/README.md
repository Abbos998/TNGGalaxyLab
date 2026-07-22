# `tnggalaxylab.fourier` — scientific Fourier analysis of galaxy disks

Publication-quality Fourier decomposition of projected surface density
for N-body and cosmological simulation snapshots (IllustrisTNG, EAGLE,
GADGET, RAMSES). Drop-in upgrade of the legacy Fourier module with
backward-compatible API.

---

## Quick start

```python
from tnggalaxylab.fourier import (
    compute_fourier, global_lopsidedness,
    compute_pattern_diagnostics, bootstrap_fourier,
    plot_amplitude_profiles, export_csv, export_json,
)

# 1. Radial Fourier profile (particle-direct method — recommended)
profile = compute_fourier(
    x, y, mass,
    method="particles",
    r_in=0.0, r_out=20.0, n_bins=40, m_max=4,
)

# 2. R&Z95 global lopsidedness over [1.5, 2.5] R_d
gm = global_lopsidedness(profile, scale_length=3.0)
print(f"A_1 = {gm.A1:.3f}  (literature average)")
print(f"A_1 = {gm.A1_integral:.3f}  (area-weighted integral)")

# 3. Bar and pattern diagnostics
diag = compute_pattern_diagnostics(profile, bar_threshold=0.2)
print(f"Bar length  = {diag.bar_length:.2f} kpc")
print(f"Bar angle   = {np.degrees(diag.bar_angle):.1f}°")
print(f"Coherence   = {diag.pattern_coherence:.2f}")

# 4. Bootstrap uncertainties
boot = bootstrap_fourier(
    x, y, mass, method="particles",
    n_bootstrap=200, seed=0,
    r_in=0.0, r_out=20.0, n_bins=40, m_max=4,
    scale_length=3.0,
)
print(f"A_1 = {boot['global_A1_mean']:.3f} ± {boot['global_A1_std']:.3f}")

# 5. Publication plots + CSV/JSON
plot_amplitude_profiles(boot["profile"],
                        r_range=gm.r_range_kpc,
                        global_modes=gm,
                        savepath="A_m.png")
export_csv(boot["profile"], global_modes=gm, outpath="fourier.csv")
export_json(boot["profile"], global_modes=gm, pattern_diag=diag,
            outpath="fourier.json")
```

---

## Mathematical formulation

### Surface density expansion

The Fourier expansion of the projected surface density at radius R is

```
Σ(R, φ) = a₀(R)/2 + Σ_{m=1}^{M} [ a_m(R) cos(mφ) + b_m(R) sin(mφ) ]
```

The amplitude and phase of mode m, in the convention of Rix & Zaritsky (1995):

```
A_m(R)  = √(a_m² + b_m²) / (a₀/2)                          [Eq. 1]
Φ_m(R)  = (1/m) · arctan2(b_m, a_m)                        [Eq. 2]
```

### FFT method

Project particles onto a 2-D Cartesian grid, optionally Gaussian-smooth,
then sample on a polar lattice. The azimuthal `rfft` returns complex
coefficients c_m. In the NumPy convention c_m = (a_m − i b_m)/2, so:

```
a_m = 2 Re(c_m)
b_m = −2 Im(c_m)
A_m = √(a_m² + b_m²) / (a₀/2)
```

### Particle-direct method (recommended for validation)

No gridding, no smoothing, no interpolation. For each annulus the
mass-weighted complex Fourier sums are:

```
C_m(R) = Σ_{j ∈ annulus} m_j cos(m θ_j)
S_m(R) = Σ_{j ∈ annulus} m_j sin(m θ_j)
M₀(R)  = Σ_{j ∈ annulus} m_j

A_m(R) = 2 · √(C_m² + S_m²) / M₀                           [Eq. 3]
Φ_m(R) = (1/m) · arctan2(S_m, C_m)                         [Eq. 4]
```

#### Derivation of the factor of 2

For a continuous surface density Σ(R, φ), the mass-weighted moment is

```
<cos(mφ)>_Σ = ∫ Σ cos(mφ) dφ / ∫ Σ dφ
             = (a_m π) / (a₀ π)
             = a_m / a₀
             = (1/2) · a_m / (a₀/2)
             = A_m / 2
```

Therefore the discrete particle estimator C_m/M₀ equals A_m/2 in the
R&Z95 normalisation. The factor of 2 in Eq. 3 reconciles the particle
moment estimator with the standard literature amplitude. Without it,
the particle method silently gives the Saha 2007 I_m convention, which
is half the value quoted in observational lopsidedness catalogues.

### Global lopsidedness

Two equivalent disk-integrated estimators are computed.

**Literature average (R&Z95, BCJ05, default):**

```
⟨A_m⟩ = (1/N_aperture) · Σ_{R_i ∈ [1.5, 2.5] R_d}  A_m(R_i)         [Eq. 5]
```

**Area-weighted integral (Jog 2002):**

```
⟨A_m⟩_int = ∫_{R_in}^{R_out} A_m(R) · R dR  /  ∫_{R_in}^{R_out} R dR  [Eq. 6]
```

The default aperture follows R&Z95 (1.5–2.5 R_d). It can be overridden:

```python
gm = global_lopsidedness(profile, scale_length=3.0)            # auto
gm = global_lopsidedness(profile, r_range=(4.5, 7.5))           # explicit
gm = global_lopsidedness(profile, scale_length=3.0,
                          method="global_integral")             # Eq. 6
```

### Pattern diagnostics

The bar position angle is the **amplitude-weighted circular mean** of
Φ₂(R) over the disk:

```
ψ_bar = arg( Σ A_2(R_i) exp(i Φ_2(R_i)) / Σ A_2(R_i) )
```

The bar length is the outermost radius where A_2 exceeds a threshold
(default 0.2, after Aguerri, Elias-Rosa & Corsini 2009):

```
L_bar = max { R_i : A_2(R_i) > A_threshold }
```

The pattern coherence is the fraction of radial bins whose Φ_2 falls
within ±π/8 of ψ_bar (after Chequers, Widrow & Darling 2016):

```
C = (1/N) · # { i : |Φ_2(R_i) − ψ_bar| < π/8 }
```

Phase scatter for each mode is the circular standard deviation:

```
R̄_m = | mean( exp(i Φ_m(R_i)) ) |
σ_m  = √(−2 ln R̄_m)
```

### Bootstrap uncertainties

For each iteration b = 1 … N_boot, particles are resampled with
replacement and the entire pipeline is re-run. The bootstrap estimate
of σ(A_m(R)) is the per-bin standard deviation across iterations.
Phase uncertainties use circular statistics. The same procedure
propagates to global A_1, A_2, bar length, and bar angle.

---

## API reference

### `compute_fourier(x, y, mass, *, method="fft", r_in, r_out, n_bins, m_max, ...)` → `FourierProfile`
Main entry. Dispatches to `fft_fourier` or `particle_fourier`.

### `fft_fourier(x, y, mass, ...)` → `FourierProfile`
Surface-density-grid FFT pipeline. Kwargs: `n_pix`, `sigma_smooth`,
`theta_bins`.

### `particle_fourier(x, y, mass, ...)` → `FourierProfile`
Particle-direct mass-weighted Fourier sums (Saha 2007 with R&Z95
normalisation).

### `global_lopsidedness(profile, scale_length=None, r_range=None, method="literature_average")` → `GlobalModes`
Disk-integrated A_1/A_2 over [1.5, 2.5] R_d (or explicit `r_range`).
Both literature-average and area-integral are always stored.

### `compute_pattern_diagnostics(profile, bar_threshold=0.2, coherence_tol=π/8, r_min, r_max)` → `PatternDiagnostics`
Dominant mode, pattern angle, bar length and angle, phase coherence
and scatter.

### `bootstrap_fourier(x, y, mass, *, method, n_bootstrap, seed, ...)` → `dict`
Bootstrap resampling. Returns means, std devs, and a
`FourierProfile` with `.amp_err` / `.phase_err` populated.

### Synthetic generators
```
make_exponential_disk(n_particles, R_d, r_max, seed)
make_lopsided_disk    (n_particles, R_d, epsilon_1, phi_1, r_max, seed)
make_barred_disk      (n_particles, R_d, epsilon_2, phi_2, r_max, seed)
make_logarithmic_spiral(n_particles, R_d, epsilon_s, pitch_angle_deg, r_max, seed)
```
Each returns a `SyntheticGalaxy` whose `.analytic_amps[m](R)` gives the
exact analytic A_m(R) used for validation.

### Plot and export
```
plot_amplitude_profiles(profile, modes, r_range, global_modes, savepath)
plot_phase_profiles    (profile, modes, savepath)
plot_method_comparison (profile_fft, profile_par, m, savepath)
plot_validation        (profile, analytic_amps, m, savepath)
export_csv             (profile, global_modes, outpath)
export_json            (profile, global_modes, pattern_diag, outpath)
```

---

## Validation results

All four synthetic galaxies recover their analytic amplitudes:

| Galaxy | Analytic | Recovered (N = 100k, particle method) |
|---|---|---|
| Exponential | A_m = 0 for all m | A_m < 0.02 everywhere |
| Lopsided ε₁ = 0.3 | A_1 = 0.30 | A_1 = 0.300 ± 0.005 |
| Barred ε₂ = 0.4 | A_2 = 0.40 | A_2 = 0.401 ± 0.008 |
| Spiral ε_s = 0.3 | A_2 = 0.30, Φ_2 ∝ ln R | Confirmed |

---

## Backward compatibility

All pre-existing APIs are preserved:

- `fft_fourier(x, y, mass, ...)` still works as before
- `particle_fourier(x, y, mass, ...)` available (new)
- `GlobalModes.A1` defaults to the literature average; the old call
  pattern `gm.A1` returns the same numerical value as before for code
  that already used the 1.5–2.5 R_d aperture
- `FourierProfile.amp_err` and `.phase_err` are new optional fields,
  default `None` so old downstream code is unaffected

---

## References

Rix, H.-W. & Zaritsky, D. 1995, ApJ, 447, 82
Zaritsky, D. & Rix, H.-W. 1997, ApJ, 477, 118
Jog, C. J. 2002, A&A, 391, 471
Bournaud, F., Combes, F., Jog, C. J. 2005, A&A, 437, 69
Saha, K., Combes, F., Jog, C. J. 2007, MNRAS, 382, 419
Aguerri, J. A. L., Elias-Rosa, N., Corsini, E. M. 2009, A&A, 494, 891
Chequers, M. H., Widrow, L. M., Darling, K. 2016, MNRAS, 463, 1631
Efron, B. 1979, Ann. Stat., 7, 1
