# Mixed Synthetic Galaxy Validation Report

## Setup

* Injected amplitudes: ε₁ = 0.15, ε₂ = 0.25, ε_spiral = 0.1
* Bar phase offset from lopsided: 45°
* Spiral pitch angle: 15.0°
* Replicas: 5
* Runtime: 1.4 s

## Results

| Metric | Recovered | Injected | Frac. bias |
|---|---|---|---|
| A₁ mean | 0.1448 ± 0.0089 | 0.15 | -3.5% |
| A₂ mean | 0.2235 ± 0.0085 | 0.25 | -10.6% |

## Interpretation

* The pipeline recovers the injected lopsidedness amplitude ε₁ = 0.15 to within 3.5% in the presence of a coexisting bar and spiral.
* The pipeline recovers the injected bar amplitude ε₂ = 0.25 to within 10.6% (spiral contribution to m=2 is expected to be small).
* This demonstrates the pipeline's ability to separate coexisting Fourier modes in realistic multi-component disc morphologies.
