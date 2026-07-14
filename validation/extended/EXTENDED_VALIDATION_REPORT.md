# Extended Validation Report — TNGGalaxyLab

**Reference disc**: lopsided, ε₁ = 0.3, R_d = 3.0 kpc
**N particles**: 100,000
**Replicas per level**: 5
**Total wall time**: 7.4 s (0.1 min)

## Test 1 — Inclination

| Level | A_1 mean | A_1 std | |ΔA_1/A_1| |
|---|---|---|---|
| 0.0 degrees | 0.3031 | 0.0072 | 1.04% |
| 15.0 degrees | 0.3032 | 0.0067 | 1.08% |
| 30.0 degrees | 0.3079 | 0.0082 | 2.64% |
| 45.0 degrees | 0.3299 | 0.0058 | 9.98% |
| 60.0 degrees | 0.3761 | 0.0091 | 25.37% |

## Test 2 — Centring offset

| Level | A_1 mean | A_1 std | |ΔA_1/A_1| |
|---|---|---|---|
| 0.0 kpc | 0.2954 | 0.0088 | 1.54% |
| 0.1 kpc | 0.3384 | 0.0067 | 12.81% |
| 0.5 kpc | 0.4598 | 0.0055 | 53.25% |
| 1.0 kpc | 0.6180 | 0.0101 | 106.00% |
| 2.0 kpc | 0.8949 | 0.0045 | 198.29% |

## Test 3 — Particle loss

| Level | A_1 mean | A_1 std | |ΔA_1/A_1| |
|---|---|---|---|
| 0%  | 0.3017 | 0.0129 | 0.56% |
| 10%  | 0.3006 | 0.0095 | 0.21% |
| 30%  | 0.3017 | 0.0074 | 0.58% |
| 50%  | 0.3068 | 0.0070 | 2.26% |

## Key findings

* **Inclination**: at 60° tilt, |ΔA₁/A₁| = 25.4%
* **Centring**: 1 kpc off-centre → |ΔA₁/A₁| = 106.0%
* **Particle loss**: 30% loss → |ΔA₁/A₁| = 0.6%, σ = 0.0074
