# float32 degradation — do the champions survive the MISRA target?

The ledger reasons in float64, but the MISRA-C export emits `float32_t`. This report re-evaluates every champion with all arithmetic rounded to f32.

**40 champions checked. 30 are machine-exact in f64 but not in f32.**

This is expected physics, not a defect: f32 carries ~7 significant digits. It is listed so nobody ships an "exact" kernel to an embedded target without knowing its real f32 error.

| task | mse f64 | f32 deviation | exact in f64 | exact in f32 |
|---|---|---|---|---|
| `mel_scale` | 3.95e-25 | 2.29e-8 | no | no |
| `rope_freq` | 8.45e-7 | 6.82e-11 | no | no |
| `laguerre_l2` | 0.00e+0 | 1.27e-12 | yes | no |
| `logit_ml` | 0.00e+0 | 7.89e-13 | yes | no |
| `a_weighting` | 2.63e-30 | 5.43e-13 | yes | no |
| `smootherstep` | 3.57e-31 | 5.53e-14 | yes | no |
| `probit_quantile` | 1.90e-5 | 3.90e-14 | no | no |
| `bessel_j0` | 2.64e-4 | 3.71e-14 | no | no |
| `blackbody_b` | 2.26e-4 | 1.69e-14 | no | no |
| `silu` | 0.00e+0 | 1.68e-14 | yes | no |
| `mish` | 0.00e+0 | 1.67e-14 | yes | no |
| `gelu` | 1.55e-34 | 7.82e-15 | yes | no |
| `cosh_curve` | 3.57e-33 | 6.68e-15 | yes | no |
| `bias_slope` | 0.00e+0 | 6.42e-15 | yes | no |
| `blackbody_g` | 1.30e-4 | 5.42e-15 | no | no |
| `back_ease_out` | 2.17e-32 | 2.92e-15 | yes | no |
| `huber_loss` | 0.00e+0 | 2.65e-15 | yes | no |
| `bs_d2_sigma` | 1.62e-18 | 2.09e-15 | no | no |
| `bs_d1_sigma` | 1.80e-30 | 1.98e-15 | yes | no |
| `bessel_j2` | 1.03e-4 | 1.91e-15 | no | no |
| `aces_fit` | 0.00e+0 | 1.85e-15 | yes | no |
| `srgb_gamma` | 2.72e-28 | 1.52e-15 | yes | no |
| `legendre_p2` | 3.25e-33 | 1.14e-15 | yes | no |
| `bessel_j1` | 5.20e-6 | 1.06e-15 | no | no |
| `erf_prob` | 0.00e+0 | 8.84e-16 | yes | no |
| `asin_hard` | 0.00e+0 | 6.90e-16 | yes | no |
| `smoothstep` | 0.00e+0 | 6.82e-16 | yes | no |
| `fresnel_schlick` | 1.14e-33 | 5.30e-16 | yes | no |
| `sigmoid` | 0.00e+0 | 5.28e-16 | yes | no |
| `gaussian_cdf` | 1.39e-34 | 3.45e-16 | yes | no |
