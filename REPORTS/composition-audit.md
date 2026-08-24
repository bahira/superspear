# Composition Audit — 2026-08-24T02:54:24.301Z

## Inverse pair [sigmoid, logit_ml]

| composition | domain | max error |
|---|---|---|
| sigmoid∘logit_ml | x ∈ [0.02, 0.98] | 1.060e-2 |
| logit_ml∘sigmoid | y ∈ [-6, 6] | 1.628e+0 |

**PAIRE DÉGRADÉE** — au moins une dérive

## Exact-grade sweep (metric < 0.000001, single-var trees, 400 points)

| task | metric | sweep | output range | finite | monotone (expected dir) | anomalies |
|---|---|---|---|---|---|---|
| sigmoid | 4.791e-33 | [-10, 10] | [4.540e-5, 1.000e+0] | OK | OK (inc) | 0 |
| lambert_w | 0.000e+0 | [-10, 10] | [-3.679e-1, 2.203e+5] | OK | OK (inc) | 0 |
| layernorm_scale | 0 | — | — | — | — | skipped: no top-level tree |
| gaussian_kernel | 5.356e-34 | [-10, 10] | [1.389e-11, 1.000e+0] | OK | OK (dec) | 0 |
| bilinear_interp | 0.000e+0 | [0, 1] | [0.000e+0, 1.000e+0] | OK | OK (dec) | 0 |
| temporal_grad | 0 | — | — | — | — | n/a: multi-var (b, a) |
| lorentz | 0.000e+0 | [-10, 10] | [1.005e-1, 1.000e+4] | OK | n/a | 0 |
| kerr | 1.398e-7 | [-10, 10] | [-4.044e+4, 4.044e+4] | OK | n/a | 0 |
| kerr_spin | 7.518e-10 | [-10, 10] | [-7.044e+1, 3.991e+3] | OK | n/a | 0 |
| eigen3_sym | 3.325277645075658e-11 | — | — | — | — | n/a: multi-var (t, u, w) |
| ik_reach | 4.0626336618882106e-32 | — | — | — | — | n/a: multi-var (d, l2, l3) |
| gemv4 | 1.6074532951505583e-7 | — | — | — | — | n/a: multi-var (x0, x1, x2, x3) |
| rope_rot | 0 | — | — | — | — | n/a: multi-var (th, y, x) |
| gauss_shader | 8.177e-34 | [-10, 10] | [1.389e-11, 1.000e+0] | OK | OK (dec) | 0 |
| ema_smooth | 1.635e-11 | [-10, 10] | [-1.191e+3, 4.111e+2] | OK | n/a | 0 |
| smoothstep | 3.284e-7 | [-10, 10] | [-1.689e+3, 2.287e+3] | OK | n/a | 0 |
| atan_unit | 0.000e+0 | [-10, 10] | [-1.471e+0, 1.471e+0] | OK | OK (inc) | 0 |
| srgb_decode | 2.544e-7 | [-10, 10] | [1.458e-3, 1.447e+2] | OK | n/a | 0 |
| huber_loss | 2.992e-7 | [-10, 10] | [-2.811e-4, 9.514e+0] | OK | n/a | 0 |
| cosh_curve | 2.633e-8 | [-10, 10] | [1.000e+0, 1.739e+3] | OK | n/a | 0 |
| pmt_finance | 4.3680216924352057e-7 | — | — | — | — | n/a: multi-var (n, r) |

Audited 14 exact-grade single-var kernels out of 53 ledger entries.

## Findings

- **[sigmoid, logit_ml]**: PAIRE DÉGRADÉE — logit_ml est une approximation (metric=8.206e-4); l'erreur de composition explose aux extrêmes de probabilité (logit∘sigmoid max 1.628e+0 sur [-6,6]). Restreindre l'usage au domaine d'entraînement ou re-miner logit_ml.
- **lambert_w**: formula contains clamp constructs (relu) — piecewise fit; extrapolation beyond training range unreliable (sweep outputs reached [-3.679e-1, 2.203e+5]).
- **layernorm_scale**: metric=0 but no `tree` field in ledger (fastTree only) — excluded from sweep.
- **temporal_grad**: exact-grade but multi-var (b, a) — single-axis sweep not applicable.
- **eigen3_sym**: exact-grade but multi-var (t, u, w) — single-axis sweep not applicable.
- **ik_reach**: exact-grade but multi-var (d, l2, l3) — single-axis sweep not applicable.
- **gemv4**: exact-grade but multi-var (x0, x1, x2, x3) — single-axis sweep not applicable.
- **rope_rot**: exact-grade but multi-var (th, y, x) — single-axis sweep not applicable.
- **ema_smooth**: formula contains clamp constructs (max) — piecewise fit; extrapolation beyond training range unreliable (sweep outputs reached [-1.191e+3, 4.111e+2]).
- **srgb_decode**: formula contains clamp constructs (min, max) — piecewise fit; extrapolation beyond training range unreliable (sweep outputs reached [1.458e-3, 1.447e+2]).
- **huber_loss**: formula contains clamp constructs (max) — piecewise fit; extrapolation beyond training range unreliable (sweep outputs reached [-2.811e-4, 9.514e+0]).
- **pmt_finance**: exact-grade but multi-var (n, r) — single-axis sweep not applicable.
