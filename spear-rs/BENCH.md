# spear-rs — benchmark du registre

Généré par `spear bench --rows 4096 --samples 9 --repeat 8` (build release).

**Méthodologie.** Une passe = `rows` éléments par variable ; 1 chauffe puis 9 passes mesurées de 8 répétitions, médiane conservée ; sortie lue dans un `black_box`.
`ns/él` est donc **ns par élément** (divisé par le nombre de variables quand il y en a plusieurs). Ce sont des chiffres *relatifs sur cette machine*, pas des performances absolues : pas d'épinglage de cœur, pas de fréquence fixe.

`gain mesuré` = champion ÷ fast slot sur cette machine (colonne `fast ns/él` non nulle). `loi ns/él` = la loi génératrice (`data/laws.json`), absente pour 6 tâches sans forme close.

`RMSE champion` / `RMSE fast` = erreur contre la loi **sur la grille de sonde de spear-rs**, qui n'est PAS la grille d'entraînement TypeScript : à comparer aux métriques du ledger (`spear show <id>`), jamais entre elles.

| tâche | vars | coût champion | coût fast | champion ns/él | fast ns/él | gain mesuré | loi ns/él | RMSE champion | RMSE fast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `a_weighting` | 1 | 45 | 43 | 98.47 | 99.81 | ×0.99 | 109.58 | 6.27e-5 | 6.27e-5 |
| `aces_fit` | 1 | 13 | 8 | 49.09 | 23.51 | ×2.09 | 54.49 | 0.00e0 | 1.83e-2 |
| `amp_damp_fid` | 3 | 90 | 62 | 154.90 | 265.88 | ×0.58 | 85.39 | 0.00e0 | 7.77e-1 |
| `asin_hard` | 1 | 20 | 1 | 17.22 | 7.56 | ×2.28 | 17.83 | 0.00e0 | 4.59e-2 |
| `atan_unit` | 1 | 20 | 1 | 15.10 | 7.56 | ×2.00 | 13.04 | 0.00e0 | 3.04e-2 |
| `back_ease_out` | 1 | 9 | — | 36.74 | — | — | 39.67 | 0.00e0 | — |
| `bessel_i0e` | 1 | 65 | 8 | 156.81 | 46.01 | ×3.41 | 10625.92 | 1.88e1 | 1.13e2 |
| `bessel_j0` | 1 | 67 | 13 | 193.84 | 70.82 | ×2.74 | 3292.72 | 5.41e-3 | 1.62e-2 |
| `bessel_j1` | 1 | 48 | — | 60.18 | — | — | 3484.04 | 2.27e-3 | — |
| `bessel_j2` | 1 | 154 | 26 | 473.23 | 38.00 | ×12.45 | 3376.95 | 2.43e-3 | 9.95e-3 |
| `bias_slope` | 1 | 8 | 6 | 17.47 | 16.21 | ×1.08 | 17.47 | 0.00e0 | 9.16e-2 |
| `bilateral_weight` | 1 | 30 | 24 | 34.23 | 34.69 | ×0.99 | 39.60 | 0.00e0 | 3.28e-2 |
| `bilinear_interp` | 1 | 2 | 1 | 10.61 | 7.69 | ×1.38 | 7.72 | 0.00e0 | 0.00e0 |
| `blackbody_b` | 1 | 31 | 29 | 68.24 | 48.04 | ×1.42 | — | — | — |
| `blackbody_g` | 1 | 72 | — | 85.46 | — | — | — | — | — |
| `blackbody_r` | 1 | 47 | 14 | 52.45 | 39.55 | ×1.33 | 58.97 | 3.66e-17 | 1.91e-3 |
| `bs_d1_sigma` | 1 | 11 | 7 | 39.22 | 25.41 | ×1.54 | 31.97 | 1.31e-15 | 4.94e-2 |
| `bs_d2_sigma` | 1 | 7 | 6 | 20.22 | 17.07 | ×1.18 | 58.78 | 1.29e-9 | 4.94e-2 |
| `chsh_correlation` | 4 | 88 | — | 59.20 | — | — | 59.05 | 0.00e0 | — |
| `concurrence_pure` | 4 | 5 | — | 26.37 | — | — | 26.36 | 0.00e0 | — |
| `cosh_curve` | 1 | 43 | 3 | 30.25 | 15.25 | ×1.98 | 32.19 | 0.00e0 | 6.09e-2 |
| `damped_oscillation` | 1 | 44 | 27 | 40.45 | 50.65 | ×0.80 | 36.13 | 0.00e0 | 1.63e-2 |
| `damped_pendulum` | 1 | 46 | 23 | 44.19 | 23.80 | ×1.86 | — | — | — |
| `diffusion_beta` | 1 | 25 | 14 | 55.17 | 58.48 | ×0.94 | 37.84 | 5.32e-7 | 4.45e2 |
| `doppler_effect` | 2 | 7 | 3 | 26.43 | 20.90 | ×1.26 | 24.23 | 0.00e0 | 1.27e1 |
| `eigen3_sym` | 3 | 114 | 11 | 257.45 | 50.15 | ×5.13 | 167.01 | 1.35e-1 | 3.89e-1 |
| `elliptic_k` | 1 | 121 | 42 | 444.80 | 25.17 | ×17.67 | 628.41 | 0.00e0 | 6.04e1 |
| `ema_smooth` | 1 | 23 | 2 | 22.12 | 13.16 | ×1.68 | 21.91 | 0.00e0 | 3.26e-3 |
| `erf_prob` | 1 | 20 | 7 | 15.96 | 20.83 | ×0.77 | 18.55 | 0.00e0 | 8.39e-2 |
| `european_call` | 1 | 62 | 2 | 86.15 | 14.88 | ×5.79 | 20.78 | 8.62e1 | 1.16e2 |
| `fast_exp_alu` | 1 | 20 | 3 | 12.97 | 17.66 | ×0.73 | 12.27 | 0.00e0 | 3.08e-2 |
| `fog_exp2` | 1 | 24 | 7 | 25.59 | 37.23 | ×0.69 | 25.85 | 0.00e0 | 2.04e0 |
| `free_fall` | 1 | 3 | — | 16.47 | — | — | 10.97 | 4.16e-3 | — |
| `fresnel_schlick` | 1 | 9 | 8 | 44.68 | 28.23 | ×1.58 | 44.11 | 0.00e0 | 1.02e-2 |
| `gauss_shader` | 1 | 23 | 7 | 22.37 | 20.66 | ×1.08 | 26.20 | 0.00e0 | 7.09e-1 |
| `gaussian_cdf` | 1 | 23 | 8 | 27.29 | 26.93 | ×1.01 | 27.57 | 3.71e-17 | 3.00e-2 |
| `gaussian_kernel` | 1 | 23 | 3 | 21.71 | 19.12 | ×1.14 | 21.88 | 0.00e0 | 2.03e-1 |
| `gelu` | 1 | 24 | 3 | 34.49 | 16.93 | ×2.04 | 34.72 | 4.96e-17 | 5.46e-2 |
| `gemv4` | 4 | 7 | — | 41.04 | — | — | 41.06 | 0.00e0 | — |
| `grover_amplitude` | 3 | 50 | — | 52.60 | — | — | 52.86 | 0.00e0 | — |
| `hill` | 1 | 9 | — | 21.06 | — | — | 22.41 | 0.00e0 | — |
| `huber_loss` | 1 | 9 | 3 | 49.22 | 16.91 | ×2.91 | 41.60 | 0.00e0 | 8.23e-2 |
| `idm_following` | 3 | 19 | — | 80.16 | — | — | 128.82 | 3.11e-16 | — |
| `ik_reach` | 3 | 31 | 13 | 68.08 | 52.59 | ×1.29 | 67.89 | 0.00e0 | 6.04e-2 |
| `implied_vol` | 4 | 546 | 27 | 1715.71 | 129.19 | ×13.28 | — | — | — |
| `kdv_soliton` | 2 | 53 | 7 | 81.80 | 36.49 | ×2.24 | 82.35 | 0.00e0 | — |
| `kelly_criterion` | 2 | 8 | — | 34.36 | — | — | 34.05 | 0.00e0 | — |
| `kepler` | 1 | 5 | 3 | 25.70 | 19.14 | ×1.34 | 14.20 | 6.38e-3 | 1.27e1 |
| `kepler_solver` | 2 | 5025 | — | 7412.92 | — | — | 7224.35 | 8.49e-4 | — |
| `kerr` | 1 | 17 | 7 | 42.38 | 25.72 | ×1.65 | 41.24 | 8.20e-8 | 8.71e5 |
| `kerr_spin` | 2 | 21 | 9 | 56.80 | 48.71 | ×1.17 | 55.96 | 0.00e0 | 3.62e4 |
| `kv_cache` | 1 | 0 | — | 4.39 | — | — | — | — | — |
| `laguerre_l2` | 1 | 5 | 4 | 30.65 | 28.96 | ×1.06 | 31.01 | 0.00e0 | 6.29e-1 |
| `lambert_w` | 1 | 22 | 3 | 23.04 | 18.67 | ×1.23 | 19.73 | 0.00e0 | 3.78e1 |
| `layernorm_scale` | 1 | 6 | — | 14.40 | — | — | 14.45 | 0.00e0 | — |
| `legendre_p2` | 1 | 3 | — | 18.59 | — | — | 18.02 | 0.00e0 | — |
| `lennard_jones` | 1 | 17 | — | 46.48 | — | — | 46.58 | 0.00e0 | — |
| `light_falloff_punctual` | 1 | 12 | 7 | 63.46 | 24.17 | ×2.63 | 45.26 | 0.00e0 | 3.48e0 |
| `logistic_growth` | 1 | 28 | 5 | 41.82 | 33.10 | ×1.26 | 62.77 | 0.00e0 | 4.17e-2 |
| `logit_ml` | 1 | 25 | — | 28.27 | — | — | 28.20 | 0.00e0 | — |
| `logsumexp2` | 2 | 67 | 3 | 68.46 | 22.36 | ×3.06 | 69.49 | 0.00e0 | 4.99e-1 |
| `lorentz` | 1 | 9 | 8 | 25.94 | 22.15 | ×1.17 | 22.06 | 0.00e0 | 0.00e0 |
| `loschmidt_rate` | 1 | 69 | 47 | 78.81 | 71.55 | ×1.10 | 68.47 | 5.63e-6 | 2.64e-1 |
| `mel_scale` | 1 | 23 | — | 29.65 | — | — | 30.02 | 1.59e-14 | — |
| `michaelis_menten` | 1 | 6 | — | 22.84 | — | — | 22.85 | 0.00e0 | — |
| `mish` | 1 | 62 | 3 | 62.88 | 22.77 | ×2.76 | 65.37 | 0.00e0 | 7.52e-2 |
| `mm1_queue_wait` | 2 | 6 | — | 24.58 | — | — | 25.52 | 0.00e0 | — |
| `pendulum_hybrid` | 2 | 179 | 177 | 263.18 | 252.19 | ×1.04 | — | — | — |
| `pmt_finance` | 2 | 90 | 34 | 94.90 | 71.47 | ×1.33 | 94.78 | 0.00e0 | 1.16e-1 |
| `probit_quantile` | 1 | 959 | 15 | 5128.23 | 61.10 | ×83.94 | 7423.52 | 4.41e-9 | 4.68e-2 |
| `qfi_dephasing` | 3 | 28 | — | 49.13 | — | — | 48.54 | 0.00e0 | — |
| `rayleigh_phase` | 1 | 24 | 0 | 32.62 | — | — | 26.95 | 5.64e-9 | — |
| `rc_circuit` | 1 | 23 | 8 | 25.56 | 29.59 | ×0.86 | 21.00 | 2.77e-3 | 2.62e-2 |
| `rl_distillation` | 1 | 21 | 5 | 31.76 | 15.33 | ×2.07 | 31.63 | 0.00e0 | 1.04e-1 |
| `rope_freq` | 1 | 43 | — | 43.64 | — | — | 55.39 | 0.00e0 | — |
| `rope_rot` | 3 | 44 | 43 | 43.92 | 39.59 | ×1.11 | 75.30 | 0.00e0 | 0.00e0 |
| `rsi_momentum` | 2 | 6 | — | 24.33 | — | — | 23.91 | 0.00e0 | — |
| `sigmoid` | 1 | 26 | 8 | 31.06 | 30.57 | ×1.02 | 31.01 | 0.00e0 | 2.21e-2 |
| `silu` | 1 | 26 | 3 | 32.05 | 18.62 | ×1.72 | 32.09 | 0.00e0 | 8.56e-2 |
| `smootherstep` | 1 | 12 | 4 | 79.97 | 26.75 | ×2.99 | 62.73 | 0.00e0 | 1.80e-2 |
| `smoothstep` | 1 | 4 | 2 | 24.87 | 15.75 | ×1.58 | 24.92 | 0.00e0 | 3.78e-2 |
| `softplus` | 1 | 41 | 3 | 34.19 | 18.85 | ×1.81 | 34.43 | 0.00e0 | 1.69e-1 |
| `srgb_decode` | 1 | 41 | 3 | 33.50 | 28.86 | ×1.16 | 52.89 | 0.00e0 | 1.14e-2 |
| `srgb_gamma` | 1 | 45 | 4 | 80.93 | 18.62 | ×4.35 | 45.73 | 1.42e-15 | 1.00e-2 |
| `stefan_boltzmann` | 1 | 4 | — | 19.30 | — | — | 21.83 | 3.87e-15 | — |
| `swiglu_fused` | 2 | 27 | 4 | 37.92 | 24.51 | ×1.55 | 37.11 | 0.00e0 | 1.42e-1 |
| `tanh_sat` | 1 | 20 | 4 | 26.45 | 26.59 | ×0.99 | 25.93 | 0.00e0 | 8.48e-2 |
| `temperature_softmax` | 2 | 30 | 8 | 48.19 | 39.14 | ×1.23 | 47.65 | 0.00e0 | 4.66e-2 |
| `temporal_grad` | 2 | 1 | — | 12.88 | — | — | 19.02 | 0.00e0 | — |
| `uncharted2_tonemap` | 1 | 17 | 8 | 126.08 | 35.52 | ×3.55 | 89.89 | 7.86e-15 | 1.03e2 |

**Synthèse.** 64 tâches ont un fast slot ; gain médian mesuré **x1.42** (min x0.58, max x83.94).

### ⚠ 6 fast slot(s) mesurés plus lents que leur champion

Le modèle de coût les déclare « moins chers » ; la machine dit le contraire. Ce sont des fast slots dont l'arbre contient des transcendantes (souvent la loi exacte restaurée par `scripts/backfill-fast.ts`).

| tâche | gain mesuré |
|---|---:|
| `amp_damp_fid` | x0.58 |
| `damped_oscillation` | x0.80 |
| `erf_prob` | x0.77 |
| `fast_exp_alu` | x0.73 |
| `fog_exp2` | x0.69 |
| `rc_circuit` | x0.86 |

---

Régénéré par `spear bench --md`. Kernels émis : `spear emit <id> [--fast]`.
