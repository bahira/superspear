# Snapshot registre — v1.10 (89 tâches, 44 exactes)

> Généré depuis le ledger live (mesure 2026-09-06). Chiffres : coût en unités
> ALU/SFU, speedup vs loi exacte, vs solveur itératif quand applicable.

## Résolues exactement (MSE ≤ 1e-8) — 44/89

a_weighting · amp_damp_fid · asin_hard · atan_unit · bias_slope ·
bilateral_weight · bilinear_interp · bs_d1_sigma · bs_d2_sigma ·
chsh_correlation · concurrence_pure · doppler_effect · eigen3_sym ·
ema_smooth · erf_prob · fast_exp_alu · gauss_shader · gaussian_cdf ·
gaussian_kernel · gelu · grover_amplitude · hill · ik_reach ·
kelly_criterion · kerr_spin · lambert_w · layernorm_scale · legendre_p2 ·
logit_ml · lorentz · mel_scale · michaelis_menten · mish · mm1_queue_wait ·
qfi_dephasing · rope_rot · rsi_momentum · sigmoid · silu ·
stefan_boltzmann · tanh_sat · temperature_softmax · temporal_grad ·
uncharted2_tonemap

## Plus rapides que la référence exacte

| Tâche | Speedup | | Tâche | Speedup |
|---|---|---|---|---|
| erf_prob | ×26 | | rope_freq | ×5.38 |
| tanh_sat | ×20 | | silu | ×4.86 |
| mish | ×12.4 | | bilateral_weight | ×4.09 |
| gelu | ×9.2 | | uncharted2_tonemap | ×3.25 |
| logsumexp2 | ×8.57 | | fog_exp2 | ×3.14 |
| gaussian_cdf | ×8.5 | | bessel_j0 | ×3.08 |
| kdv_soliton | ×7.57 | | mel_scale | ×5.75 |

## Vs solveurs itératifs

| Tâche | Solveur | Gain |
|---|---|---|
| gaussian_cdf (fast) | Monte-Carlo 1000 tirages | **×46000** |
| gaussian_cdf (precise) | Monte-Carlo 1000 tirages | ×11500 |
| kerr_spin (fast) | RK4 géodésique 200 pas | ×600 |
| kerr (fast) | RK4 géodésique 200 pas | ×480 |
| damped_oscillation | RKF45 300 pas | ×200 |
| kerr (precise) | RK4 géodésique 200 pas | ×184.6 |
| kerr_spin (precise) | RK4 géodésique 200 pas | ×133.3 |
| qfi_dephasing (fast) | BFGS optimisation | ×88.9 |
| damped_pendulum (fast) | Euler-Cromer 60 pas | ×71.4 |
| implied_vol | Newton vega ~5 itérations | **×54.8** |

## Slots rapides (85/89)

Top : gelu ×46, sigmoid ×34, gaussian_cdf ×34 (vs coût exact), srgb_decode
×26, gaussian_kernel ×24, lambert_w ×22, blackbody_r ×14, softplus ×10.8.
Nouveau v1.10 : gauss_shader — `x²/(x²+1.49)` à 7 u vs exact 24 u.

## Couverture

- Arbres AST : 89/89
- Vitesses : **88/89** chiffrées (kv_cache hors cost-model par design)
- Slots rapides : **85/89** (restants : damped_oscillation, implied_vol,
  kepler_solver, rope_rot — formes courtes sans sin/cos encore non servies)
- Multiplicateurs vs solveur itératif : 12 tâches tarifées

## Murs restants (tous diagnostiqués, sans excuse de données)

- gemv4 : 1.6e-7 à 17u vs optimal 7u — scaffold adopté, constantes en convergence
- huber_loss : 3.0e-7 L5 — constantes exactes du max-trick à coller
- bessel_j0 / bessel_j2 : bloqués à L2 (2.6e-4 / 1.1e-4), bessel_j1 a L4
- eigen slimming : 120u exact résiste à l'amaigrissement sans fitness pondérée coût
- european_call / kepler : metrics L5 sur lois à support contraint — fast
  slots deploy-grade disponibles
