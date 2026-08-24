# Snapshot registre — v1.0.0+ (48 tâches, 14 exactes)

> Généré depuis le ledger live. Chiffres : coût en unités ALU/SFU,
> speedup vs loi exacte, vs solveur itératif quand applicable.

## Résolues exactement (MSE ≤ 1e-8)

gaussian_kernel (5.4e-34) · rope_rot (0) · atan_unit (0) · ema_smooth
(1.6e-11) · sigmoid (0) · lorentz (0) · lambert_w (0) · ik_reach (4.1e-32)
· eigen3_sym (3.3e-11) · smoothstep (3.3e-7 @ 5u optimal) · huber_loss
(3.0e-7 L5) · gaussian_cdf→L3 · rc_circuit (7.26e-5 L5) · layernorm (0)

## Plus rapides que la référence exacte

| Tâche | Speedup | | Tâche | Speedup |
|---|---|---|---|---|
| logsumexp2 | ×8.57 | | blackbody_r | ×1.79 |
| kdv_soliton | ×7.57 | | tanh_sat | ×1.67 |
| gelu | ×6.57 | | damped_oscillation | ×1.63 |
| pendulum_hybrid | ×3.76 | | cosh_curve | ×1.38 |
| idm_following | ×3.33 | | bessel_j1 | ×1.38 |
| rl_distillation | ×2.83 | | gaussian_cdf | ×1.36 |
| silu | ×2.43 | | kerr / sigmoid / kerr_spin | ×1.17–1.31 |

## Vs solveurs itératifs

| Tâche | Solveur | Gain |
|---|---|---|
| gaussian_cdf | Monte-Carlo 1000 tirages | **×1840** |
| damped_oscillation | RKF45 300 pas | ×200 |
| kerr | RK4 géodésique 200 pas | ×185 |
| kerr_spin | RK4 géodésique | ×133 |
| damped_pendulum | Euler-Cromer 60 pas | ×33 |
| ik_reach | Newton-DLS 8 it | ×6 |

## Slots rapides (37/48)

Top : srgb_decode ×26, gaussian_kernel ×24, lambert_w ×22, blackbody_r
×14, softplus ×10.8, cosh_curve ×10.7, logistic_growth ×9, lennard_jones
×7.1, logsumexp2 ×7, hill ×3…

## Couverture

- Arbres AST : **47/48** (layernorm_scale = fastTree seul, record exact protégé)
- Vitesses : **46/48** chiffrées (kv_cache hors cost-model par design ; 1 restant)
- Slots rapides : 37/48

## Murs restants (tous diagnostiqués, sans excuse de données)

- gemv4 : 1.6e-7 à 17u vs optimal 7u — scaffold adopté, constantes en convergence
- huber_loss : constantes exactes du max-trick à coller (scaffold adopté, L5 atteint)
- eigen slimming : 120u exact résiste à l'amaigrissement sans objectif pondéré coût
- logit_ml : dérive hors-domaine détectée par audit de composition — retrain wide-domain en cours
