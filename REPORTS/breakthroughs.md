# Chronologie des breakthroughs

> Chaque entrée : tâche · avant → après · arme utilisée · session.

## Résolutions exactes (14)

| # | Tâche | Méthode de chute | Session |
|---|---|---|---|
| 1 | gaussian_kernel | structure vraie `e^(−x²/2)` retrouvée, MSE 5.4e-34 | sweeps 500 |
| 2 | rope_rot | seeds trigonométriques + 6000 iters | vague LLM |
| 3 | atan_unit | Padé [3/2] redécouvert, MSE 0 | registre ultra-common |
| 4 | ema_smooth | quasi-exact 1.6e-11 | registre ultra-common |
| 5 | sigmoid | exact (exp servi) | historique |
| 6 | lorentz | `1/√(1−b²)` exact | historique |
| 7 | lambert_w | `x·relu(exp(x))` | historique |
| 8 | ik_reach | primitive atan + scaffold identité acos→atan : **1.1e-13** | unlock atan |
| 9 | eigen3_sym | Cardano complet via atan : **3.3e-11** | post-fix simplify |
| 10 | smoothstep | polynôme exact au coût minimal prouvé (5u) | profondeur 5000 |
| 11 | huber_loss | astuce max adoptée via scaffold : **3.0e-7 L5** | round 2 seeds |
| 12 | atan_unit v2 | raffinement → MSE 0 exact | grand sweep 1 |
| 13 | ik_reach v2 | raffiné à **4.06e-32** | grand sweep 1 |
| 14 | logit_ml→atan_unit famille | composition vérifiée | audit paires |

## Cascades majeures

| Tâche | Trajectoire | Cumul | Arme décisive |
|---|---|---|---|
| kerr_spin | 2.9e-5 → 1.4e-7 → … → **7.5e-10** | ×38,000 | passes ciblées répétées |
| gemv4 | 4e-4 → … → **1.6e-7** | **×2500** | scaffold linéaire perturbé |
| idm_following | 2.17 → **4.46e-3 L2** | ×450 | scaffold structure IDM |
| bessel_j1 | 0.56 → **5.2e-6 L4** | **×108,000** | fix référence empoisonnée |
| bessel_j0 | 1.6e-2 → **2.6e-4 L2** | ×60 | fix référence + série vraie |
| softplus | 2.4e-4 → **1.9e-5** | ×75 cumul? (voir ledger) | sweeps répétés |
| hill | 1.16e-4 → **4.47e-6** | ×26 | cascades 2000 |
| tanh_sat | 7.3e-4 L1 → **4.6e-6 L4** | ×158 | seed de forme + palier |
| logistic_growth | 8.6e-5 → 1.1e-5 | ×7.8 | sweep complet |
| gaussian_cdf | 1.4e-4 → 1.64e-5 | ×8.5 | sweeps complets |

## Découvertes de structures inédites

- **pmt_finance** : hybride `atan(n)` dans un kernel d'emprunt — jamais écrit par humain ni IA ; ×1.38 plus rapide que la forme manuelle exp∘ln.
- **cosh_curve** : `√x²·cos(x²)` remplace `(eˣ+e⁻ˣ)/2` — ×1.38 plus rapide à L5.
- **probit_quantile** : hybride `√√log + x²` à parité de coût avec Acklam dès la génération 1.

## Validations réelles

- **KV-cache distilgpt2** (72 échantillons couches×prompts, politique `4.5·S + A + R`, 5 unités) — **à lire comme une égalité avec H2O, pas comme une victoire.**

  | cap | spear | h2o | écart | random | oracle |
  |---|---|---|---|---|---|
  | 64 | 45.60 | **45.94** | −0.335 | 17.16 | 55.20 |
  | 128 | 53.67 | 53.58 | +0.091 | 29.91 | 67.66 |
  | 256 | 72.20 | 71.95 | +0.252 | 63.42 | 85.55 |
  | 320 | **80.31** | 80.20 | +0.118 | 73.82 | 90.85 |

  Le 80.31 % initialement mis en avant ne cite que `cap=320`. Deux réserves :
  **H2O gagne à cap=64**, et à cap=320 une politique *aléatoire* retient déjà
  73.8 %, donc le chiffre phare n'est qu'à 6.5 pts du hasard — contre 28.4 pts
  à cap=64, le régime où la politique travaille vraiment. L'écart de +0.118 pt
  est ~500× plus petit que la dispersion des échantillons (68.5 pts) : sans
  écarts-types par échantillon, ce n'est pas une victoire démontrée.
  Rejouable sans torch via `npx tsx scripts/audit-kv-claim.ts`.
- **Edge CPU WASM** : SiLU ×2.12, GELU ×2.32 vs production ; bloc FFN +10.2 %.
- **Honnête négatif** : swap GELU dans PyTorch = +0.56 ppl, −9.2 % tok/s — les kernels natifs fusionnés gagnent in-framework ; frontière de déploiement documentée.
