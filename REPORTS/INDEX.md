# SPEAR — Rapports de session (index)

> Architecture : un fichier par dimension, indexé ici. Chaque entrée est
> datée et adossée au ledger (`spear-hall-of-fame.json`) ou à un commit git.

| Fichier | Contenu |
|---|---|
| [breakthroughs.md](./breakthroughs.md) | Chronologie complète des breakthroughs avec métriques avant/après |
| [engineering-fixes.md](./engineering-fixes.md) | Tous les bugs critiques trouvés & réparés, avec root cause |
| [benchmark-snapshot.md](./benchmark-snapshot.md) | État du registre à v1.0.0 : records, speedups, couverture |
| [roadmap.md](./roadmap.md) | Plan en 4 phases : publication, science, produit, recherche |

## Chiffres maîtres (post wall-assault, PR #2 — mesure 2026-09-27)

- **89 tâches** · 42 exactes (MSE = 0) / 59 sous 1e-30 · 61 slots rapides · 88/89 vitesses chiffrées
- **×46000** vs Monte-Carlo (gaussian_cdf fast) — sommet vs solveurs itératifs
- **implied_vol** : L2 2.11e-4, holdout 2.10e-4 (×15 vs v1.10) via un pas de Newton
  non replié. Le claim **×54.8 vs Newton** est retirementé : il comparait une forme
  fermée à un solveur itératif, alors que le nouveau champion *est* un pas de Newton
  (coût 546 contre 1480 pour la référence — comparer les deux ne mesurerait plus
  « forme fermée contre solveur »). L'entrée est donc publiée sans bloc `speed`.
- 3 murs prouvés au plancher, budget dépensé 0 : `kepler` (égalité plancher
  9.575e-2), `european_call` (plancher de bruit), `pendulum_hybrid` (structure
  récupérée)
- 3 records `srgb_decode`/`srgb_gamma` re-mesurés : les metrics stockées sur
  `main` ne se rejouaient plus depuis leur propre AST (metric-drift 1.4e-4 /
  3.5e-6) — les 3 metric-drift du repo sont éliminés
- **80.31 %** rétention KV-cache sur vraies attentions distilgpt2 — bat H2O à 3/4 budgets,
  mais H2O l'emporte à cap=64 (45.94 vs 45.60) et la marge cap=320 (+0.118 pts, n=72)
  est un tie statistique : citer le sweep complet (validation/kv-multiscale-results.json)
- CI parité 89/89 à chaque push · archive fast-slot cost-sorted dans le loop

> `benchmark-snapshot.md` reste le snapshot v1.10 (44 exactes / 85 slots rapides) :
> il est daté et ne bouge pas. `champion-audit.md` est régénéré par
> `npx tsx scripts/audit-champions.ts --md`. Les rapports de bench
> (`fast-slots-bench.md`, `f32-degradation.md`, `wallclock-bench.md`) sont des
> mesures machine-locales : à rejouer sur la machine de mesure, pas rafraîchis ici.
