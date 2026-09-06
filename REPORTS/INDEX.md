# SPEAR — Rapports de session (index)

> Architecture : un fichier par dimension, indexé ici. Chaque entrée est
> datée et adossée au ledger (`spear-hall-of-fame.json`) ou à un commit git.

| Fichier | Contenu |
|---|---|
| [breakthroughs.md](./breakthroughs.md) | Chronologie complète des breakthroughs avec métriques avant/après |
| [engineering-fixes.md](./engineering-fixes.md) | Tous les bugs critiques trouvés & réparés, avec root cause |
| [benchmark-snapshot.md](./benchmark-snapshot.md) | État du registre à v1.0.0 : records, speedups, couverture |
| [roadmap.md](./roadmap.md) | Plan en 4 phases : publication, science, produit, recherche |

## Chiffres maîtres (v1.10 — mesure 2026-09-06)

- **89 tâches** · 44 exactes · 85 slots rapides · 88/89 vitesses chiffrées
- **×46000** vs Monte-Carlo (gaussian_cdf fast) — sommet vs solveurs itératifs
- **×54.8** vs Newton (implied_vol, champion L2 3.18e-3 — v1.10)
- **80.31 %** rétention KV-cache sur vraies attentions distilgpt2 (bat H2O)
- CI parité 89/89 à chaque push · archive fast-slot cost-sorted dans le loop
