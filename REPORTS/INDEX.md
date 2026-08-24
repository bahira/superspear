# SPEAR — Rapports de session (index)

> Architecture : un fichier par dimension, indexé ici. Chaque entrée est
> datée et adossée au ledger (`spear-hall-of-fame.json`) ou à un commit git.

| Fichier | Contenu |
|---|---|
| [breakthroughs.md](./breakthroughs.md) | Chronologie complète des breakthroughs avec métriques avant/après |
| [engineering-fixes.md](./engineering-fixes.md) | Tous les bugs critiques trouvés & réparés, avec root cause |
| [benchmark-snapshot.md](./benchmark-snapshot.md) | État du registre à v1.0.0 : records, speedups, couverture |
| [roadmap.md](./roadmap.md) | Plan en 4 phases : publication, science, produit, recherche |

## Chiffres maîtres (v1.0.0+)

- **50 tâches** · 14 exactes · 37 slots rapides · 46/48 vitesses chiffrées
- **×1840** vs Monte-Carlo (gaussian_cdf) — sommet vs solveurs itératifs
- **80.31 %** rétention KV-cache sur vraies attentions distilgpt2 (bat H2O)
- 2 bugs critiques moteur exterminés · CI parité 47/47 à chaque push
