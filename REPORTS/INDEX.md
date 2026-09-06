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
- **×33** vs Euler-Cromer (damped_pendulum) — sommet vs solveurs itératifs *réels* (tâches sans forme fermée). L'ancien ×1840 vs Monte-Carlo est retiré : strawman, voir AUDIT-2026-09.md §7
- KV-cache distilgpt2 : **égalité avec H2O**, pas une victoire. Le balayage complet donne spear devant à 3 budgets sur 4 (+0.09 à +0.25 pt) mais **H2O l'emporte à cap=64** (45.94 vs 45.60). Le 80.31 % souvent cité est le cap=320, celui où une politique *aléatoire* atteint déjà 73.8 % — voir `scripts/audit-kv-claim.ts`
- 2 bugs critiques moteur exterminés · CI parité 47/47 à chaque push
