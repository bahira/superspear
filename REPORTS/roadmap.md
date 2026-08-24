# Roadmap — plan en 4 phases

## Phase 1 — Publication & visibilité (semaine 1) ✅ quasi-complète
- [x] RELEASE_NOTES.md + LAUNCH_POST.md rédigés
- [x] Description + topics GitHub (via gh CLI)
- [x] npm pack vérifié — **publish restant** : `cd packages/spear-kernels && npm publish` (compte romainabdelaal@gmail.com, OTP requis)
- [ ] Post HN/Reddit à partir de LAUNCH_POST.md

## Phase 2 — Science du papier (2 semaines)
- [x] Validation KV-cache réelle distilgpt2 (80.31 % vs h2o 80.20 %)
- [x] kv_multiscale.py prêt (caps 64→320, table multi-échelle distilgpt2)
- [ ] Llama-3.2-1B multiscale : accepter licence HF + `$env:HF_TOKEN` puis
      `--models distilgpt2,Llama-3.2-1B` (~2.5 GB téléchargement)
- [x] Adaptateur PySR exporté (12 datasets + manifest, runner prêt)
- [ ] Exécuter PySR head-to-head (`pip install pysr pandas` puis run-pysr.py)
- [x] Audit de composition généralisé (REPORTS/composition-audit.md)
- [ ] Fix logit_ml wide-domain (retrain en cours à l'écriture de ce doc)
- [ ] Finaliser PAPER.md → LaTeX (paper/main.tex drafté) → arXiv cs.NE

## Phase 3 — Produit & revenus (mois 1+)
- [ ] Deploy dashboard Fly/Railway + Stripe metered sur /api/spear/discover
- [ ] Landing page avec la story ×1840
- [ ] Pack Fab/Unity « Verified Math Kernels » ($14.99) depuis le kit
- [ ] Démo MCU ESP32 des noyaux MISRA-C

## Phase 4 — Frontières recherche (fond continu)
| Frontière | Arme | Statut |
|---|---|---|
| gemv4 kernel exact | scaffold + passes profondes | convergence active (1.6e-7 @ 17u vs 7u) |
| huber constantes exactes | raffinement ciblé scaffold | proche |
| bessel au-delà de L2/L4 | seed non-mutable / contrainte coût | à tester |
| eigen slimming | fitness = MSE + λ·coût | résiste sans ça |
| primitive erfc servie | probit directe plus courte | design |
| logsumexp n-logits | généralisation LSE | design |

## Dette technique absorbable
- kv_cache hors cost-model par design → documenter dans benchmarks.ts
- probe-f32 généralisé en audit permanent inter-backends
- Nettoyer scripts debug restants
