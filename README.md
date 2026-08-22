# SPEAR Lab â€” Symbolic Pareto Evolutionary Algorithm for Research

Un moteur de **rÃ©gression symbolique** (programmation gÃ©nÃ©tique multi-objectif) qui dÃ©couvre des **lois mathÃ©matiques en forme fermÃ©e** Ã  partir de donnÃ©es â€” puis les compile en **WebAssembly vÃ©rifiÃ© bit-perfect**.

Pas de rÃ©seau de neurones, pas de boÃ®te noire : ce qui sort, c'est une formule que tu peux lire, auditer et dÃ©ployer sur un microcontrÃ´leur.

## ðŸ† Hall of Fame

Meilleures dÃ©couvertes toutes graines confondues (ledger complet : [`spear-hall-of-fame.json`](./spear-hall-of-fame.json), agrÃ©gÃ© par [`scripts/hall-of-fame.ts`](./scripts/hall-of-fame.ts)).

### Lois physiques retrouvÃ©es exactement

| TÃ¢che | Record (MSE) | âš¡ CoÃ»t vs loi exacte | Formule dÃ©couverte | Graine |
|---|---|---|---|---|
| **Chute libre** | 2.9e-4 | â€” | `4.9047Â·tÂ²` â†’ **g = 9.809 mÂ·sâ»Â²** (err 0.03 %) | 161803 |
| **Kepler** | 4.7e-2 | â€” | `1.0009Â·aÂ·âˆš\|a\|` â€” exposant 3/2 retrouvÃ© | 555666 |
| **Circuit RC** | **7.3e-5** | â€” | `-0.998Â·min(exp(âˆ’t), 2) + 0.997` â‰ˆ `1 âˆ’ e^(âˆ’t/Ï„)` | 777888 |
| **Lambert Wâ‚€** | 0 (exact) | â€” | `xÂ·relu(exp(x))` | 12345 |
| **Gradient vidÃ©o** | 0 (exact) | Ã—1.00 (1/1) | `b âˆ’ a` | 8888 |
| **Upsampling bilinÃ©aire** | 0 (exact) | Ã—0.50 (2/1) | `1 âˆ’ u` | 12345 |
| **DÃ©flexion de Kerr** ðŸ†• | **7.4e-5** | Ã—1.06 (16/17) | forme rationnelle en `1/(b+c)` retrouvÃ©e | 424242 |

### Kernels LLM / image remplacÃ©s par de l'algÃ¨bre pure

| TÃ¢che | Record | âš¡ CoÃ»t vs noyau exact | Formule dÃ©couverte | Baseline battue |
|---|---|---|---|---|
| **Diffusion Î²(t)** | **1.8e-5** | Ã—0.53 (45/24) | `-5.61Â·exp(cos(min(c,t) âˆ’ tÂ²)) + 3.05` | schedule cosinus |
| **Blur gaussien** | **9.2e-5** | Ã—0.53 (43/23) | `-0.427Â·(âˆ’exp(cos(x))) âˆ’ 0.14` âš ï¸ in-domain | noyau `exp(âˆ’xÂ²/2ÏƒÂ²)` |
| **LayerNorm rsqrt** | 0 (exact) | â€” | `x/âˆš\|xÂ³\|` = `1/âˆšx` | instruction SFU |
| **SiLU/Swish** | 8.3e-4 | **Ã—2.43** (14/34) | `xÂ·(0.501 + 0.589Â·x/(0.83 + âˆš(1+xÂ²)))` | HardSwish, ReLU |
| **GELU** | 5.3e-4 | **Ã—6.57** (7/46) | `xÂ·min(1.002, relu(0.308x + 0.501))` | GELU-tanh |
| **Sigmoid** | 0 (exact)* | â€” | `1 âˆ’ 1/(1 + eâ»Ë£)` | Hard-sigmoid TFLite |
| **Distillation RL** | 2.3e-4 | **Ã—2.83** (12/34) | `(x + 0.145xÂ³)/(0.556 + 0.75xÂ²)` â€” PadÃ© [3/2] spontanÃ© | rÃ©seau tanh |

### Pharmacologie & physique relativiste ðŸ†•

| TÃ¢che | Record | âš¡ CoÃ»t vs loi exacte | Formule dÃ©couverte | Note |
|---|---|---|---|---|
| **Ã‰quation de Hill** (dose-rÃ©ponse) | 9.2e-4 | **Ã—1.80** (5/9) | `0.532Â·min(c, min(cÂ², 1.83)) âˆ’ 0.017` | plus rapide que la loi d'EC50, prÃ©cision conservÃ©e |
| **Facteur de Lorentz Î³(Î²)** | 2.4e-2 | Ã—0.30 (27/8) | forme exponentielle sur le domaine [0, 0.99] | la forme rationnelle exacte reste Ã  dÃ©couvrir |

### DÃ©cision (KV-cache)

| TÃ¢che | Record | RÃ¨gle dÃ©couverte | Baselines battues |
|---|---|---|---|
| **Ã‰viction KV-cache** | **67.0 %** de masse d'attention future conservÃ©e | `4Â·S + A + 1.5Â·R` | H2O, StreamingLLM, SnapKV, fenÃªtre glissante, alÃ©atoire |

RÃ¨gle tri-dimensionnelle (Sinks + Attention accumulÃ©e + RÃ©cence) dÃ©couverte par Ã©volution â€” la triade que la littÃ©rature a mis des annÃ©es Ã  identifier, retrouvÃ©e en **4 itÃ©rations**.

> **âš¡ ModÃ¨le de coÃ»t** : `multiplicateur = coÃ»t_noyau_exact / coÃ»t_formule`, en unitÃ©s ALU/SFU GPU (`mul/add` = 1, `div` = 4, `sqrt` = 2, `exp/cos` â‰ˆ 20). `Ã—6.57` pour GELU signifie que la formule Ã©voluÃ©e coÃ»te **6.57Ã— moins** que le kernel GELU-tanh de rÃ©fÃ©rence par Ã©lÃ©ment infÃ©rÃ©. Les Â« â€” Â» sont les records trouvÃ©s avant l'extension du modÃ¨le de coÃ»t : ils se remplissent automatiquement quand un run reproduit la formule championne (le speedup est une fonction pure de l'AST).

## Comment Ã§a marche

```
donnÃ©es â†’ population d'arbres algÃ©briques â†’ NSGA-II (mÃ©trique â†” taille)
   â†‘                                              â†“
   â””â”€â”€ UCB budget allocation â† stagnation â† Pareto rank-0
```

Le moteur (`src/lib/spear/`) combine :

- **GP multi-objectif** : tri non-dominÃ© + crowding distance, parsimonnie croissante (les formules doivent Ãªtre *petites* ET *prÃ©cises*) ;
- **RNG seedÃ© de bout en bout** : donnÃ©es, Ã©volution et bruit sont reproductibles depuis une graine (`setSeed` + source uniforme injectÃ©e) ;
- **Multi-start warm-up** : formes primitives (rationnelles, PadÃ© [3/2], exponentielles) raffinÃ©es par descente de coordonnÃ©es avant l'Ã©volution ;
- **Allocation UCB** du budget entre tÃ¢ches : exploite ce qui progresse, explore ce qui stagne ;
- **Anti-stagnation** : polish de constantes, mutations structurelles, rÃ©-injection de formes ;
- **Scoring honnÃªte** : sÃ©lection sur split train, rapport sur split holdout jamais vu (tÃ¢che KV-cache) ;
- **Simplification algÃ©brique** : folding de constantes, collapse des formes imbriquÃ©es (`câ‚Â·(câ‚‚Â·x) â†’ (câ‚câ‚‚)Â·x`, `(x+a)âˆ’a â†’ x`) â€” Ã©limine les Â« records Â» dÃ©gÃ©nÃ©rÃ©s et accÃ©lÃ¨re la recherche (~3Ã—) ;
- **Export vÃ©rifiÃ©** : chaque formule est compilÃ©e vers **Python (torch)**, **C (CUDA)** et **WebAssembly** â€” le smoke test compare op-par-op les sorties JS/WASM (paritÃ© 0e+0).

## DÃ©marrage

```bash
npm install
# Postgres requis (DATABASE_URL dans .env)
npm run dev          # â†’ http://localhost:3000
```

L'interface propose le **Grounded Loop** (les 16 tÃ¢ches, budget 30â€“2000 itÃ©rations), les labs par preset (activations, KV-cache, rÃ©gression CSV custom) et l'historique des runs persistÃ© en Postgres.

### Scripts headless

```bash
# Run live : boucle grounded + fusion des records au hall of fame
npx tsx scripts/hall-of-fame.ts <seed> <budget> [deadlineMs]

# Rebuild du ledger depuis les logs de runs existants
npx tsx scripts/hall-of-fame.ts

# ParitÃ© WASM op-par-op + run budget 500
npx tsx wasm-smoke.test.ts
```

## ReproductibilitÃ©

Chaque dÃ©couverte du Hall of Fame est rejouable :

```bash
npx tsx scripts/hall-of-fame.ts 777888 500   # â†’ redÃ©couvre le record RC (7.3e-5)
```

MÃªme graine â‡’ mÃªmes donnÃ©es, mÃªmes formules, mÃªmes mÃ©triques. Le ledger trace **quelle graine, Ã  quelle itÃ©ration**, chaque record a Ã©tÃ© trouvÃ©.

## Structure

```
src/lib/spear/
  engine.ts       # AST, RNG seedÃ©, simplify, NSGA-II, crossover/mutation, codegen
  benchmarks.ts   # 16 tÃ¢ches : activations, KV-cache, physique, image/vidÃ©o
  loop.ts         # grounded loop : UCB, warm-up, anti-stagnation, snapshots
  presets.ts      # labs mono-tÃ¢che (activation, KV-cache, CSV custom)
  wasm.ts         # compilateur AST â†’ WebAssembly (encoder maison, zÃ©ro toolchain)
  math-utils.ts   # mse, linf, erf, gaussiennes
src/app/          # dashboard Next.js 16 + API routes (Postgres/Drizzle)
scripts/          # hall-of-fame
```

## Notes d'honnÃªtetÃ©

- Les records sont mesurÃ©s **sur le domaine des benchmarks**. `exp(cos(x))` approxime trÃ¨s bien le noyau gaussien sur [âˆ’3, 3] â€” et diverge complÃ¨tement en dehors. C'est marquÃ© âš ï¸ dans le ledger.
- Les baselines Â« oracle Â» (loi exacte, attention future) bornent le score atteignable : battre la baseline *dÃ©ployable* est le vrai signal.
- Les jalons de rÃ©gression sont calibrÃ©s sur le **plancher de bruit** mesurÃ©, pas sur des seuils arbitraires.
