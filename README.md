# SPEAR Lab — Symbolic Pareto Evolutionary Algorithm for Research

Un moteur de **régression symbolique** (programmation génétique multi-objectif) qui découvre des **lois mathématiques en forme fermée** à partir de données — puis les compile en **WebAssembly vérifié bit-perfect**.

Pas de réseau de neurones, pas de boîte noire : ce qui sort, c'est une formule que tu peux lire, auditer et déployer sur un microcontrôleur.

## 🏆 Hall of Fame

Meilleures découvertes toutes graines confondues (ledger complet : [`spear-hall-of-fame.json`](./spear-hall-of-fame.json), agrégé par [`scripts/hall-of-fame.ts`](./scripts/hall-of-fame.ts)).

### Lois physiques retrouvées exactement

| Tâche | Record (MSE) | Formule découverte | Graine |
|---|---|---|---|
| **Chute libre** | 2.9e-4 | `4.9047·t²` → **g = 9.809 m·s⁻²** (err 0.03 %) | 161803 |
| **Kepler** | 4.7e-2 | `1.0009·a·√\|a\|` — exposant 3/2 retrouvé | 555666 |
| **Circuit RC** | **7.3e-5** | `-0.998·min(exp(−t), 2) + 0.997` ≈ `1 − e^(−t/τ)` | 777888 |
| **Lambert W₀** | 0 (exact) | `x·relu(exp(x))` | 12345 |
| **Gradient vidéo** | 0 (exact) | `1.001·sin(b − a)` ≈ `b − a` | 12345 |
| **Upsampling bilinéaire** | 0 (exact) | `1 − u` | 12345 |

### Kernels LLM / image remplacés par de l'algèbre pure

| Tâche | Record | Formule découverte | Baseline battue |
|---|---|---|---|
| **Diffusion β(t)** | **5.6e-5** | `0.471·(relu(t²) − 1.296)³ + 1.038` | schedule cosinus |
| **Blur gaussien** | **9.2e-5** | `-0.427·(−exp(cos(x))) − 0.14` ⚠️ in-domain | noyau `exp(−x²/2σ²)` |
| **LayerNorm rsqrt** | 0 (exact) | `x/√\|x³\|` = `1/√x` | instruction SFU |
| **SiLU/Swish** | 8.3e-4 | `x·(0.501 + 0.589·x/(0.83 + √(1+x²)))` | HardSwish, ReLU |
| **GELU** | 5.3e-4 | `x·min(1.002, relu(0.308x + 0.501))` | GELU-tanh |
| **Sigmoid** | 0 (exact)* | `1 − 1/(1 + e⁻ˣ)` | Hard-sigmoid TFLite |
| **Distillation RL** | 3.3e-4 | `(x + 0.154x³)/(0.525 + 0.75x²)` — Padé [3/2] trouvé **spontanément** | réseau tanh |

\* la tâche sigmoid autorise `exp` comme primitive — l'intérêt est la comparaison de coût, pas la pureté algébrique.

### Décision (KV-cache)

| Tâche | Record | Règle découverte | Baselines battues |
|---|---|---|---|
| **Éviction KV-cache** | **67.0 %** de masse d'attention future conservée | `4·S + A + 1.5·R` | H2O, StreamingLLM, SnapKV, fenêtre glissante, aléatoire |

Règle tri-dimensionnelle (Sinks + Attention accumulée + Récence) découverte par évolution — la triade que la littérature a mis des années à identifier, retrouvée en **4 itérations**.

## Comment ça marche

```
données → population d'arbres algébriques → NSGA-II (métrique ↔ taille)
   ↑                                              ↓
   └── UCB budget allocation ← stagnation ← Pareto rank-0
```

Le moteur (`src/lib/spear/`) combine :

- **GP multi-objectif** : tri non-dominé + crowding distance, parsimonnie croissante (les formules doivent être *petites* ET *précises*) ;
- **RNG seedé de bout en bout** : données, évolution et bruit sont reproductibles depuis une graine (`setSeed` + source uniforme injectée) ;
- **Multi-start warm-up** : formes primitives (rationnelles, Padé [3/2], exponentielles) raffinées par descente de coordonnées avant l'évolution ;
- **Allocation UCB** du budget entre tâches : exploite ce qui progresse, explore ce qui stagne ;
- **Anti-stagnation** : polish de constantes, mutations structurelles, ré-injection de formes ;
- **Scoring honnête** : sélection sur split train, rapport sur split holdout jamais vu (tâche KV-cache) ;
- **Simplification algébrique** : folding de constantes, collapse des formes imbriquées (`c₁·(c₂·x) → (c₁c₂)·x`, `(x+a)−a → x`) — élimine les « records » dégénérés et accélère la recherche (~3×) ;
- **Export vérifié** : chaque formule est compilée vers **Python (torch)**, **C (CUDA)** et **WebAssembly** — le smoke test compare op-par-op les sorties JS/WASM (parité 0e+0).

## Démarrage

```bash
npm install
# Postgres requis (DATABASE_URL dans .env)
npm run dev          # → http://localhost:3000
```

L'interface propose le **Grounded Loop** (les 16 tâches, budget 30–2000 itérations), les labs par preset (activations, KV-cache, régression CSV custom) et l'historique des runs persisté en Postgres.

### Scripts headless

```bash
# Run live : boucle grounded + fusion des records au hall of fame
npx tsx scripts/hall-of-fame.ts <seed> <budget> [deadlineMs]

# Rebuild du ledger depuis les logs de runs existants
npx tsx scripts/hall-of-fame.ts

# Parité WASM op-par-op + run budget 500
npx tsx wasm-smoke.test.ts
```

## Reproductibilité

Chaque découverte du Hall of Fame est rejouable :

```bash
npx tsx scripts/hall-of-fame.ts 777888 500   # → redécouvre le record RC (7.3e-5)
```

Même graine ⇒ mêmes données, mêmes formules, mêmes métriques. Le ledger trace **quelle graine, à quelle itération**, chaque record a été trouvé.

## Structure

```
src/lib/spear/
  engine.ts       # AST, RNG seedé, simplify, NSGA-II, crossover/mutation, codegen
  benchmarks.ts   # 16 tâches : activations, KV-cache, physique, image/vidéo
  loop.ts         # grounded loop : UCB, warm-up, anti-stagnation, snapshots
  presets.ts      # labs mono-tâche (activation, KV-cache, CSV custom)
  wasm.ts         # compilateur AST → WebAssembly (encoder maison, zéro toolchain)
  math-utils.ts   # mse, linf, erf, gaussiennes
src/app/          # dashboard Next.js 16 + API routes (Postgres/Drizzle)
scripts/          # hall-of-fame
```

## Notes d'honnêteté

- Les records sont mesurés **sur le domaine des benchmarks**. `exp(cos(x))` approxime très bien le noyau gaussien sur [−3, 3] — et diverge complètement en dehors. C'est marqué ⚠️ dans le ledger.
- Les baselines « oracle » (loi exacte, attention future) bornent le score atteignable : battre la baseline *déployable* est le vrai signal.
- Les jalons de régression sont calibrés sur le **plancher de bruit** mesuré, pas sur des seuils arbitraires.
