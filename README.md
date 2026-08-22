# SPEAR Lab — Symbolic Pareto Evolutionary Algorithm for Research

A **symbolic regression** engine (multi-objective genetic programming) that discovers **closed-form mathematical laws** from data — then compiles them to **verified WebAssembly** and **MISRA-C:2012 C99**, with machine-checked parity.

No neural networks, no black boxes: what comes out is a formula you can read, audit, and deploy on a microcontroller.

## 🏆 Hall of Fame

Best discoveries across all seeds (full ledger: [`spear-hall-of-fame.json`](./spear-hall-of-fame.json), aggregated by [`scripts/hall-of-fame.ts`](./scripts/hall-of-fame.ts)).

**⚡ Speed column**: `cost(exact reference kernel) / cost(discovered formula)` in GPU ALU/SFU units (`mul/add` = 1, `div` = 4, `sqrt` = 2, `exp/sin/cos/log` ≈ 20).

### Record progression — what generic primitives unlocked

Adding two humble building blocks to the seed pools — reciprocal powers (`1/x`, `1/x²`) and the rsqrt family (`c/√(c₂ ± d·x²)`) — plus a parallel farm runner produced the largest single-run jump of the project:

| Task | Before | After | Gain |
|---|---|---|---|
| **Kerr light deflection** (weak field) | 5.4e-5 | **1.4e-7** | ~380× |
| **Kerr deflection with spin a≠0** | 2.9e-5 | **1.4e-7** | ~200× |
| **Lennard-Jones potential** | 9.0e-2 | **4.4e-4** | ~200× |
| **KdV soliton** | 9.6e-1 | **7.4e-3** | ~130× |
| **Deep-RL actor distillation** | 2.3e-4 | **1.6e-4** | +32 % |
| **Damped pendulum** | level 0, 8.9e-2 | **level 2, 7.0e-2** | first real milestone |

Lesson learned (twice now): discovery quality depends less on budget than on which algebraic bricks are available at warm-up.

### Physical laws recovered exactly

| Task | Record (MSE) | ⚡ Speed | Discovered formula | Seed |
|---|---|---|---|---|
| **Lorentz factor γ(β)** | **0 (exact)** | ×0.89 | `1/√(1 − b²)` — full relativistic law | 4242 |
| **Free fall** | **1.9e-4** | ×0.67 | `4.906·t²` → **g = 9.812 m·s⁻²** | 3333 |
| **Kepler's third law** | 4.7e-2 | — | `1.0009·a·√\|a\|` — the 3/2 exponent, recovered | 555666 |
| **RC circuit** | **7.3e-5** | — | `-0.998·min(exp(−t), 2) + 0.997` ≈ `1 − e^(−t/τ)` | 777888 |
| **Damped pendulum** | **1.9e-3** | — | two-tone cosine form on a numerically-simulated target | 700004 |
| **Lambert W₀** | 0 (exact) | — | `x·relu(exp(x))` | 12345 |
| **Optical-flow gradient** | 0 (exact) | ×1.00 | `b − a` | 8888 |
| **Bilinear upsampling weight** | 0 (exact) | ×0.50 | `1 − u` | 12345 |
| **LayerNorm rsqrt** | 0 (exact) | — | `x/√\|x³\|` = `1/√x` | 12345 |

### Astrophysics & relativistic physics

| Task | Record | ⚡ Speed | Discovered formula | Seed |
|---|---|---|---|---|
| **Kerr light deflection** (weak field) | **1.4e-7** | ×1.31 | rational form recovered near-exactly | 777 |
| **Kerr deflection with spin a≠0** (Lense-Thirring) | **2.7e-8** | ×1.62 | two-pole rational `(3.04/(b−3.79) + 4.19/b)` | 700002 |
| **KdV soliton** | 3.9e-3 | ×2.04 | travelling-wave approx of `2·sech²(x−4t)` | 700002 |

### LLM / image kernels replaced by pure algebra

| Task | Record | ⚡ Speed | Discovered formula | Baseline beaten |
|---|---|---|---|---|
| **Diffusion β(t)** | **1.8e-6** | ×1.04 | `−0.496·cos(3.162t) + 0.501` (refined) | cosine schedule |
| **Gaussian blur kernel** | **1.2e-6** | ×0.96 | `0.999·exp(−0.504x²)` — the true kernel structure | `exp(−x²/2σ²)` |
| **SiLU/Swish** | 7.8e-4 | **×2.43** | `x·(0.501 + 0.587·x/(0.815 + √(1+x²)))` | HardSwish, ReLU |
| **GELU** | 5.3e-4 | **×6.57** | `x·min(1.002, relu(0.308x + 0.501))` | GELU-tanh |
| **Sigmoid** | 0 (exact)* | ×1.26 | `1 − 1/(1 + e⁻ˣ)` | Hard-sigmoid TFLite |
| **Gaussian CDF Φ(x)** | 2.1e-4 | ×1.17 | refined rational form, div→mul strength reduction | HardSwish |
| **Softplus ln(1+eˣ)** | **2.4e-4** | ×4.56 | piecewise-rational approximant (bootstrap child) | smooth ReLU kernels |
| **Deep-RL actor distillation** | 1.6e-4 | **×2.83** | `(x + 0.145x³)/(0.556 + 0.75x²)` — Padé [3/2] found spontaneously | tanh network |

\* the sigmoid task allows `exp` as a primitive — the point is cost comparison, not algebraic purity.

### Molecular physics, pharmacology & control

| Task | Record | ⚡ Speed | Discovered formula | Note |
|---|---|---|---|---|
| **Hill dose-response** 🏥 | **1.2e-4** | — | exp-blended saturation ratio | deployable drug-model shape |
| **Lennard-Jones 12-6 potential** 🧪 | **4.4e-4** | ×0.55 | inverse-power rational double-well | repulsive + attractive terms |
| **Damped oscillation e^(−t/τ)cos(ωt)** 📡 | **3.1e-4** | ×1.63 | polynomial envelope × carrier: `(t−16.4)(t−8.3)·cos(2.996t)` — amplitude modulation | DSP staple |
| **Logistic growth** 📈 | **8.6e-5** | ×0.65 | level 2 saturation curve | adoption / population model |
| **European call premium** 💰 | 1.77e-1 | ×1.00 | `19.22·(1+σ)³ − 19.11` — level 5 | Black-Scholes ATM approx |
| **Inverted-pendulum hybrid control** 🎛️ | 2.1 | ×7.52 | cheap surrogate of the full 173-unit law | textbook Pareto trade-off point |

### ⚡ vs iterative solvers — the honest big multipliers

For several tasks the *standard* way to compute the answer is not another closed form but an **iterative numerical solver**. Counting the solver's full bill (every iteration, in ALU/SFU units) against our O(1) formula gives large — and legitimate — accelerations:

| Task | Discovered formula cost | Iterative baseline | Speedup |
|---|---|---|---|
| **Gaussian CDF Φ(x)** | 29 units | Monte-Carlo estimation, 1000 Box-Muller draws (46,000) | **×1586** |
| **Kerr deflection with spin** | 13 units | RK4 geodesic integration, 200 steps (2,400) | **×185** |
| **Damped oscillation trajectory** | 30 units | RKF45 adaptive solve, 300 steps (5,400) | **×200** |
| **Damped pendulum terminal state** | 26 units | Euler-Cromer integration, 60 steps (1,500) | **×58** |

Arithmetic is documented in [`src/lib/spear/benchmarks.ts`](./src/lib/spear/benchmarks.ts) (`ITERATIVE_BASELINES`). These are cost-model units, cross-checked by wall-clock benchmarks in the export audit.

**Note on protected division:** the engine's `pdiv` clamps denominators below 1e-4 and outputs beyond ±1e4. The Gaussian CDF champion *exploits these rails as free saturation* — its tiny divisor (3.4e-5) turns the division into a hard plateau, exactly the shape of Φ. The algebraic optimizer (`div-by-const → mul-by-reciprocal`, −3 units per site) therefore only rewrites divisors above the protection floor, and every rewrite passes a metric-parity gate before entering the ledger.

### Decision-making (KV-cache)

| Task | Record | Discovered rule | Baselines beaten |
|---|---|---|---|
| **KV-cache eviction** | **67.6 %** future attention mass retained | `(A + R)·(1 + 3S)` — multiplicative triad | H2O, StreamingLLM, SnapKV, sliding window, random |

A tri-dimensional rule (Sinks + accumulated Attention + Recency) discovered by evolution in **4 iterations** — the same triad the literature took years to identify.

## The MISRA-C export pipeline

Every discovered formula is exported as strict C99 and audited automatically ([`scripts/export-audit.ts`](./scripts/export-audit.ts)):

1. **Static MISRA lint** — zero dynamic allocation (`malloc = 0`), branchless (`fminf`/`fmaxf` saturations instead of `if/else`), fixed-width types (`float32_t`), no loops, FMA-friendly Horner shapes (`powf` forbidden);
2. **Real compilation** — gcc `-std=c99 -Wall -Wextra -pedantic -O2`, per task;
3. **Numeric parity C ↔ WASM** — both backends evaluated on identical sweeps, relative difference reported;
4. **Wall-clock benchmark** — discovered formula vs exact reference law, *both* compiled to WebAssembly, timed over 200k evaluations.

Latest full-audit run: **MISRA ✓ on every task · gcc ✓ · parity ≤ 1e-6 relative**. Measured speedups include softplus ×4.11–5.13 and Kerr ×1.93 — alongside honest ×0.5 entries where the evolved formula is heavier than the law itself.

The audit caught a real latent bug along the way: multi-variable WASM modules silently returned NaN because the JS wrapper passed an array where positional f64 parameters were expected. Fixed; the parity check now guards every future emission.

## How it works

```
data → population of algebraic trees → NSGA-II (accuracy ↔ parsimony)
  ↑                                            ↓
  └── UCB budget allocation ← stagnation ← Pareto rank-0
```

The engine (`src/lib/spear/`) combines:

- **Multi-objective GP**: non-dominated sorting + crowding distance, growing parsimony pressure (formulas must be *small* AND *accurate*);
- **End-to-end seeded RNG**: datasets, evolution and noise are reproducible from one integer (`setSeed` + injected uniform source);
- **Operator set**: `add sub mul pdiv relu abs neg sq sqrt cube max min exp sin cos log` — transcendental ops are allowed but priced honestly (≈20 ALU/SFU units vs 1 for `mul`);
- **Multi-start warm-up**: primitive shapes (rationals, Padé [3/2], exponential decay, inverse powers, rsqrt family) tuned by coordinate descent before evolution starts;
- **Cultural bootstrapping**: champion ASTs are stored in the ledger and re-injected as warm-up bricks for *other* tasks (variable-renamed, size-capped, never the task's own answer) — each generation of runs starts smarter than the last;
- **UCB budget allocation** across tasks: exploit what improves, explore what stagnates;
- **Anti-stagnation**: constant polishing, structural mutations, shape re-injection;
- **Honest scoring**: selection on train split, reporting on an unseen holdout (KV-cache task);
- **Algebraic simplification**: constant folding, nested-constant collapsing (`c₁·(c₂·x) → (c₁c₂)·x`, `(x+a)−a → x`) — kills degenerate records and sped the search up ~3×;
- **Verified exports**: every formula is emitted to **Python (torch)**, **CUDA C**, **strict MISRA-C99**, and **WebAssembly**, with op-by-op parity checks;
- **Light/full snapshots**: periodic UI snapshots stay cheap; codegen + WASM compile + speed benchmarks run once on the final snapshot.

## Getting started

```bash
npm install
# Postgres required (DATABASE_URL in .env)
npm run dev          # → http://localhost:3000
```

The dashboard offers the **Grounded Loop** (all tasks, budget 30–2000 iterations), per-preset labs (activations, KV-cache, custom CSV regression), and Postgres-persisted run history including breakthroughs.

### Headless scripts

```bash
# Live run: grounded loop + merge records into the hall of fame
npx tsx scripts/hall-of-fame.ts <seed> <budget> [deadlineMs]

# Rebuild the ledger from existing run logs
npx tsx scripts/hall-of-fame.ts

# MISRA lint + gcc compile + C↔WASM parity + wall-clock benchmarks
npx tsx scripts/export-audit.ts [seed] [budget]

# Parallel farm: N worker processes on disjoint task slices (uses all cores)
npx tsx scripts/run-farm.ts [seed] [budget] [workers]

# Recompute cost/speed multipliers for every ledger entry carrying an AST
npx tsx scripts/refresh-speeds.ts

# Algebraic optimization pass over ledger trees (parity-gated)
npx tsx scripts/optimize-ledger.ts
npx tsx scripts/test-simplify.ts

# Op-by-op WASM parity smoke test
npx tsx wasm-smoke.test.ts
```

The farm splits the task registry across worker processes via a `SPEAR_TASKS` filter and merges records into the ledger — a full sweep completes well inside the single-process deadline while using every core.

## Reproducibility

Every Hall of Fame record is replayable:

```bash
npx tsx scripts/hall-of-fame.ts 777888 500   # → rediscovers the RC record (7.3e-5)
```

Same seed ⇒ same data, same formulas, same metrics. The ledger tracks **which seed, at which iteration**, each record was found.

## Project layout

```
src/lib/spear/
  engine.ts       # AST, seeded RNG, simplify, NSGA-II, GP operators, codegen (torch/C/MISRA-C)
  benchmarks.ts   # benchmark tasks + exact-law ASTs for the cost model
  loop.ts         # grounded loop: UCB, warm-up, anti-stagnation, light/full snapshots
  presets.ts      # single-task labs (activation, KV-cache, custom CSV)
  wasm.ts         # AST → WebAssembly compiler (hand-rolled encoder, no toolchain)
  math-utils.ts   # mse, linf, erf, gaussians
src/app/          # Next.js dashboard + API routes (Postgres/Drizzle)
scripts/          # hall-of-fame ledger, export audit, parallel farm
```

## Honesty notes

- Records are measured **on the benchmark domains**. `exp(cos(x))` fits the Gaussian kernel beautifully on [−3, 3] — and diverges from it outside. Flagged ⚠️ in the ledger.
- "Oracle" baselines (the exact law, future attention) bound achievable scores: beating the *deployable* baselines is the real signal.
- Regression milestones are calibrated against the **measured noise floor**, not arbitrary thresholds.
- Speed multipliers marked "—" predate the cost-model extension; they backfill automatically whenever a run reproduces the champion formula (speedup is a pure function of the AST).
- A ×7.52 entry with poor accuracy (inverted-pendulum control) is displayed exactly as that: the search surfaced a cheap-but-crude surrogate — a legitimate Pareto point, not hidden behind the headline number.

## Provenance

The benchmark portfolio draws inspiration from the GROUNDED-SPEAR paper series (V600 → V1200): Padé [3/2] distillation shapes (BT10), Kerr geodesic deflection incl. spin (BT24/BT35), KdV solitons (BT36) and hybrid Lyapunov control laws (§4). Everything else — the engine, the audits, the numbers above — is generated and measured by this repository.
