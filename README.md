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
| **Damped pendulum** | **6.8e-4** | ×33 vs Euler-Cromer | clamped-carrier beat: `max(cos t, 0.68)·cos(2.787t)` on simulated target | 930101 |
| **Lambert W₀** | 0 (exact) | — | `x·relu(exp(x))` | 12345 |
| **Optical-flow gradient** | 0 (exact) | ×1.00 | `b − a` | 8888 |
| **Bilinear upsampling weight** | 0 (exact) | ×0.50 | `1 − u` | 12345 |
| **LayerNorm rsqrt** | 0 (exact) | — | `x/√\|x³\|` = `1/√x` | 12345 |

### Astrophysics & relativistic physics

| Task | Record | ⚡ Speed | Discovered formula | Seed |
|---|---|---|---|---|
| **Kerr light deflection** (weak field) | **1.4e-7** | ×1.31 | rational form recovered near-exactly | 777 |
| **Kerr deflection with spin a≠0** (Lense-Thirring) | **1.3e-8** | ×1.40 | rational double-pole in `(b−0.196)` — accuracy-first form | 930202 |
| **KdV soliton** | **1.7e-3** | ×7.57 | `max(√|6.66−max(3.17,x)|, 0.60)` — cheap piecewise crest | 930202 |

### LLM / image kernels replaced by pure algebra

| Task | Record | ⚡ Speed | Discovered formula | Baseline beaten |
|---|---|---|---|---|
| **Diffusion β(t)** | **1.8e-6** | ×1.04 | `−0.496·cos(3.162t) + 0.501` (refined) | cosine schedule |
| **Gaussian blur kernel** | **5.4e-34 — EXACT** | — | `√|exp(−x²)|` = e^(−x²/2): the law recovered to machine precision | `exp(−x²/2σ²)` |
| **SiLU/Swish** | 7.8e-4 | **×2.43** | `x·(0.501 + 0.587·x/(0.815 + √(1+x²)))` | HardSwish, ReLU |
| **GELU** | 5.3e-4 | **×6.57** | `x·min(1.002, relu(0.308x + 0.501))` | GELU-tanh |
| **Sigmoid** | 0 (exact)* | ×1.26 | `1 − 1/(1 + e⁻ˣ)` | Hard-sigmoid TFLite |
| **Gaussian CDF Φ(x)** | 2.1e-4 | ×1.17 | refined rational form, div→mul strength reduction | HardSwish |
| **Softplus ln(1+eˣ)** | **2.3e-4** | ×4.56 | piecewise-rational approximant (bootstrap child) | smooth ReLU kernels |
| **Deep-RL actor distillation** | 1.6e-4 | **×2.83** | `(x + 0.145x³)/(0.556 + 0.75x²)` — Padé [3/2] found spontaneously | tanh network |

\* the sigmoid task allows `exp` as a primitive — the point is cost comparison, not algebraic purity.

### Molecular physics, pharmacology & control

| Task | Record | ⚡ Speed | Discovered formula | Note |
|---|---|---|---|---|
| **Hill dose-response** 🏥 | **1.16e-4** | — | min-cascade saturation on c² | deployable drug-model shape |
| **Lennard-Jones 12-6 potential** 🧪 | **4.4e-4** | ×0.55 | inverse-power rational double-well | repulsive + attractive terms |
| **Damped oscillation e^(−t/τ)cos(ωt)** 📡 | **3.1e-4** | ×1.63 | polynomial envelope × carrier: `(t−16.4)(t−8.3)·cos(2.996t)` — amplitude modulation | DSP staple |
| **Logistic growth** 📈 | **7.9e-5** | ×0.65 | level 2 saturation curve | adoption / population model |
| **European call premium** 💰 | 1.77e-1 | ×1.00 | `19.22·(1+σ)³ − 19.11` — level 5 | Black-Scholes ATM approx |
| **Inverted-pendulum hybrid control** 🎛️ | **2.04** | ×7.52 | cheap surrogate of the full 173-unit law — `d·cos(5.05·th/d)` form | textbook Pareto trade-off point |

### From the SPEAR CODEX (3DSPEAR) — new hunting grounds

Three Codex entries became reproducible benchmark tasks, with the iterative solvers they replace priced as honest baselines:

| Task | Codex source | Replaces | Record | vs solver |
|---|---|---|---|---|
| **eigen3_sym** — λmax of symmetric 3×3 from its invariants | BT29 Cardano trisection | Jacobi/QR sweeps (~200 units) | 3.6e-2, L1, 13 units | **×15 vs Jacobi · 4 balayages** |
| **ik_reach** — 2-link elbow angle from reach & lengths | BT33 analytic IK | Newton-DLS chains (~320 units) | 2.2e-3, L2, 16 units | **×20 vs Newton-DLS · 8 it.** |
| **idm_following** — Intelligent-Driver-Model acceleration | city.ts traffic phase B | hand-tuned law only | open challenge (L0) | — |

Both eigen and IK truths require `acos`, a transcendental the engine deliberately does not serve — the GP must rebuild the triple-angle / inverse-cosine shape from algebra alone, exactly the Padé-style discipline the Codex doctrine prescribes.

### LLM inference kernels: can SR find "a formula for matmul"?

Short answer explored empirically: **the matmul itself already IS the minimal closed form** — a generic dot product has tensor rank n, so `4 mul + 3 add` is provably unbeatable per output lane. No solver beats it because there is nothing iterative to replace. What *can* be attacked is everything glued around the GEMV:

| Task | What it tests | Record | Status |
|---|---|---|---|
| **gemv4** — decode-lane dot product | Can the engine recover the provably-minimal kernel (7 units, MSE → noise floor)? | 4.1e-4, L2, 12 units | open — optimality test, not a speed play |
| **rope_rot** — RoPE lane rotation, paid every token of every head | Algebraic replacement for cos+sin (43 units) vs CORDIC micro-rotations (64) | 6.7e-1, L0, 23 units (`−y·sinθ` term captured) | open — hardest unserved-transcendental task yet |

The real speed story stays where it already lives in this ledger: activation kernels (SiLU/GELU/softplus), softmax-free eviction policy (KV-cache), and quantized fast variants of both.

### ⚡ vs iterative solvers — the honest big multipliers

For several tasks the *standard* way to compute the answer is not another closed form but an **iterative numerical solver**. Counting the solver's full bill (every iteration, in ALU/SFU units) against our O(1) formula gives large — and legitimate — accelerations:

| Task | Discovered formula cost | Iterative baseline | Speedup |
|---|---|---|---|
| **Gaussian CDF Φ(x)** | 29 units | Monte-Carlo estimation, 1000 Box-Muller draws (46,000) | **×1586** |
| **Kerr deflection with spin** | 15 units | RK4 geodesic integration, 200 steps (2,400) | **×160** |
| **Damped oscillation trajectory** | 30 units | RKF45 adaptive solve, 300 steps (5,400) | **×200** |
| **Damped pendulum terminal state** | 46 units | Euler-Cromer integration, 60 steps (1,500) | **×33** |

Arithmetic is documented in [`src/lib/spear/benchmarks.ts`](./src/lib/spear/benchmarks.ts) (`ITERATIVE_BASELINES`). These are cost-model units, cross-checked by wall-clock benchmarks in the export audit.

### 🎚️ Two operating points per law: `precise` and `fast`

One formula per task hides the accuracy/latency trade-off that real deployments live by: a rendering loop or an LLM hot path happily accepts 1e-4 relative error if it shaves half the ALU bill; a physics integration does not. So the ledger keeps **two validated forms per task**: the *precise* champion (lowest error ever found) and the *fast* variant (cheapest form that still passes the engine's level-2 validation gate). **19 of 27 tasks** have a genuinely cheaper second form; the rest have champions already at minimal cost.

Biggest fast-slot wins (cost = precise → fast, in ALU/SFU units):

| Task | Precise | Fast | Cost cut |
|---|---|---|---|
| **Lambert W₀** | exact, 22 units | 1.1e-2, 1 unit | **×22** |
| **Gaussian blur kernel** | exact e^(−x²/2), 24 units | 5.7e-3, 1 unit | **×24** |
| **Hill dose-response** | 1.16e-4, 9 units | 1.8e-3, 1 unit | **×9** |
| **Gaussian CDF Φ(x)** | 2.1e-4, 29 units | 7.6e-4, 6 units | ×4.8 |
| **Softplus** | 2.3e-4, 9 units | 3.2e-3, 2 units | ×4.5 |
| **Sigmoid** | exact, 27 units | 7.2e-4, 7 units — `x/(1+|x|)` | ×3.9 |
| **Kerr deflection w/ spin** | 1.32e-8, 15 units (×160 vs RK4) | `10.49/(s+b)`, 3.6e-5, 5 units | **×480 vs RK4** |

Displaced champions are never lost: when a more accurate form takes over, the old one is demoted to the fast slot, [`scripts/backfill-fast.ts`](./scripts/backfill-fast.ts) resurrects historical forms from git archaeology, and every ledger refresh prunes any "fast" variant that is no longer actually cheaper than its champion.

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

# Resurrect cheap validated forms from git history into the `fast` slot
npx tsx scripts/backfill-fast.ts

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
