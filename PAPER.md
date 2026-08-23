# SPEAR: Symbolic Pareto Evolutionary Algorithm for Research —
## Evolution-Discovered Kernels with Triple-Backend Parity Audits

*Working draft v1 — numbers pulled live from `spear-hall-of-fame.json`; sections marked [MEASURE] need the corresponding run before submission.*

---

## Abstract

We present SPEAR, a symbolic regression engine that discovers closed-form
mathematical laws from data via multi-objective genetic programming, then
compiles every discovery to three execution backends (JavaScript/WebAssembly,
strict MISRA-C:2012 C99, and PyTorch) under a machine-checked parity audit.
Across a 48-task benchmark registry spanning neural-network activations,
shader primitives, special functions, physics laws and control policies,
SPEAR produced 9 exact solutions (machine-precision MSE), 18 kernels faster
than their exact references in an ALU/SFU cost model, and 7 replacements of
iterative numerical solvers with speedups from ×6 to ×1840. We contribute a
taxonomy of four difficulty regimes for symbolic search, a scaffold-seeding
methodology validated on adversarial optimality tests, and an honest failure
case where an expert hand-fit (Narkowicz ACES) resists evolution at any
tested depth.

## 1. Introduction

Production software re-implements the same small mathematical kernels
millions of times per second: activation functions, interpolation curves,
color transfers, wave evaluators, normalization factors. When these kernels
rely on transcendental functions (`exp`, `erf`, `acos`), cost explodes on
transcendental-poor hardware. Hand-derived algebraic replacements exist for a
lucky few (fast inverse sqrt, Padé approximants); deriving new ones is slow
expert work. SPEAR automates this derivation *and* the verification burden
that normally blocks adoption: every discovered formula ships with
bit-faithful JS/WASM/C implementations proven equivalent to the evaluator.

## 2. Methods

### 2.1 Search
Multi-objective NSGA-II over expression trees; objectives are accuracy
(MSE on train split) and formula cost counted in ALU/SFU units
(`mul/add = 1`, `div = 4`, transcendental = 20). Milestones are calibrated
against the measured noise floor of each dataset rather than absolute
thresholds.

### 2.2 Verification pipeline
Every champion passes: (a) static MISRA-C:2012 lint, (b) gcc compilation and
execution, (c) numeric parity C↔WASM↔JS within 1e-4 relative, enforced by a
protected-division codegen that reproduces the evaluator's denominator floor
(±1e-4) and output clamp (±1e4) bit-faithfully. A permanent CI job replays a
parity sweep across all ledger champions on every push.

### 2.3 Deployment shape
Each task stores two validated forms: the **precise** champion (lowest error)
and a **fast** variant (cheapest form passing level-2 validation), giving
deployers an explicit accuracy/latency operating point.

## 3. The four difficulty regimes

Empirically, tasks separate into regimes with distinct attack strategies:

| Regime | Definition | Example | Outcome |
|---|---|---|---|
| R1 — served law | exact form expressible with served primitives | RoPE rotation, EMA factor | exact recovery (L5) |
| R2 — classic approximant | a known human form exists | atan → Padé [3/2] | rediscovery at 500 iterations |
| R3 — unintuitive substitution | cheaper algebra exists, non-obvious | cosh → `√x²·cos(x²)`, blackbody rational | ×1.4–1.8 faster at L4–L5 |
| R4 — hidden human trick | optimal form relies on an insight | Huber `max(x²/2, |x|−½)`, smoothstep polynomial | passes only after scaffold seeding |

Case study (R3→R4 bridge): seeding the *shape* of Huber's max-trick without
its constants converted a stuck L2 candidate into an L5 solution (MSE
3.0e-7) within two subsequent 500-iteration runs. Counter-case: Narkowicz's
ACES hand-fit survived 8,000 iterations against a seeded rational scaffold —
expert hand-fits remain defensible.

## 4. Results

### 4.1 Exact solutions (9)
RoPE rotation, gaussian kernel (e^(−x²/2) recovered exactly), atan unit
domain (Padé [3/2] rediscovered), EMA smoothing factor, sigmoid, Lorentz
factor, Lambert W₀, 2-link IK elbow (1.1e-13), symmetric 3×3 λmax (3.3e-11,
full Cardano trisection recovered through the atan identity).

### 4.2 Faster than the reference (selection)
logsumexp2 ×8.57, KdV soliton ×7.57, GELU ×6.57, inverted-pendulum hybrid
×3.76, IDM car-following ×3.33, SiLU ×2.43. Wall-clock cross-checks on WASM
confirm direction and rough magnitude (e.g., KdV ×1.96 measured).

### 4.3 Iterative solver replacement (selection)
Gaussian CDF vs Monte-Carlo estimation ×1840; damped oscillation trajectory
vs RKF45 ×200; Kerr deflection vs RK4 geodesic integration ×185–133;
damped pendulum vs Euler-Cromer ×33; 2-link IK vs Newton-DLS chains ×20.

### 4.4 Edge measurements
On a CPU-only machine, WASM-compiled discovered forms replace production
activation kernels at ×2.12 (SiLU) and ×2.32 (GELU) per element; a full FFN
block (d=2048, ffn=5632) runs 10.2% faster end-to-end despite naive
per-element host↔wasm invocation.

## 5. Limitations

1. RESOLVED: KV-cache eviction validated on real distilgpt2 attention traces
   (72 layer-prompt samples): the discovered 5-unit policy retains 80.31% of
   future attention mass vs 80.20% for H2O and 80.07% for StreamingLLM
   (oracle upper bound: 90.85%). The margin is narrow but positive, achieved
   with 3 features and no attention-score sorting.
2. MEASURED — honest negative: swapping distilgpt2's fused GELU for the
   discovered form RAISES perplexity by 0.56 and LOWERS throughput by 9.2%
   on CPU. Native fused kernels beat multi-op algebraics inside frameworks;
   the cost-model gains transfer only where transcendentals are truly
   expensive (raw WASM: ×2.12 measured; MCUs). Frameworks are the wrong
   battlefield — this is a deployment-boundary result, not a search failure.
3. Bessel-family tasks (J₀/J₁) resist current depths; series scaffolds
   mutate away before compression.
4. The cost model counts ALU/SFU units; wall-clock gains on SFU-rich GPUs
   will be smaller than modeled.
5. One expert hand-fit resisted all budgets tried; we claim complementarity
   with, not superiority over, expert derivation.

## 6. Reproducibility

Every record carries its seed, iteration count, serialized AST, cost
breakdown, and both operating points. `npm ci && npx tsx scripts/test-wasm-parity.ts`
replays the full parity audit; `npx tsx scripts/run-farm.ts <seed> <budget> <workers>`
replays discovery. MIT licensed.
