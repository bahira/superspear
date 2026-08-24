# SPEAR v1.0.0 — evolution-discovered math kernels, parity-audited

First tagged release of SPEAR Lab: a symbolic-regression engine (multi-objective genetic programming) that discovers closed-form mathematical laws from data, then compiles them to verified WebAssembly and MISRA-C:2012 C99 with machine-checked parity. No neural networks, no black boxes — what comes out is a formula you can read, audit, and deploy on a microcontroller.

## Highlights

- **48-task benchmark registry** spanning NN activations, LLM inference glue, physics, finance, shaders, and control.
- **14 exact solutions** — tasks solved to machine precision (MSE ≤ 1e-8), including 7 records at *exactly* 0: Lorentz factor γ(β), Lambert W₀, RoPE rotation, optical-flow gradient, bilinear upsampling weight, LayerNorm rsqrt, atan unit.
- **18 kernels faster than their exact reference kernel**, priced in GPU ALU/SFU units (`mul/add`=1, `div`=4, `exp/sin/cos/log`≈20).
- **Iterative solvers replaced by O(1) formulas**: up to **×1840** vs Monte-Carlo estimation for the Gaussian CDF Φ(x); documented replacements span ×6–×1840 (RK4 geodesics, RKF45, Euler-Cromer, Jacobi sweeps, Newton-DLS IK).
- **Edge CPU measurements (WASM, 200k elements)**: SiLU **×2.12** (864 → 409 ns/el), GELU ×2.32 vs industry tanh approximant. Slower variants are reported as measured, not hidden.
- **Real LLM validation**: the 5-unit KV-cache eviction rule `4.5·S + A + R` validated on genuine distilgpt2 attention traces — **80.31 %** future attention mass retained vs **80.20 %** for H2O and 80.07 % for StreamingLLM.

## What's included

- **Engine** (`src/lib/spear/`): NSGA-II GP, seeded end-to-end RNG, cultural bootstrapping, UCB budget allocation, AST → WASM / MISRA-C / CUDA / PyTorch codegen.
- **48-task registry** (`benchmarks.ts`) with exact-law ASTs for cost-model accounting.
- **npm package** [`packages/spear-kernels`](./packages/spear-kernels): **47 production kernels** as ESM + CJS + TypeScript definitions, with generated CUDA-C and PyTorch source strings per kernel.
- **3DSPEAR integration kit** ([`integrations/3dspear-kit`](./integrations/3dspear-kit)): shader-ready GLSL of the exp-free gaussian menu and friends.
- **Discovery API**: `POST /api/spear/discover` — send a target formula, get back a Pareto-audited replacement plus WASM/MISRA-C/torch exports.
- **PAPER.md** draft covering methodology, optimality tests, and honest-negative results.

## Engineering notes

Two critical bugs were found *by* the audit pipeline during this cycle:

1. **`simplify()` NaN-collapse** on `(c·x)·Y`-shaped subtrees silently ate valid expressions during search (surfaced while cracking the unserved-transcendental tasks). Fixed and guarded by `scripts/test-simplify.ts`.
2. **Poisoned Bessel reference series** — the bessel_j0/bessel_j1 ground truths used a wrong reference expansion. After rebuilding the references: bessel_j1 record dropped ×108,000, bessel_j0 ×60. Refusing to accept walls caught it.

Also in this release:

- **Concurrency lockfile**: `run-farm.ts` takes a `.farm-lock` single-writer guard — two simultaneous farms previously overwrote each other's records last-writer-wins.
- **`atan` primitive served across all backends** (JS/WASM/C/torch, cost 20). With acos→atan identity scaffolds, both remaining transcendental-wall tasks fell to machine precision: ik_reach 1.1e-13, eigen3_sym 3.3e-11 (full Cardano scaffold).

## Install & quickstart

```bash
git clone <repo-url> && cd superspear
npm ci                 # Postgres required (DATABASE_URL in .env)

# Op-by-op WASM parity smoke test
npx tsx scripts/test-wasm-parity.ts

# Dashboard + discovery API
npm run dev            # → http://localhost:3000
```

Discover a replacement kernel:

```bash
curl -X POST http://localhost:3000/api/spear/discover \
  -H "Content-Type: application/json" \
  -d '{
    "formula": "sin(x) * exp(-x / 3)",
    "lo": -5,
    "hi": 5,
    "rows": 400,
    "populationSize": 120,
    "generations": 60,
    "transcendentals": true
  }'
```

Use the shipped kernels without any build step:

```bash
cd packages/spear-kernels && npm publish   # maintainers only
```

```js
import { kernels } from "spear-kernels";
const silu = kernels.silu.fast.eval; // JS closure; .c/.py carry CUDA/torch sources
```

## Honesty notes

Records are measured on benchmark domains and flagged when they diverge outside them; every ledger entry carries its seed and iteration; same seed ⇒ same data, same formulas, same metrics. Speed multipliers marked as cost-model units are cross-checked by wall-clock benchmarks in `scripts/export-audit.ts`.

MIT licensed — free to use, modify, and ship; the discovered formulas are yours too.
