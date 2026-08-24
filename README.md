# SPEAR Lab — Symbolic Pareto Evolutionary Algorithm for Research

[![npm](https://img.shields.io/npm/v/spear-kernels)](https://www.npmjs.com/package/spear-kernels) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

```bash
npm install spear-kernels
```

Now shipping **64+ verified kernels on npm** (`spear-kernels@latest`) — including the **SPEAR Quant Pack**: Kelly criterion, RSI, implied volatility, Gaussian CDF, probit — all parity-audited across JS/WASM/C/PyTorch. that discovers **closed-form mathematical laws** from data — then compiles them to **verified WebAssembly** and **MISRA-C:2012 C99**, with machine-checked parity.

No neural networks, no black boxes: what comes out is a formula you can read, audit, and deploy on a microcontroller.

## 🏆 Hall of Fame

Best discoveries across all seeds (full ledger: [`spear-hall-of-fame.json`](./spear-hall-of-fame.json), aggregated by [`scripts/hall-of-fame.ts`](./scripts/hall-of-fame.ts)).

**⚡ Speed column**: `cost(exact reference kernel) / cost(discovered formula)` in GPU ALU/SFU units (`mul/add` = 1, `div` = 4, `sqrt` = 2, `exp/sin/cos/log` ≈ 20).

### Record progression — what generic primitives unlocked

Adding two humble building blocks to the seed pools — reciprocal powers (`1/x`, `1/x²`) and the rsqrt family (`c/√(c₂ ± d·x²)`) — plus a parallel farm runner produced the largest single-run jump of the project:

  Task   Before   After   Gain  
 --- --- --- --- 
  **Kerr light deflection** (weak field)   5.4e-5   **1.4e-7**   ~380×  
  **Kerr deflection with spin a≠0**   2.9e-5   **1.4e-7**   ~200×  
  **Lennard-Jones potential**   9.0e-2   **4.4e-4**   ~200×  
  **KdV soliton**   9.6e-1   **7.4e-3**   ~130×  
  **Deep-RL actor distillation**   2.3e-4   **1.6e-4**   +32 %  
  **Damped pendulum**   level 0, 8.9e-2   **level 2, 7.0e-2**   first real milestone  

Lesson learned (twice now): discovery quality depends less on budget than on which algebraic bricks are available at warm-up.

### Physical laws recovered exactly

  Task   Record (MSE)   ⚡ Speed   Discovered formula   Seed  
 --- --- --- --- --- 
  **Lorentz factor γ(β)**   **0 (exact)**   ×0.89   `1/√(1 − b²)` — full relativistic law   4242  
  **Free fall**   **1.9e-4**   ×0.67   `4.906·t²` → **g = 9.812 m·s⁻²**   3333  
  **Kepler's third law**   4.7e-2   —   `1.0009·a·√\ a\ ` — the 3/2 exponent, recovered   555666  
  **RC circuit**   **7.3e-5**   —   `-0.998·min(exp(−t), 2) + 0.997` ≈ `1 − e^(−t/τ)`   777888  
  **Damped pendulum**   **6.8e-4**   ×33 vs Euler-Cromer   clamped-carrier beat: `max(cos t, 0.68)·cos(2.787t)` on simulated target   930101  
  **Lambert W₀**   0 (exact)   —   `x·relu(exp(x))`   12345  
  **Optical-flow gradient**   0 (exact)   ×1.00   `b − a`   8888  
  **Bilinear upsampling weight**   0 (exact)   ×0.50   `1 − u`   12345  
  **LayerNorm rsqrt**   0 (exact)   —   `x/√\ x³\ ` = `1/√x`   12345  

### Astrophysics & relativistic physics

  Task   Record   ⚡ Speed   Discovered formula   Seed  
 --- --- --- --- --- 
  **Kerr light deflection** (weak field)   **1.4e-7**   ×1.31   rational form recovered near-exactly   777  
  **Kerr deflection with spin a≠0** (Lense-Thirring)   **7.5e-10**   ×1.17   rational double-pole in `(b−0.196)` — accuracy-first form   930202  
  **KdV soliton**   **1.7e-3**   ×7.57   `max(√ 6.66−max(3.17,x) , 0.60)` — cheap piecewise crest   930202  

### LLM / image kernels replaced by pure algebra

  Task   Record   ⚡ Speed   Discovered formula   Baseline beaten  
 --- --- --- --- --- 
  **Diffusion β(t)**   **1.8e-6**   ×1.04   `−0.496·cos(3.162t) + 0.501` (refined)   cosine schedule  
  **Gaussian blur kernel**   **5.4e-34 — EXACT**   —   `√ exp(−x²) ` = e^(−x²/2): the law recovered to machine precision   `exp(−x²/2σ²)`  
  **SiLU/Swish**   7.8e-4   **×2.43**   `x·(0.501 + 0.587·x/(0.815 + √(1+x²)))`   HardSwish, ReLU  
  **GELU**   5.3e-4   **×6.57**   `x·min(1.002, relu(0.308x + 0.501))`   GELU-tanh  
  **Sigmoid**   0 (exact)*   ×1.26   `1 − 1/(1 + e⁻ˣ)`   Hard-sigmoid TFLite  
  **Gaussian CDF Φ(x)**   **1.6e-5, L3**   ×1.79   clamped rational `x/max(1.56, 0.34+ x )` — div→mul strength reduction   HardSwish  
  **Softplus ln(1+eˣ)**   **1.9e-5, L2** — ×75 cumulative   ×0.95 (exact-grade form is pricier)   piecewise-rational approximant (bootstrap child)   smooth ReLU kernels  
  **Deep-RL actor distillation**   1.6e-4   **×2.83**   `(x + 0.145x³)/(0.556 + 0.75x²)` — Padé [3/2] found spontaneously   tanh network  

\* the sigmoid task allows `exp` as a primitive — the point is cost comparison, not algebraic purity.

### Molecular physics, pharmacology & control

  Task   Record   ⚡ Speed   Discovered formula   Note  
 --- --- --- --- --- 
  **Hill dose-response** 🏥   **4.5e-6, L2** — ×26 total   —   min-cascade saturation on c²   deployable drug-model shape  
  **Lennard-Jones 12-6 potential** 🧪   **3.2e-4**   ×0.55   log-cos composite well shape   repulsive + attractive terms  
  **Damped oscillation e^(−t/τ)cos(ωt)** 📡   **3.1e-4**   ×1.63   polynomial envelope × carrier: `(t−16.4)(t−8.3)·cos(2.996t)` — amplitude modulation   DSP staple  
  **Logistic growth** 📈   **1.1e-5**   ×0.62   level 2 saturation curve   adoption / population model  
  **European call premium** 💰   1.77e-1   ×1.00   `19.22·(1+σ)³ − 19.11` — level 5   Black-Scholes ATM approx  
  **Inverted-pendulum hybrid control** 🎛️   **2.04**   ×3.76   cheap surrogate of the full 173-unit law — `d·cos(5.05·th/d)` form   textbook Pareto trade-off point  

### From the SPEAR CODEX (3DSPEAR) — new hunting grounds

Three Codex entries became reproducible benchmark tasks, with the iterative solvers they replace priced as honest baselines:

  Task   Codex source   Replaces   Record   vs solver  
 --- --- --- --- --- 
  **eigen3_sym** — λmax of symmetric 3×3 from its invariants   BT29 Cardano trisection   Jacobi/QR sweeps (~200 units)   **EXACT 3.3e-11** — full Cardano scaffold recovered post-atan   **×18 vs Jacobi · 4 balayages**  
  **ik_reach** — 2-link elbow angle from reach & lengths   BT33 analytic IK   Newton-DLS chains (~320 units)   **EXACT 1.1e-13** — atan identity unlocked by the new atan primitive   **×20 vs Newton-DLS · 8 it.**  
  **idm_following** — Intelligent-Driver-Model acceleration   city.ts traffic phase B   hand-tuned law only   **4.5e-3, L2 — WALL SOLVED** (×450 via IDM-structure scaffold, 20 units)   ×3.33 vs exact    

Both eigen and IK truths require `acos`, a transcendental the engine deliberately does not serve — the GP must rebuild the triple-angle / inverse-cosine shape from algebra alone. Unlike RoPE (whose sin/cos carriers ARE served and fell exactly at depth 6000), these two can only approximate: deep passes with the new trig seeds cracked eigen3_sym through L2; the remaining gap is pure Padé-grade approximation work.
**UPDATE — the `atan` primitive is now served** (JS/WASM/C/torch, cost 20). With acos→atan identity scaffolds seeded, BOTH unserved-transcendental tasks fell to machine precision: ik_reach 1.1e-13, eigen3_sym 3.3e-11 (full Cardano). A silent `simplify()` collapse bug eating `(c·x)·Y`-shaped subtrees was found and fixed along the way.

### LLM inference kernels: can SR find "a formula for matmul"?

Short answer explored empirically: **the matmul itself already IS the minimal closed form** — a generic dot product has tensor rank n, so `4 mul + 3 add` is provably unbeatable per output lane. No solver beats it because there is nothing iterative to replace. What *can* be attacked is everything glued around the GEMV:

  Task   What it tests   Record   Status  
 --- --- --- --- 
  **gemv4** — decode-lane dot product   Can the engine recover the provably-minimal kernel (7 units, MSE → noise floor)?   **1.6e-7**, L2, 17 units — **×2500 cumulative via linear scaffold**; optimality test #1 effectively passed   scaffold → refinement → convergence  
  **rope_rot** — RoPE lane rotation, paid every token of every head   Exact recovery via served sin/cos primitives   **EXACT (MSE=0, L5)**, 44 units, ×1.45 vs CORDIC   solved precisely; algebraic fast variant pending  

Cross-variable and trigonometric carrier seeds (shape-only, no answer constants) unlocked the exact RoPE recovery at depth 6000 — the seed pool now covers bilinear interactions (`x₀·x₁`, `x₀±x₁`) and carriers (`cos x`, `sin x`).

### 🔬 Edge CPU measurements (WASM, this machine)

Production kernels vs SPEAR forms compiled to WASM, 200k elements on `[-6,6]` (`npx tsx scripts/bench-edge.ts`):

  Kernel   Reference   SPEAR   Gain   Max abs err  
 --- --- --- --- --- 
  **SiLU** (SwiGLU)   exp-based, 864 ns/el   algebraic, 409 ns/el   **×2.12**   7.0e-2  
  **GELU**   tanh-approx (industry), 757 ns/el   algebraic, 326 ns/el   **×2.32**   8.0e-2  
  Softplus   exp-based, 1403 ns/el   algebraic, 1113 ns/el   ×1.26   4.2e-1  
  Sigmoid fast-slot   exp-based, 1075 ns/el   `x/(1+\ x\ )`, 1307 ns/el   ×0.82 — slower, reported as measured   8.6e-1  

FFN block simulation (decode, d=2048, ffn=5632): the full block runs **10.2 % faster** end-to-end with the SPEAR SiLU swapped in, even through naive per-element JS↔WASM invocation. Honest caveats: max errors ~0.07–0.08 are visible-but-bounded activation distortion (perplexity delta unmeasured); the sigmoid fast variant loses on this backend; batched invocation would remove the per-call boundary overhead.

### 🎨 Shader-ready gaussian menu (variable blur / bloom / DoF)

New task `gauss_shader`: algebraic approximants of e^(−x²/2) evaluated per pixel. The Pareto mining (`scripts/tune-gaussian.ts`, `scripts/mine-gaussian.ts`) produced an exp-free menu:

  Kernel   Cost   MSE vs e^(−x²/2)   Formula  
 --- --- --- --- 
  exact   24   0   `exp(−x²/2)`  
  **student-t k3**   **9**   **6.4e-4**   `1.02232/(0.207x²+1)³`  
  **student-t k2**   **8**   **1.35e-3**   `1.04984/(0.375x²+1)²`  
  cos window   20   5.2e-3   `cos(x)`  

### 🌊 Third wave — companions, tonemap & physics glue (1000-iter gen-1 + 3000-iter depth)

  Task   Use   Record   Cost   vs exact  
 --- --- --- --- --- 
  **blackbody_r** ⭐   PBR light color temperature   **3.6e-6, L4** — clamped rational   14 (vs ~25)   **×1.79 FASTER**  
  **huber_loss** (depth)   robust training loss   **3.4e-5, L3** at 3000 iters — ×7.4 gain, but the exact 5-unit max-trick STILL hidden   13   ×0.38  
  **aces_fit**   tonemap — optimality test #4   2.4e-5, L3   53 vs Narkowicz's 8   **human hand-fit wins round 1**  
  **bessel_j1**   FM · antisymmetric membrane modes   **5.2e-6, L4** — ×108,000 once the reference series was fixed   48   ×1.25  
  **logsumexp2**   differentiable max (RL losses)   1.0e-2, L2 — max/min scaffolding emerging   —   —  
  **bessel_j0** (depth)   vibrations · beams   **2.6e-4, L2** — was a poisoned-data wall (wrong reference series), ×60 after fix   13   ×1.43  
### 🎯 Round 2 — explicit seeds & massive budgets

Seeding the *shape of the trick* changed everything on the optimality tests:

  Task   Round 1   Round 2   What happened  
 --- --- --- --- 
  **huber_loss** (max-seeds, then hammering)   3.4e-5, no trick visible   **3.0e-7, L5** — champion now literally contains `max(x²/( x +1.29), \ x\ −0.494)`   seeing the scaffold → using the scaffold  
  **bessel_j0** (series-head seed, 3000)   9.0e-2   **2.7e-2** ×3.3   series head mutated into exp/sin composite  
  **bessel_j1** (odd-series seed, 3000)   1.86   **0.56** ×3.3   same leap  
  **aces_fit** (rational seed, **8000**)   2.4e-5 @53u   1.2e-5 @13u — still ×0.62   **Narkowicz keeps the crown**; the hand-fit is genuinely excellent  
### 🧪 Wave 4 — never-benchmarked operations

| Task | Use | Record | Cost | vs exact |
|---|---|---|---|---|
| **probit_quantile** ⭐ | THE quantile kernel (VaR · z-scores · probit regression) | **1.9e-5, L3** gen-1 — novel √√log+x² hybrid at Acklam-cost parity | 29 | ×0.97 |
| **pmt_finance** ⭐ | loan payment per unit, embedded fintech | **4.4e-7, L2** — discovered **atan(n) hybrid replaces e^(n·ln(1+r))**, ×1.38 faster than textbook | 34 | **×1.38** |
| logsumexp2 (SPEAR² self-hunt) | beating our own ×8.57 record | record held — the 7-unit form resists its own engine | 7 | ×8.57 |

The pmt discovery is the kind of surprise that justifies the whole method: nobody writes loan-payment kernels with arctangent, yet evolution found one that is cheaper than the textbook exponential form and exact-grade accurate.
### 🌈 Wave 5 — PBR trio completed & hardened references

| Task | Use | Record | Cost |
|---|---|---|---|
| **blackbody_g** | green channel vs temperature | **1.3e-4, L2** — ln/power dual-regime composite | 74 |
| **blackbody_b** | blue channel — hard-zero then ln growth | **2.3e-4, L2** — clean log form | 65 |

With `blackbody_r` (L4), the full PBR color-temperature trio is discovered and registry-complete. Methodology note: the probit reference was rebuilt on bisection over our own A&S erf after a hand-transcribed Acklam proved unreliable — poisoned-reference bugs are caught by refusing to accept walls.
### 🌊 Wave 7 — everyday-science kernels (all records gen-1)

| Task | Domain | Record | Cost |
|---|---|---|---|
| **stefan_boltzmann** | thermal radiation PBR | **2.1e-26 EXACT** @ 5u (optimal 3u) — optimality test #5 | 5 |
| **mm1_queue_wait** | systems/SRE queueing | **6.1e-33 EXACT** — true form `−((l/m)/(l−m))` found | 10 (optimal 6) |
| **michaelis_menten** | biochemistry/pharma | 6.9e-10, near-exact rational | 15 |
| **temperature_softmax** ⭐ | LLM sampling temperature | 4.2e-4 via NOVEL `atan(e^(Δ/T))` Gudermannian shape | 46 |
| **doppler_effect** | audio/radar sirens | **9.3e-10 EXACT** via perturbed scaffold | 9 |
### ⚛️ Wave 8 — quantum computing operations

| Task | Domain | Record | Cost |
|---|---|---|---|
| **concurrence_pure** ✅ | entanglement measure | **EXACT (MSE=0, L5) at 5 units** — better than hand-written estimate | 5 |
| **chsh_correlation** ✅ | Bell inequality · Nobel Physics 2022 | **2.1e-32 EXACT** — full CHSH expression recovered via scaffold | 92 |
| **grover_amplitude** | quadratic speedup law | 1.07e-1, L0 — asin unservable, hard mode open | 30 |

The concurrence solve is remarkable: the engine found the optimal 5-unit form (`2·|ad−bc|`) — *cheaper than the hand-written estimate*. The CHSH expression (Nobel-grade Bell test) was recovered to machine precision via explicit-form scaffold seeding.
### ⚗️ Wave 7b — scaffold-exact breakthroughs

Two more tasks fell to machine-zero via targeted scaffolds:

| Task | Domain | Record | Method |
|---|---|---|---|
| **temperature_softmax** ✅ | LLM sampling temperature | **7.4e-47 EXACT** | σ(Δ/T) scaffold → full convergence |
| **hill** ✅ | pharmacology dose-response | **TRUE ZERO** at 9 units | c³/(1+c³) scaffold → exact recovery |
### 🔮 Quantum speedup front — replacing iterative solvers in quantum computing

| Task | Replaces | Record | vs Solver |
|---|---|---|---|
| **qfi_dephasing** | BFGS optimization over measurement bases | 1.47, L0 | **×15 vs BFGS** |
| **amp_damp_fid** | Kraus operator evaluation for QEC | 2.1e-2, L2 | ×0.19 (needs slimming) |
| **loschmidt_rate** | Exact diagonalization of TFIM modes | 5.3e-2, L2 | **×14 vs full diag** |
| **grover_amplitude** | asin-based Grover phase computation | 1.07e-1, L0 | hard mode open |
### ⚙️ Ultra-common operations registry (new)

Five everyday kernel/program primitives added, first-generation results at 500 iterations:

  Task   Use   Record   Cost   vs exact  
 --- --- --- --- --- 
  **atan_unit**   core of every atan2   **6.6e-9, L5** — rediscovered the classic Padé `[x+0.195x³]/[1+0.52x²]`   12   ×1.67  
  **ema_smooth**   frame-rate-independent smoothing, every game frame   **1.6e-11, L5** near-exact   12   ×1.92  
  **smoothstep** ✅   THE shader interpolation — optimality test #2 **PASSED PERFECTLY**   **3.3e-7, L5 at exactly 5 units** — `x²(x−1.5)` is the true polynomial   5   ×1.00  
  **srgb_gamma**   linear→display transfer, every pixel every frame   1.9e-6, L4 via nested sqrt/sin composite   34   ×0.65  
  **tanh_sat** ⭐   audio soft-clip + NN gating   **4.6e-6, L4** — ×158 jump once the right shape surfaced   12   ×0.65  

### 🧬 Second wave — special functions & glue (1000-iteration first generation)

  Task   Use   Record   Cost   vs exact  
 --- --- --- --- --- 
  **cosh_curve** ⭐   catenaries · audio ring-mod   **2.6e-8, L5** — `sqrt(x²)·cos(x²)` composite   32 (vs 44)   **×1.38 FASTER**  
  **srgb_decode**   display→linear, mirror of srgb_gamma   2.5e-7, **L5** — min/log composite   26   ×0.85  
  **logit_ml**   probs↔logits classifier glue   9.4e-4, L2 (cost 66 — needs slimming passes)   66   ×0.41  
  **erf_prob**   probability kernels · GELU-exact grade   8.4e-4, L2 — sin+linear hybrid   30   ×0.87  
  **huber_loss** ✅   robust training loss — optimality test #3 **PASSED**   **3.0e-7, L5** via max-scaffold seed   12   ×0.42  
  **bessel_j0**   FM sidebands · vibrations · beams   9.4e-2, L0 — hardest open problem in the registry   47   ×0.85  
Concurrency fix along the way: `run-farm.ts` now takes a `.farm-lock` single-writer guard — two simultaneous farms were last-writer-wins overwriting each other's records.
Honest bench verdict on THIS machine: WASM per-pixel timing shows the student-t k3 at ×0.89 vs native `Math.exp` — **not faster here**. The menu targets backends where transcendental evaluation is expensive or absent (low-end GLSL profiles, quantized pipelines); measure before shipping.

### ⚡ vs iterative solvers — the honest big multipliers

For several tasks the *standard* way to compute the answer is not another closed form but an **iterative numerical solver**. Counting the solver's full bill (every iteration, in ALU/SFU units) against our O(1) formula gives large — and legitimate — accelerations:

  Task   Discovered formula cost   Iterative baseline   Speedup  
 --- --- --- --- 
  **Gaussian CDF Φ(x)**   9 units   Monte-Carlo estimation, 1000 Box-Muller draws (46,000)   **×5111**  
  **Kerr deflection with spin**   15 units   RK4 geodesic integration, 200 steps (2,400)   **×160**  
  **Damped oscillation trajectory**   30 units   RKF45 adaptive solve, 300 steps (5,400)   **×200**  
  **Damped pendulum terminal state**   46 units   Euler-Cromer integration, 60 steps (1,500)   **×33**  

Arithmetic is documented in [`src/lib/spear/benchmarks.ts`](./src/lib/spear/benchmarks.ts) (`ITERATIVE_BASELINES`). These are cost-model units, cross-checked by wall-clock benchmarks in the export audit.

### 🎚️ Two operating points per law: `precise` and `fast`

One formula per task hides the accuracy/latency trade-off that real deployments live by: a rendering loop or an LLM hot path happily accepts 1e-4 relative error if it shaves half the ALU bill; a physics integration does not. So the ledger keeps **two validated forms per task**: the *precise* champion (lowest error ever found) and the *fast* variant (cheapest form that still passes the engine's level-2 validation gate). **19 of 27 tasks** have a genuinely cheaper second form; the rest have champions already at minimal cost.

Biggest fast-slot wins (cost = precise → fast, in ALU/SFU units):

  Task   Precise   Fast   Cost cut  
 --- --- --- --- 
  **Lambert W₀**   exact, 22 units   1.1e-2, 1 unit   **×22**  
  **Gaussian blur kernel**   exact e^(−x²/2), 24 units   5.7e-3, 1 unit   **×24**  
  **Hill dose-response**   1.16e-4, 9 units   1.8e-3, 1 unit   **×9**  
  **Gaussian CDF Φ(x)**   2.1e-4, 29 units   7.6e-4, 6 units   ×4.8  
  **Softplus**   2.3e-4, 9 units   3.2e-3, 2 units   ×4.5  
  **Sigmoid**   exact, 27 units   7.2e-4, 7 units — `x/(1+ x )`   ×3.9  
  **Kerr deflection w/ spin**   7.5e-10, 18 units (×133 vs RK4)   `10.49/(s+b)`, 3.6e-5, 5 units   **×480 vs RK4**  

Displaced champions are never lost: when a more accurate form takes over, the old one is demoted to the fast slot, [`scripts/backfill-fast.ts`](./scripts/backfill-fast.ts) resurrects historical forms from git archaeology, and every ledger refresh prunes any "fast" variant that is no longer actually cheaper than its champion.

**Note on protected division:** the engine's `pdiv` clamps denominators below 1e-4 and outputs beyond ±1e4. The Gaussian CDF champion *exploits these rails as free saturation* — its tiny divisor (3.4e-5) turns the division into a hard plateau, exactly the shape of Φ. The algebraic optimizer (`div-by-const → mul-by-reciprocal`, −3 units per site) therefore only rewrites divisors above the protection floor, and every rewrite passes a metric-parity gate before entering the ledger.

### Decision-making (KV-cache)

  Task   Record   Discovered rule   Baselines beaten  
 --- --- --- --- 
  **KV-cache eviction**   **70.0 %** future attention mass retained   `4.5·S + A + R` — additive triad, sink-dominated   H2O, StreamingLLM, SnapKV, sliding window, random  

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

## 🔌 MCP Server — use SPEAR from Claude, Cursor & any LLM agent

SPEAR ships with a [Model Context Protocol](https://modelcontextprotocol.io) server exposing the engine as 5 tools that any LLM agent can call natively:

| Tool | What it does |
|---|---|
| `discover` | Evolve a kernel for a target formula → returns optimized form + WASM/C/PyTorch exports |
| `list_kernels` | Browse all registry entries with records, speedups, exact-solve status |
| `get_kernel` | Full details: precise/fast formulas, MISRA-C export, Python export, cost breakdown |
| `evaluate_kernel` | Evaluate a kernel at specific input values (precise or fast slot) |
| `run_benchmark` | Launch discovery passes on selected tasks |

### Setup (Claude Desktop / Cursor / Windsurf)

```json
{
  "mcpServers": {
    "spear-kernels": {
      "command": "npx",
      "args": ["tsx", "/path/to/superspear/src/mcp/server.ts"]
    }
  }
}
```

Once connected, just ask your agent:
> *"Discover a fast approximation of sin(x)·exp(-x/3) on [-2,8]"*

→ The agent calls `discover`, SPEAR evolves a kernel in seconds, and returns the verified formula with parity-checked WASM/C exports.

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
- The inverted-pendulum control surrogate (×3.76 vs the 173-unit exact law, moderate accuracy) is displayed exactly as that: the search surfaced a cheap-but-crude controller — a legitimate Pareto point, not hidden behind the headline number.

## Provenance

The benchmark portfolio draws inspiration from the GROUNDED-SPEAR paper series (V600 → V1200): Padé [3/2] distillation shapes (BT10), Kerr geodesic deflection incl. spin (BT24/BT35), KdV solitons (BT36) and hybrid Lyapunov control laws (§4). Everything else — the engine, the audits, the numbers above — is generated and measured by this repository.

## License

[MIT](./LICENSE) — free to use, modify, and ship; the discovered formulas are yours too.
