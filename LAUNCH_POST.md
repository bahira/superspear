# SPEAR v1.0.0 — launch post

I've been building a symbolic regression engine that replaces hot math kernels with closed-form formulas discovered by genetic programming. It's tagged v1.0.0 and MIT licensed. Numbers first:

- **48-task registry. 14 exact solutions** (MSE ≤ 1e-8; 7 at exactly 0 — Lorentz factor, Lambert W₀, RoPE rotation, LayerNorm rsqrt, among others). **18 kernels are cheaper than their exact reference** in ALU/SFU units.
- Where the textbook solution is an iterative solver, an O(1) formula wins: Gaussian CDF via 1000-draw Monte-Carlo estimation replaced by a 9-unit rational form, **×1840 fewer units** (documented replacements span ×6–×1840).
- On this machine (WASM, 200k elements): algebraic SiLU runs **×2.12 faster** than the exp-based reference (409 vs 864 ns/el), GELU ×2.32. The fast sigmoid slot is *slower* here (×0.82) — reported as measured.
- A 5-unit KV-cache eviction rule (`4.5·S + A + R`) was validated on real distilgpt2 attention traces: **80.31 % future attention mass retained vs 80.20 % for H2O**, 80.07 % StreamingLLM.

How it works: NSGA-II evolves algebraic expression trees under an accuracy-vs-parsimony trade-off, with a seeded RNG so every record replays from one integer. Champions export to WebAssembly, strict MISRA-C:2012 C99, CUDA, and PyTorch — each audited by real gcc compilation plus op-by-op numeric parity (C↔WASM ≤ 1e-6 relative).

The audits caught two real bugs during this cycle: a `simplify()` collapse silently eating `(c·x)·Y` subtrees mid-search, and a poisoned Bessel reference series (wrong ground truth; after rebuilding it, records dropped ×108,000 and ×60). Both are documented in the release notes.

Ships with: the engine, the task registry, an npm package (`spear-kernels`, 47 kernels as ESM/CJS/types with generated CUDA/torch sources), a GLSL integration kit for 3DSPEAR, a discovery endpoint (`POST /api/spear/discover`), and a PAPER.md draft.

Known limits, stated in the repo: records hold on benchmark domains; some discovered forms are slower than the law itself and are kept visible as Pareto points.

Next up: multiscale validation of the KV policy beyond distilgpt2, and a systematic comparison against PySR.
