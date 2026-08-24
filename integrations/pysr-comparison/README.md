# SPEAR vs PySR head-to-head harness

Answers the reviewer question *"how does SPEAR compare to PySR?"* by running both
engines on identical datasets: the same registry tasks, the same rows, the same
MSE definition. Raw numbers only — no normalization tricks.

## What's here

| File | Role |
|---|---|
| `export-datasets.ts` | Exports 12 registry tasks to `datasets/<id>.csv` (X columns + `y` column) and writes `datasets/manifest.json` (var names, domains, row counts, SPEAR champion formula / cost units / MSE). Data regenerates the engine's exact evaluation grids via each task's `exactFn`/generating AST; a self-check re-scores every ledger champion on the exported rows and prints the ratio vs the ledger metric (should be ×1.000). |
| `run-pysr.py` | Fits `PySRRegressor(niterations=40, populations=15, binary_operators=[+,-,*,/], unary_operators=[exp,sin,cos,sqrt,log,atan], model_selection="best")` per CSV, records wall-clock time and best-equation MSE on the same rows, writes `results.json`, prints a comparison table. |
| `datasets/` | The CSVs + `manifest.json`. |

## How to run

```bash
# 1. Export datasets (repo root, uses the repo's own builders)
npx tsx integrations/pysr-comparison/export-datasets.ts

# 2. Run PySR (heavy: first run installs a Julia toolchain, several minutes)
pip install pysr pandas numpy
python integrations/pysr-comparison/run-pysr.py
```

The 12 tasks span activation kernels (silu, gelu, sigmoid, softplus, tanh_sat,
atan_unit, srgb_gamma), probability/shader curves (gaussian_cdf) and physical
laws (hill, logistic_growth, damped_oscillation, kdv_soliton).

## How to read the numbers honestly

**MSE columns.** Both engines are scored on exactly the same rows with plain
mean squared error. This part is directly comparable.

**Time column is NOT comparable across engines.** SPEAR runs inside this repo's
Node process; PySR pays Julia startup, JIT compilation and operator-definition
overhead, mostly on the first task. Report raw wall-clock, but do not claim
engine-speed conclusions from it. If you need budget-matched comparisons, fix an
equal wall-clock cap per task and report MSE-at-budget from both sides — that is
a different experiment than this script performs.

**SPEAR "cost units" are not FLOPs.** They are GPU-flavored ALU/SFU op weights
(add/mul = 1, div = 4, sqrt = 2, transcendental = 20) used by SPEAR's internal
Pareto search. They measure *deployed-kernel cheapness*, not search effort.
PySR's `complexity` counts nodes and is a different scale again. Never divide
one by the other.

**Different search budgets.** SPEAR champions come from long hall-of-fame runs
(hundreds of iterations plus constant refinement); PySR here gets a fixed
40×15 default setup. A PySR loss on a task may be a budget artifact, and vice
versa. Rerun with more iterations before claiming a trend.

## Legitimate conclusions

- ✅ "On these 12 fixed datasets with these settings, final-train-MSE was X vs Y."
- ✅ "Engine A reached MSE ≤ threshold on k/12 tasks within budget B."
- ✅ Structural observations: which operator vocabulary each engine needed
  (e.g. whether algebraic-only approximants emerge without being seeded).
- ❌ "SPEAR formulas are Nx faster" from cost units alone (units are not FLOPs).
- ❌ Any wall-clock speed ranking from this harness as-is.
- ❌ Generalization claims: everything here is train-set MSE on noiseless grids;
  there is no holdout split in this harness.

## Known caveats

- Datasets are noiseless deterministic grids (the registry's own evaluation
  data). Interpolation-style MSE favors smooth-formula engines; add noise if you
  care about robustness.
- `model_selection="best"` trades accuracy vs complexity inside PySR; switching
  to `"accuracy"` changes its MSE. State the setting used.
- SPEAR champion MSEs in the manifest were re-scored on the exported rows
  (self-check ×1.000 vs ledger metrics at export time).
- Non-finite PySR predictions (e.g. `sqrt`/`log` domain violations outside what
  PySR protects) are excluded from its MSE and counted in
  `nonfinite_preds` in `results.json`.
