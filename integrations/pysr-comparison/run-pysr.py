#!/usr/bin/env python3
"""Run PySR on the exported SPEAR task datasets and compare against SPEAR champions.

Usage:
    pip install pysr pandas numpy
    python integrations/pysr-comparison/run-pysr.py

Reads datasets/<id>.csv + datasets/manifest.json (produced by export-datasets.ts),
fits one PySRRegressor per task, measures wall-clock time and best-equation MSE
on the same rows, writes results.json and prints a comparison table.

NOTE: PySR installs its own Julia toolchain on first run (several minutes,
one-time). Wall-clock times include Julia startup on the first task; see README.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    from pysr import PySRRegressor
except ImportError:
    sys.exit("PySR not installed. Run: pip install pysr pandas numpy")

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not installed. Run: pip install pysr pandas numpy")

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"


def fit_task(task: dict) -> dict:
    csv = DATASETS / f"{task['id']}.csv"
    df = pd.read_csv(csv)
    X = df[task["varNames"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    model = PySRRegressor(
        niterations=40,
        populations=15,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "sin", "cos", "sqrt", "log", "atan"],
        model_selection="best",
    )
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0

    pred = np.asarray(model.predict(X), dtype=float)
    nonfinite = int(np.count_nonzero(~np.isfinite(pred)))
    if nonfinite < pred.size:
        mse = float(np.mean((pred[np.isfinite(pred)] - y[np.isfinite(pred)]) ** 2))
    else:
        mse = math.inf

    try:
        formula = str(model.sympy())
    except Exception:
        eqs = getattr(model, "equations_", None)
        formula = str(eqs.iloc[-1]["equation"]) if eqs is not None and len(eqs) else "?"
    complexity = int(getattr(model.get_best(), "complexity", 0) or 0)

    return {
        "id": task["id"],
        "rows": len(df),
        "pysr": {
            "formula": formula,
            "complexity": complexity,
            "mse": mse,
            "time_s": round(elapsed, 1),
            "nonfinite_preds": nonfinite,
        },
        "spear": {
            "formula": (task.get("spearChampion") or {}).get("formula"),
            "costUnits": (task.get("spearChampion") or {}).get("costUnits"),
            "mse": (task.get("spearChampion") or {}).get("mse"),
        },
    }


def fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return "inf"
    return f"{x:.3e}"


def main() -> None:
    manifest = json.loads((DATASETS / "manifest.json").read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    print(f"{len(tasks)} tasks | results -> {HERE / 'results.json'}\n")

    results = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task['id']} ({task['rows']} rows) ...", flush=True)
        try:
            results.append(fit_task(task))
        except Exception as e:  # keep going: one failed task must not kill the run
            print(f"    FAILED: {e}")
            results.append({"id": task["id"], "error": str(e)})

    (HERE / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== SPEAR vs PySR (same CSV rows, MSE lower is better) ===")
    print(f"{'task':<20} {'spear MSE':>11} {'PySR MSE':>11} {'ratio':>8} "
          f"{'spear cost':>10} {'PySR t(s)':>10}")
    for r in results:
        if "error" in r:
            print(f"{r['id']:<20} ERROR: {r['error'][:60]}")
            continue
        sm, pm = r["spear"].get("mse"), r["pysr"]["mse"]
        ratio = f"{pm / sm:.2f}" if sm and pm and math.isfinite(pm) and sm > 0 else "—"
        cost = r["spear"].get("costUnits")
        print(f"{r['id']:<20} {fmt(sm):>11} {fmt(pm):>11} {ratio:>8} "
              f"{cost if cost is not None else '—':>10} {r['pysr']['time_s']:>10.1f}")
    print("\nFormulas in results.json. Caveats: see README.md (cost units are "
          "ALU/SFU weights, search budgets differ, wall-clock is not comparable).")


if __name__ == "__main__":
    main()
