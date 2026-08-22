// One-shot algebraic optimization pass over ledger trees. Applies the current
// `simplify` rules (incl. div-by-const -> mul-by-reciprocal) to every stored
// AST, verifies metric parity on the task's own evaluator, and keeps the
// cheaper form only when the metric did not degrade.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseNode, serializeNode, simplify, nodeToString, estimateCost } from "../src/lib/spear/engine";
import { buildTasks } from "../src/lib/spear/benchmarks";

interface Speed { formulaCost: number; exactCost?: number; speedup?: number; vsIterative?: { label: string; speedup: number } }
interface Entry {
  formula: string;
  metric: number;
  speed?: Speed;
  tree?: unknown;
}
type Ledger = Record<string, Entry>;

async function main() {
  const path = join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json");
  const ledger = JSON.parse(readFileSync(path, "utf8")) as Ledger;
  const defs = new Map(buildTasks().map((t) => [t.id, t]));

  let touched = 0;
  for (const [id, entry] of Object.entries(ledger)) {
    if (!entry.tree) continue;
    const def = defs.get(id);
    if (!def) continue;
    let orig;
    try { orig = parseNode(entry.tree as never); } catch { continue; }
    const opt = simplify(orig);
    const costBefore = estimateCost(orig);
    const costAfter = estimateCost(opt);
    if (costAfter >= costBefore && nodeToString(opt) === nodeToString(orig)) continue;

    // parity gate on the task's official evaluator
    const before = def.evaluate(orig).metric;
    const after = def.evaluate(opt).metric;
    const degraded = after > before * (1 + 1e-6) + 1e-12;
    console.log(
      `[${id}] cost ${costBefore} -> ${costAfter}` +
      (costBefore !== costAfter ? ` (−${costBefore - costAfter})` : "") +
      `  metric ${before.toExponential(4)} -> ${after.toExponential(4)}${degraded ? "  ✗ DEGRADED, keep original" : "  ✓"}`,
    );
    if (degraded || costAfter >= costBefore) continue;

    entry.formula = nodeToString(opt);
    entry.metric = after;
    entry.tree = JSON.parse(JSON.stringify(serializeNode(opt))); // ledger format is {o,v,n,c}
    if (entry.speed) {
      entry.speed.formulaCost = costAfter;
      if (entry.speed.exactCost) entry.speed.speedup = entry.speed.exactCost / Math.max(1, costAfter);
      if (entry.speed.vsIterative?.label) {
        // recompute vs-iterative from ITERATIVE_BASELINES via def
        const it = def.iterativeBaseline;
        if (it) entry.speed.vsIterative = { label: it.label, speedup: it.totalCost / Math.max(1, costAfter) };
      }
    }
    touched++;
  }

  writeFileSync(path, JSON.stringify(ledger, null, 2));
  console.log(`\noptimized: ${touched} entries rewritten`);
}

main().catch((e) => { console.error(e); process.exit(1); });
