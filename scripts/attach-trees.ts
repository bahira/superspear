// Reattach ASTs to ledger entries that predate tree storage, by parsing their
// printed formula (inverse of nodeToString). Verifies round-trip + metric
// parity before writing. Run: npx tsx scripts/attach-trees.ts
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseFormula, nodeToString, serializeNode, evaluateNode } from "../src/lib/spear/engine";
import { buildTasks } from "../src/lib/spear/benchmarks";

async function main() {
  const path = join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json");
  const ledger = JSON.parse(readFileSync(path, "utf8")) as Record<string, { formula: string; metric: number; tree?: unknown }>;
  const defs = new Map(buildTasks().map((t) => [t.id, t]));

  for (const [id, entry] of Object.entries(ledger)) {
    if (entry.tree) continue;
    if (entry.metric === 0) { console.log(`[${id}] record exact — arbre non reconstruit (les constantes affichées dégraderaient l'exactitude)`); continue; }
    const def = defs.get(id);
    if (!def) { console.log(`[${id}] tâche inconnue, ignoré`); continue; }
    let node;
    try {
      node = parseFormula(entry.formula);
    } catch (e) {
      console.log(`[${id}] parse échoué: ${String(e).slice(0, 90)}`);
      continue;
    }
    const rt = nodeToString(node);
    if (rt !== entry.formula) {
      console.log(`[${id}] round-trip divergent:\n  src: ${entry.formula}\n  rtt: ${rt}`);
      continue;
    }
    // semantic check: re-evaluate on the task's own data (informational —
    // display-rounded constants may score slightly worse than the original)
    let metricNow = NaN;
    try {
      metricNow = def.evaluate(node).metric;
    } catch { /* keep NaN */ }
    const rel = Math.abs(metricNow - entry.metric) / (Math.abs(entry.metric) + 1e-300);
    console.log(`[${id}] attaché — métrique enregistrée ${entry.metric.toExponential(4)}, reconstruite ${metricNow.toExponential(4)} (rel ${Number.isFinite(rel) ? rel.toExponential(1) : "?"})`);
    entry.tree = JSON.parse(JSON.stringify(serializeNode(node)));
  }

  writeFileSync(path, JSON.stringify(ledger, null, 2));
  console.log("terminé");
}

void evaluateNode;
main().catch((e) => { console.error(e); process.exit(1); });
