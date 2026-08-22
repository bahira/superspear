// Refresh cost-model speeds (incl. vs-iterative) for every ledger entry that
// carries a serialized AST — no reproduction needed, speed is AST-pure.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseNode, estimateCost } from "../src/lib/spear/engine";
import { buildTasks } from "../src/lib/spear/benchmarks";

async function main() {
  const path = join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json");
  const ledger = JSON.parse(readFileSync(path, "utf8")) as Record<string, {
    formulaCost?: number;
    speed?: { formulaCost: number; exactCost?: number; speedup?: number; vsIterative?: { label: string; speedup: number } };
    tree?: unknown;
  }>;
  const defs = new Map(buildTasks().map((t) => [t.id, t]));
  let updated = 0;
  for (const [id, entry] of Object.entries(ledger)) {
    if (!entry.tree) continue;
    const def = defs.get(id);
    if (!def) continue;
    let node;
    try { node = parseNode(entry.tree as never); } catch { continue; }
    const formulaCost = estimateCost(node);
    if (formulaCost < 1) continue;
    // exact reference: same source as benchmarkSpeed (may be absent for
    // simulation-based tasks — the vs-iterative multiplier still applies)
    const exactCost = def.exactCost ?? (def.exactRefNode ? estimateCost(def.exactRefNode) : undefined);
    const it = def.iterativeBaseline;
    if (!exactCost && !it) continue;
    entry.speed = {
      ...entry.speed,
      formulaCost,
      ...(exactCost ? { exactCost, speedup: exactCost / Math.max(1, formulaCost) } : {}),
      vsIterative: it
        ? { label: it.label, speedup: it.totalCost / Math.max(1, formulaCost) }
        : entry.speed?.vsIterative,
    };
    updated++;
  }
  writeFileSync(path, JSON.stringify(ledger, null, 2));
  console.log(`speeds refreshed: ${updated}`);
  for (const id of ["kerr", "kerr_spin", "damped_pendulum", "damped_oscillation", "gaussian_cdf", "kdv_soliton"]) {
    const s = (ledger[id] as { speed?: { speedup?: number; vsIterative?: { label: string; speedup: number } } } | undefined)?.speed;
    const exact = s?.speedup !== undefined ? `×${s.speedup.toFixed(2)}` : "—";
    console.log(`${id}: ${exact}${s?.vsIterative ? ` | ×${s.vsIterative.speedup.toFixed(0)} ${s.vsIterative.label}` : ""}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
