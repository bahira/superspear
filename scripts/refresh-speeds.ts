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
    fast?: { formulaCost?: number; speed?: { formulaCost: number; exactCost?: number; speedup?: number; vsIterative?: { label: string; speedup: number } } };
    fastTree?: unknown;
  }>;
  const defs = new Map(buildTasks().map((t) => [t.id, t]));
  let updated = 0;
  for (const [id, entry] of Object.entries(ledger)) {
    const def = defs.get(id);
    if (!def) continue;
    // exact reference: same source as benchmarkSpeed (may be absent for
    // simulation-based tasks — the vs-iterative multiplier still applies)
    const exactCost = def.exactCost ?? (def.exactRefNode ? estimateCost(def.exactRefNode) : undefined);
    const it = def.iterativeBaseline;
    if (!entry.tree && !entry.fastTree) continue;
    if (!exactCost && !it) continue;

    const fill = (tree: unknown): { formulaCost: number; exactCost?: number; speedup?: number; vsIterative?: { label: string; speedup: number } } | null => {
      let node;
      try { node = parseNode(tree as never); } catch { return null; }
      const cost = estimateCost(node);
      if (cost < 1) return null;
      return {
        formulaCost: cost,
        ...(exactCost ? { exactCost, speedup: exactCost / Math.max(1, cost) } : {}),
        vsIterative: it ? { label: it.label, speedup: it.totalCost / Math.max(1, cost) } : undefined,
      };
    };

    if (entry.tree) {
      const s = fill(entry.tree);
      if (s) { entry.speed = { ...entry.speed, ...s }; updated++; }
    }
    if (entry.fastTree) {
      const sf = fill(entry.fastTree);
      if (sf) {
        entry.fast = { ...(entry.fast ?? {}), speed: { ...entry.fast?.speed, ...sf } };
        updated++;
      }
    }
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
