// Price every ledger entry against its task's iterative baseline:
//   speedup = solver total cost / formula cost (same ALU/SFU units)
// Cheap VALIDATED fast slots previously had their x-multipliers computed only
// in prose (README tables) â€” never stored in the ledger itself. This pass
// makes them first-class records.
//
// Usage: npx tsx scripts/price-fast.ts
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { buildTasks } from "../src/lib/spear/benchmarks";

interface IterRef { label: string; speedup: number }
interface FastEntry { formula: string; metric: number; level: number; formulaCost: number; seed?: number; vsIterative?: IterRef }
interface Finding {
  taskId: string;
  speed?: { formulaCost?: number; vsIterative?: IterRef };
  fast?: FastEntry;
}
type Ledger = Record<string, Finding>;

const round1 = (v: number): number => Math.round(v * 10) / 10;

async function main() {
  const path = join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json");
  const ledger = JSON.parse(readFileSync(path, "utf8")) as Ledger;
  const defs = new Map(buildTasks().map((t) => [t.id, t]));

  let priced = 0;
  for (const [id, e] of Object.entries(ledger)) {
    const ib = defs.get(id)?.iterativeBaseline;
    if (!ib) continue;
    // refresh precise pricing if missing, stale-labelled or mis-rounded
    if (e.speed?.formulaCost) {
      const wantPrecise = round1(ib.totalCost / Math.max(1, e.speed.formulaCost));
      if (!e.speed.vsIterative || e.speed.vsIterative.label !== ib.label ||
          Math.abs(e.speed.vsIterative.speedup - wantPrecise) > 0.005) {
        e.speed.vsIterative = { label: ib.label, speedup: wantPrecise };
        priced++;
      }
    }
    // THE point: fast slots earn their own multiplier record
    if (e.fast?.formulaCost) {
      const wantFast = round1(ib.totalCost / Math.max(1, e.fast.formulaCost));
      if (!e.fast.vsIterative || Math.abs(e.fast.vsIterative.speedup - wantFast) > 0.005) {
        e.fast.vsIterative = { label: ib.label, speedup: wantFast };
        priced++;
      }
    }
  }

  writeFileSync(path, JSON.stringify(ledger, null, 2));
  console.log(`entrÃ©es tariffÃ©es: ${priced}\n`);
  console.log("== multiplicateurs vs solveur ==");
  const rows: { id: string; mult: number; what: string; ref: string }[] = [];
  for (const [id, e] of Object.entries(ledger)) {
    if (e.speed?.vsIterative) rows.push({ id, mult: e.speed.vsIterative.speedup, what: "precise", ref: e.speed.vsIterative.label });
    if (e.fast?.vsIterative) rows.push({ id, mult: e.fast.vsIterative.speedup, what: "fast   ", ref: e.fast.vsIterative.label });
  }
  rows.sort((a, b) => b.mult - a.mult);
  for (const r of rows.slice(0, 20)) {
    console.log(`  Ã—${String(r.mult).padStart(8)}  [${r.id}] ${r.what}  vs ${r.ref}`);
  }
}

main().catch((err) => { console.error(err); process.exit(1); });
