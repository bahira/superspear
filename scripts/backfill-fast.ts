import { loadLedger, saveLedger } from "../src/lib/spear/ledger";
// Backfill the `fast` deployment variant of every ledger entry from git
// history: displaced champions were accurate-but-slower forms that still
// passed validation (level >= 2). Resurrect the cheapest one per task.

import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { estimateCost, nodeToString, parseNode, serializeNode, type SpearNode } from "../src/lib/spear/engine";
import { buildTasks } from "../src/lib/spear/benchmarks";
import type { TaskDef } from "../src/lib/spear/tasks/types";

interface Speed { formulaCost?: number; exactCost?: number; speedup?: number }
interface Finding {
  taskId: string;
  metric: number;
  level: number;
  formula: string;
  seed: number;
  iteration: number;
  tree?: unknown;
  speed?: Speed;
  fast?: { formula: string; metric: number; level: number; formulaCost: number; seed: number };
  fastTree?: unknown;
}
type Ledger = Record<string, Finding>;

function levelOf(t: TaskDef, metric: number): number {
  let lvl = 0;
  for (const m of t.milestones) if (Number.isFinite(metric) && m.test(metric)) lvl = Math.max(lvl, m.level);
  return lvl;
}

// NOTE — why this script re-derives everything from the AST instead of
// trusting what the ledger stores (incident of 2026-10-04, 7 tasks at once):
// old revisions priced a direct primitive call at 1 unit, so erf_prob's exact
// law `erf(x)` was stored with formulaCost=1 (commit e44f29b). Comparing that
// against a validated algebraic fast slot (real cost 7) made this script
// overwrite the good one with the exact law, everywhere it found such an entry.
// The AST is the only trustworthy source for cost, metric, level and text.

function historicalLedgers(repo: string): Ledger[] {
  const shas = execFileSync("git", ["rev-list", "HEAD", "--", "spear-hall-of-fame.json"], { cwd: repo, encoding: "utf8" })
    .split("\n").filter(Boolean);
  const out: Ledger[] = [];
  for (const sha of shas) {
    try {
      const raw = execFileSync("git", ["show", `${sha}:spear-hall-of-fame.json`], { cwd: repo, encoding: "utf8" });
      out.push(JSON.parse(raw));
    } catch { /* file absent at this revision */ }
  }
  return out;
}

async function main() {
  const repo = join(import.meta.dirname ?? ".", "..");
  
  const ledger = loadLedger() as Ledger;
  const tasks = new Map(buildTasks().map((t) => [t.id, t]));

  let revived = 0;
  let skippedUnvalidated = 0;
  let skippedAsGoodAsChampion = 0;
  for (const old of historicalLedgers(repo)) {
    for (const [id, e] of Object.entries(old)) {
      const cur = ledger[id];
      if (!cur || (e.level ?? 0) < 2) continue;
      if (!e.tree) continue;
      const def = tasks.get(id);
      if (!def) continue;
      // Re-score the candidate against the CURRENT task data, and STORE the
      // normalized tree that was actually scored. Copying the historical
      // metric/level/text produced audit's `fast-metric-drift` and
      // `fast-level-inflated`: what ships is the AST, so the metadata has to be
      // derived from that AST against today's data.
      let node: SpearNode;
      try {
        node = parseNode(e.tree as never);
      } catch { continue; }
      let metric: number;
      try {
        const ev = def.evaluate(node);
        if (!ev.finite || !Number.isFinite(ev.metric)) { skippedUnvalidated++; continue; }
        metric = ev.metric;
      } catch { skippedUnvalidated++; continue; }
      // A form that is exactly as accurate as the champion is not a fast slot,
      // it is a rewrite of the champion. Comparing metrics rather than formula
      // text is what catches `1 - 1/(1+exp(x))` when the champion reads
      // `1/(1+exp(-x))` — textually different, mathematically identical.
      if (Number.isFinite(cur.metric) && !(metric > cur.metric * (1 + 1e-9) + 1e-300)) {
        skippedAsGoodAsChampion++;
        continue;
      }
      const level = Math.max(2, levelOf(def, metric));
      const cost = estimateCost(node);
      const champCost = cur.speed?.formulaCost ?? (cur.tree ? estimateCost(parseNode(cur.tree as never)) : Infinity);
      if (cost >= champCost) continue; // not cheaper than champion
      if (cost >= (cur.fast?.formulaCost ?? Infinity)) continue; // not cheaper than known fast
      cur.fast = { formula: nodeToString(node), metric, level, formulaCost: cost, seed: e.seed };
      cur.fastTree = serializeNode(node);
      revived++;
    }
  }

  saveLedger(ledger);
  console.log(`fast variants resurrected: ${revived}`);
  if (skippedAsGoodAsChampion) console.log(`candidats aussi justes que le champion (réécritures) ignorés: ${skippedAsGoodAsChampion}`);
  if (skippedUnvalidated) console.log(`candidats non revalidés (L2) ignorés: ${skippedUnvalidated}`);
  for (const [id, e] of Object.entries(ledger)) {
    if (e.fast) {
      const gain = (e.speed?.formulaCost ?? 0) / Math.max(1, e.fast.formulaCost);
      console.log(`[${id}] fast ×${gain.toFixed(2)} cheaper | metric ${e.metric.toExponential(2)} -> ${e.fast.metric.toExponential(2)} | ${e.fast.formula.slice(0, 70)}`);
    }
  }
}

main().catch((err) => { console.error(err); process.exit(1); });
