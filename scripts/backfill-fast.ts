// Backfill the `fast` deployment variant of every ledger entry from git
// history: displaced champions were accurate-but-slower forms that still
// passed validation (level >= 2). Resurrect the cheapest one per task.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { execFileSync } from "node:child_process";

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
  const path = join(repo, "spear-hall-of-fame.json");
  const ledger = JSON.parse(readFileSync(path, "utf8")) as Ledger;

  let revived = 0;
  for (const old of historicalLedgers(repo)) {
    for (const [id, e] of Object.entries(old)) {
      const cur = ledger[id];
      if (!cur || (e.level ?? 0) < 2) continue;
      const cost = e.speed?.formulaCost;
      if (!cost || cost >= (cur.speed?.formulaCost ?? Infinity)) continue; // not cheaper than champion
      if (cost >= (cur.fast?.formulaCost ?? Infinity)) continue; // not cheaper than known fast
      cur.fast = { formula: e.formula, metric: e.metric, level: e.level, formulaCost: cost, seed: e.seed };
      cur.fastTree = e.tree ?? cur.fastTree;
      revived++;
    }
  }

  writeFileSync(path, JSON.stringify(ledger, null, 2));
  console.log(`fast variants resurrected: ${revived}`);
  for (const [id, e] of Object.entries(ledger)) {
    if (e.fast) {
      const gain = (e.speed?.formulaCost ?? 0) / Math.max(1, e.fast.formulaCost);
      console.log(`[${id}] fast ×${gain.toFixed(2)} cheaper | metric ${e.metric.toExponential(2)} -> ${e.fast.metric.toExponential(2)} | ${e.fast.formula.slice(0, 70)}`);
    }
  }
}

main().catch((err) => { console.error(err); process.exit(1); });
