// Parallel farm: spawns N node workers, each running a disjoint slice of the
// benchmark tasks concurrently (one process per slice). Merges all partials
// into the hall-of-fame ledger.
//
// Usage: npx tsx scripts/run-farm.ts [seed] [budget] [workers]
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { performance } from "node:perf_hooks";

interface Finding {
  taskId: string;
  title: string;
  direction: "min" | "max";
  metric: number;
  level: number;
  formula: string;
  seed: number;
  iteration: number;
  speed?: { formulaCost: number; exactCost: number; speedup: number };
}
type Ledger = Record<string, Finding>;

const MAX_TASKS = new Set(["kv_cache"]);

function better(dir: "min" | "max", a: number, b: number): boolean {
  return dir === "min" ? a < b : a > b;
}

async function main() {
  const seed = Number(process.argv[2] ?? 4242);
  const budget = Number(process.argv[3] ?? 1000);
  const workers = Number(process.argv[4] ?? 4);

  // list all task ids (env unset here)
  delete process.env.SPEAR_TASKS;
  const { buildTasks } = await import("../src/lib/spear/benchmarks");
  const ids = buildTasks().map((t) => t.id);
  const slices: string[][] = Array.from({ length: workers }, () => []);
  ids.forEach((id, i) => slices[i % workers].push(id));

  const ledgerPath = join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json");
  const ledger: Ledger = existsSync(ledgerPath) ? JSON.parse(readFileSync(ledgerPath, "utf8")) : {};

  const dir = mkdtempSync(join(tmpdir(), "spear-farm-"));
  const tsxCli = join(process.cwd(), "node_modules", "tsx", "dist", "cli.mjs");
  const childScript = join(import.meta.dirname ?? ".", "farm-child.ts");

  console.log(`▶ farm: ${workers} workers × ${ids.length} tâches, seed=${seed} budget=${budget}`);
  const t0 = performance.now();
  const procs = slices.map((slice, w) =>
    new Promise<string>((resolve, reject) => {
      const out = join(dir, `w${w}.json`);
      try {
        execFileSync(process.execPath, [tsxCli, childScript, out, String(seed), String(budget), slice.join(",")], { stdio: ["ignore", "pipe", "pipe"] });
        resolve(out);
      } catch (e) { reject(e); }
    }),
  );
  const outs = await Promise.all(procs);
  const wall = (performance.now() - t0) / 1000;

  let records = 0;
  let totalBt = 0;
  for (const o of outs) {
    for (const f of JSON.parse(readFileSync(o, "utf8")) as (Finding & Record<string, unknown>)[]) {
      totalBt++;
      const prev = ledger[f.taskId];
      if (!prev || better(f.direction, f.metric, prev.metric)) {
        ledger[f.taskId] = f;
        records++;
      } else if (prev.formula === f.formula) {
        // backfill missing annotations on reproduced champions
        if (!prev.speed && f.speed) prev.speed = f.speed;
        if (prev.speed && !prev.speed.vsIterative && f.speed?.vsIterative) prev.speed.vsIterative = f.speed.vsIterative;
        if (!prev.tree && f.tree) prev.tree = f.tree;
      }
    }
  }
  rmSync(dir, { recursive: true, force: true });
  writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2));

  console.log(`terminé en ${wall.toFixed(1)} s — ${records} nouveaux records sur ${totalBt} résultats`);
  for (const f of Object.values(ledger).sort((a, b) => a.taskId.localeCompare(b.taskId))) {
    const m = f.direction === "min" ? f.metric.toExponential(2) : f.metric.toFixed(1) + "%";
    const sp = f.speed ? ` ×${f.speed.speedup.toFixed(2)}` : "";
    console.log(`  [${f.taskId.padEnd(18)}] ${m.padStart(10)}  L${f.level}  seed ${f.seed}${sp}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
