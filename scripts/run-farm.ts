import { loadLedger, saveLedger } from "../src/lib/spear/ledger";
// Parallel farm: spawns N node workers, each running a disjoint slice of the
// benchmark tasks concurrently (one process per slice). Merges all partials
// into the hall-of-fame ledger.
//
// Usage: npx tsx scripts/run-farm.ts [seed] [budget] [workers] [onlyIds] [tcost]
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
  /** generalisation diagnostic (held-out split) — never used for ranking */
  holdout?: number;
  level: number;
  formula: string;
  seed: number;
  iteration: number;
  tree?: unknown;
  speed?: { formulaCost: number; exactCost: number; speedup: number; vsIterative?: { label: string; speedup: number } };
  fast?: { formula: string; metric: number; level: number; formulaCost: number };
  fastTree?: unknown;
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
  const only = process.argv[5] ? new Set(process.argv[5].split(",")) : null;
  const tcost = Number(process.argv[6] ?? 0);

  // single-writer guard: concurrent farms would load the same ledger and
  // last-writer-wins over each other's records
  const lockPath = join(import.meta.dirname ?? ".", "..", ".farm-lock");
  if (existsSync(lockPath)) {
    console.error("✗ une ferme est déjà en cours (.farm-lock présent) — attends sa fin");
    process.exit(1);
  }
  writeFileSync(lockPath, String(process.pid));
  try {
    await farmInner(seed, budget, workers, only, tcost);
  } finally {
    rmSync(lockPath, { force: true });
  }
}

async function farmInner(seed: number, budget: number, workers: number, only: Set<string> | null, tcost: number): Promise<void> {

  // list all task ids (env unset here)
  delete process.env.SPEAR_TASKS;
  const { buildTasks } = await import("../src/lib/spear/benchmarks");
  const ids = buildTasks().map((t) => t.id).filter((id) => !only || only.has(id));
  const slices: string[][] = Array.from({ length: workers }, () => []);
  ids.forEach((id, i) => slices[i % workers].push(id));

  const ledgerPath = join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json");
  const ledger = loadLedger() as Ledger;

  const dir = mkdtempSync(join(tmpdir(), "spear-farm-"));
  const tsxCli = join(process.cwd(), "node_modules", "tsx", "dist", "cli.mjs");
  const childScript = join(import.meta.dirname ?? ".", "farm-child.ts");

  console.log(`▶ farm: ${workers} workers × ${ids.length} tâches, seed=${seed} budget=${budget}`);
  const t0 = performance.now();
  const procs = slices.map((slice, w) =>
    new Promise<string>((resolve, reject) => {
      const out = join(dir, `w${w}.json`);
      try {
        execFileSync(process.execPath, [tsxCli, childScript, out, String(seed), String(budget), slice.join(","), ...(tcost > 0 ? [String(tcost)] : [])], { stdio: ["ignore", "pipe", "pipe"] });
        resolve(out);
      } catch (e) { reject(e); }
    }),
  );
  const outs = await Promise.all(procs);
  const wall = (performance.now() - t0) / 1000;

  // A fast slot only means something if it is genuinely CHEAPER than the
  // champion it accompanies. The merge below can violate that in two ways: a
  // demoted ex-champion may not be cheaper than the incoming one, and a
  // carried-over `prev.fast` may be dearer than a newly improved champion.
  // Both shipped invalid slots to the ledger before (27 `fast-not-faster`
  // failures in one farm run), so the invariant is enforced here at the single
  // write point rather than patched afterwards by repair-ledger.
  // Tasks that HAVE a closed-form reference must never advertise a multiplier
  // against an iterative solver: the solver is not what a real implementation
  // runs, and the comparison is not accuracy-matched (gaussian_cdf advertised
  // ×2000 where the honest like-for-like ratio is ×1.48). The loop snapshot
  // still carries `vsIterative`, so strip it here at the write point.
  const { buildTasks: bt2 } = await import("../src/lib/spear/benchmarks");
  const hasClosedForm = new Set(
    (bt2() as any[]).filter((t) => t.exactCost !== undefined || t.exactRefNode !== undefined).map((t) => t.id),
  );
  const stripStrawman = (entry: Finding): void => {
    if (!hasClosedForm.has(entry.taskId)) return;
    if (entry.speed?.vsIterative) delete entry.speed.vsIterative;
    const fast = entry.fast as { vsIterative?: unknown } | undefined;
    if (fast?.vsIterative) delete fast.vsIterative;
  };

  const pruneFast = (entry: Finding): void => {
    if (!entry.fast) return;
    // A slot whose formula is literally the champion's is not an alternative
    // operating point (kv_cache shipped `A` as both champion and fast slot).
    if (entry.fast.formula === entry.formula) {
      delete entry.fast;
      delete entry.fastTree;
      return;
    }
    const champCost = entry.speed?.formulaCost;
    const fastCost = entry.fast.formulaCost;
    if (champCost === undefined || fastCost === undefined) return;
    if (fastCost >= champCost) {
      delete entry.fast;
      delete entry.fastTree;
    }
  };

  let records = 0;
  let totalBt = 0;
  for (const o of outs) {
    for (const f of JSON.parse(readFileSync(o, "utf8")) as (Finding & Record<string, unknown>)[]) {
      totalBt++;
      const prev = ledger[f.taskId];
      if (!prev || better(f.direction, f.metric, prev.metric)) {
        // demote the displaced champion to the fast slot if it is cheaper
        if (prev?.speed && f.speed && prev.speed.formulaCost < f.speed.formulaCost && (prev.level ?? 0) >= 2) {
          const demoted = { formula: prev.formula, metric: prev.metric, level: prev.level, formulaCost: prev.speed.formulaCost };
          if (!f.fast || demoted.formulaCost < f.fast.formulaCost) {
            f.fast = demoted;
            f.fastTree = prev.tree;
          }
        }
        pruneFast(f);
        stripStrawman(f);
        ledger[f.taskId] = f;
        records++;
      } else {
        if (prev.formula === f.formula) {
          // backfill missing annotations on reproduced champions
          if (!prev.speed && f.speed) prev.speed = f.speed;
          if (prev.speed && !prev.speed.vsIterative && f.speed?.vsIterative && !hasClosedForm.has(f.taskId)) prev.speed.vsIterative = f.speed.vsIterative;
          if (!prev.tree && f.tree) prev.tree = f.tree;
          if (prev.holdout === undefined && f.holdout !== undefined) prev.holdout = f.holdout;
        }
        // fast-slot: keep the cheapest VALIDATED form ever seen (record
        // ladder L>=2, or deployment grade R^2>=0.98 flagged `deploy`)
        if (f.fast && (f.fast.level >= 2 || (f.fast as { deploy?: boolean }).deploy) && f.fast.formulaCost < (prev.fast?.formulaCost ?? Infinity)) {
          prev.fast = f.fast;
          prev.fastTree = f.fastTree;
        }
        pruneFast(prev);
        stripStrawman(prev);
      }
    }
  }
  rmSync(dir, { recursive: true, force: true });
  saveLedger(ledger);

  console.log(`terminé en ${wall.toFixed(1)} s — ${records} nouveaux records sur ${totalBt} résultats`);
  for (const f of Object.values(ledger).sort((a, b) => a.taskId.localeCompare(b.taskId))) {
    const m = f.direction === "min" ? f.metric.toExponential(2) : f.metric.toFixed(1) + "%";
    const sp = f.speed?.speedup !== undefined ? ` ×${f.speed.speedup.toFixed(2)}` : "";
    console.log(`  [${f.taskId.padEnd(18)}] ${m.padStart(10)}  L${f.level}  seed ${f.seed}${sp}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
