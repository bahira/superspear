// Farm child: runs the grounded loop on ONE subset of tasks and writes a
// partial result JSON. Env SPEAR_TASKS must be set BEFORE module import,
// hence the dynamic imports.
async function main() {
  const outPath = process.argv[2];
  const seed = Number(process.argv[3]);
  const budget = Number(process.argv[4]);
  process.env.SPEAR_TASKS = process.argv[5];

  const { runGroundedLoop } = await import("../src/lib/spear/loop");
  const progress = await runGroundedLoop({ seed, budget, deadlineMs: 45_000 });
  const partial = progress.tasks
    .filter((t) => t.best)
    .map((t) => ({
      taskId: t.taskId,
      title: t.title,
      direction: t.metricDirection,
      metric: t.best!.metric,
      level: t.best!.level,
      formula: t.best!.formula,
      seed,
      iteration: t.iterations,
      speed: t.speed ? { formulaCost: t.speed.formulaCost, exactCost: t.speed.exactCost, speedup: t.speed.estimatedSpeedup } : undefined,
      tree: t.tree ?? undefined,
    }));
  writeFileSync(outPath, JSON.stringify(partial));
  console.log(`[child ${process.pid}] ${partial.length} tâches, status=${progress.status}, iters=${progress.iterationsUsed}`);
}

import { writeFileSync } from "node:fs";
main().catch((e) => { console.error(e); process.exit(1); });
