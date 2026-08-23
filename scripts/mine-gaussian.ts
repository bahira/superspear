// Mine the gauss_shader Pareto front: menu of shader-ready gaussian approximations.
async function main() {
  process.env.SPEAR_TASKS = "gauss_shader";
  const { runGroundedLoop } = await import("../src/lib/spear/loop");
  const { parseFormula, estimateCost } = await import("../src/lib/spear/engine");
  const { writeFileSync } = await import("node:fs");

  const t0 = Date.now();
  const progress = await runGroundedLoop({ seed: 994001, budget: 900, deadlineMs: 60_000 });
  const task = progress.tasks[0];
  if (!task) throw new Error("no gauss_shader task in progress");

  type Entry = { formula: string; metric: number; size: number; level: number; cost?: number };
  const front = ((task as unknown as { paretoFront?: Entry[] }).paretoFront ?? []).slice(0, 10);
  if (!front.length) throw new Error("empty paretoFront");

  for (const e of front) e.cost = estimateCost(parseFormula(e.formula));
  front.sort((a, b) => a.cost! - b.cost!);

  const rows = front.map(
    (e) =>
      `${String(e.cost).padStart(5)} | ${e.metric.toExponential(3).padStart(10)} | ${String(e.level).padStart(2)} | ${e.formula}`
  );
  const table = ["cost  |        MSE | lv | formula", ...rows].join("\n");

  const ready = front.filter((e) => e.cost <= 9 && e.metric <= 5e-3);
  const readyLines = ready.length
    ? ready.map((e) => `SHADER-READY: ${e.formula} (cost=${e.cost}, MSE=${e.metric.toExponential(3)})`).join("\n")
    : "SHADER-READY: none";

  console.log(table);
  console.log(readyLines);
  writeFileSync(
    "C:\\Users\\Yuri\\AppData\\Local\\Temp\\opencode\\gauss-menu.txt",
    `${table}\n${readyLines}\nruntime: ${((Date.now() - t0) / 1000).toFixed(1)}s\n`
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
