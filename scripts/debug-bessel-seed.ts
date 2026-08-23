// Debug: is the long-series seed present & healthy in bessel pools?
process.env.SPEAR_TASKS = "bessel_j0";
async function main() {
  const { readFileSync } = await import("node:fs");
  const { parseNode, evaluateScalar } = await import("../src/lib/spear/engine");
  const led = JSON.parse(readFileSync("spear-hall-of-fame.json", "utf8"));
  const { buildTasks } = await import("../src/lib/spear/benchmarks");
  const def = buildTasks().find((t) => t.id === "bessel_j0")!;
  console.log("pool size:", def.seedPool?.length);
  const pool = def.seedPool ?? [];
  // find longest polynomial-looking seed and score it on dense grid
  let bestIdx = -1, bestMetric = Infinity;
  for (let i = 0; i < pool.length; i++) {
    const nd = pool[i];
    let m = NaN;
    try { m = def.evaluate(nd).metric; } catch {}
    if (m < bestMetric) { bestMetric = m; bestIdx = i; }
    if (i >= pool.length - 4) {
      console.log(`pool[${i}] mse=${Number.isFinite(m) ? m.toExponential(3) : "NaN"} str=${JSON.stringify(nd).slice(0, 90)}`);
    }
  }
  console.log(`meilleur seed: #${bestIdx} mse=${bestMetric.toExponential(4)}`);
  void parseNode; void evaluateScalar; void led;
}
main();
