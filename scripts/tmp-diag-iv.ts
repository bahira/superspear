import { buildTasks } from "../src/lib/spear/benchmarks";
import { evaluateNode } from "../src/lib/spear/engine";
import { mse } from "../src/lib/spear/math-utils";

async function main() {
  for (const id of ["implied_vol", "pendulum_hybrid"]) {
    const task = buildTasks().find((t) => t.id === id)!;
    const def = task as unknown as {
      variables: string[];
      evaluate: (n: any) => { metric: number };
      seedPool: any[];
      holdout?: any;
    };
    console.log(`=== ${id} — seedPool ${def.seedPool.length} ===`);
    const scored = def.seedPool
      .map((s, i) => ({ i, formula: JSON.stringify(s), metric: def.evaluate(s) }))
      .sort((a, b) => a.metric.metric - b.metric.metric);
    for (const s of scored.slice(0, 10)) {
      console.log(`seed#${s.i}  MSE=${s.metric.metric.toExponential(3)}  ${s.formula.slice(0, 90)}`);
    }
    console.log("--- mes extraSeeds (indices de fin) ---");
    for (let i = Math.max(0, def.seedPool.length - 6); i < def.seedPool.length; i++) {
      const m = def.evaluate(def.seedPool[i]);
      console.log(`seed#${i}  MSE=${m.metric.toExponential(3)}  ${JSON.stringify(def.seedPool[i]).slice(0, 90)}`);
    }
  }
}
main().catch((e) => { console.error(e); process.exit(1); });