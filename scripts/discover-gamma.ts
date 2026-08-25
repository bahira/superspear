// One-shot discovery driver: mirrors MCP server.ts "discover" handler.
// Target: display gamma boost x^(1/2.2) on [0,1], pure-ALU ops so the
// exported C vectorizes with SSE (no libm transcendentals).
import { evaluateNode, nodeToString, toC, estimateCost, evolve } from "../src/lib/spear/engine.js";
import { mse, linfError, r2Score } from "../src/lib/spear/math-utils.js";

function main() {
  const vName = "x";
  const lo = 0, hi = 1, rows = 2000;
  const xs = new Float64Array(rows), ys = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    xs[i] = lo + ((hi - lo) * ((i * 0.6180339887) % 1));
    ys[i] = Math.pow(xs[i], 1 / 2.2);
  }

  const result = evolve({
    variables: [vName],
    constRange: [-5, 5],
    ops: ["add", "sub", "mul", "pdiv", "relu", "abs", "neg", "sq", "cube", "sqrt", "max", "min"],
    maxDepth: 5,
  populationSize: 120,
  generations: 40,
  }, (node: any) => {
    try {
      const pred = evaluateNode(node, { [vName]: xs }, rows);
      let bad = false;
      for (let i = 0; i < rows; i++) if (!Number.isFinite(pred[i])) { bad = true; break; }
      if (bad) return { fitness: -1e9, size: node.size };
      const m = mse(pred, ys);
      if (!Number.isFinite(m)) return { fitness: -1e9, size: node.size };
      const li = linfError(pred, ys);
      return { fitness: -(m + 0.05 * li), size: node.size, extra: li };
    } catch {
      return { fitness: -1e9, size: node.size };
    }
  });

  const best = result.best;
  const pred = evaluateNode(best, { [vName]: xs }, rows);
  console.log(JSON.stringify({
    formula: nodeToString(best),
    cost: estimateCost(best),
    exact_powf_cost: 22, // per superspear benchmarks.ts: mul+add+powf(SFU~20)
    mse: mse(pred, ys),
    r2: r2Score(pred, ys),
    linf: linfError(pred, ys),
    generations_used: result.history.length,
    duration_ms: result.durationMs,
    c_export: toC(best, "spear_gamma_boost", "const float x"),
  }, null, 2));
}

main();
