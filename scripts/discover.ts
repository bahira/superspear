// Official one-shot discovery CLI — mirrors the MCP server's `discover` tool.
//
// Usage:
//   npx tsx scripts/discover.ts "<formula>" <lo> <hi> [--alu] [--rows N] [--gens N]
//
//   npx tsx scripts/discover.ts "x^(1/2.2)" 0 1 --alu
//   npx tsx scripts/discover.ts "sin(x) * exp(-x / 3)" -2 8
//
// Prints a JSON report: discovered formula, cost vs target cost, MSE/R²/L∞,
// and the CUDA-C export. Add --alu to ban transcendental ops entirely.
import {
  ALL_OPS,
  ALGEBRAIC_OPS,
  estimateCost,
  evaluateNode,
  nodeToString,
  parseFormula,
  collectVarNames,
  evolve,
  toC,
} from "../src/lib/spear/engine";
import { mse, linfError, r2Score } from "../src/lib/spear/math-utils";

function main() {
  const args = process.argv.slice(2);
  const formula = args[0];
  const lo = Number(args[1]);
  const hi = Number(args[2]);
  const alu = args.includes("--alu");
  const rowsIdx = args.indexOf("--rows");
  const rows = rowsIdx !== -1 ? Number(args[rowsIdx + 1]) : 2000;
  const gensIdx = args.indexOf("--gens");
  const generations = gensIdx !== -1 ? Number(args[gensIdx + 1]) : 40;

  if (!formula || !Number.isFinite(lo) || !Number.isFinite(hi)) {
    console.error('usage: npx tsx scripts/discover.ts "<formula>" <lo> <hi> [--alu] [--rows N] [--gens N]');
    process.exit(1);
  }

  const target = parseFormula(formula);
  const vars = [...new Set(collectVarNames(target))];
  if (vars.length !== 1) {
    console.error(`une seule variable supportée pour l'instant (trouvé: ${vars.join(", ")})`);
    process.exit(1);
  }
  const vName = vars[0];

  const xs = new Float64Array(rows);
  const ys = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    xs[i] = lo + ((hi - lo) * ((i * 0.6180339887) % 1));
    ys[i] = evaluateNode(target, { [vName]: xs }, rows)[i];
  }

  const ops = alu ? ALL_OPS.filter((o) => ALGEBRAIC_OPS.has(o)) : ALL_OPS;
  const t0 = Date.now();
  const result = evolve(
    { variables: [vName], constRange: [-5, 5], ops, maxDepth: 5, populationSize: 120, generations },
    (node) => {
      try {
        const pred = evaluateNode(node, { [vName]: xs }, rows);
        for (let i = 0; i < rows; i++) if (!Number.isFinite(pred[i])) return { fitness: -1e9, size: node.size };
        const m = mse(pred, ys);
        if (!Number.isFinite(m)) return { fitness: -1e9, size: node.size };
        return { fitness: -(m + 0.05 * linfError(pred, ys)), size: node.size, extra: m };
      } catch {
        return { fitness: -1e9, size: node.size };
      }
    },
  );

  const best = result.best;
  const pred = evaluateNode(best, { [vName]: xs }, rows);
  console.log(JSON.stringify(
    {
      target: formula,
      domain: `[${lo}, ${hi}]`,
      discovered: nodeToString(best),
      cost_units: estimateCost(best),
      target_cost_units: estimateCost(target),
      speedup_vs_target: +(estimateCost(target) / Math.max(1, estimateCost(best))).toFixed(2),
      mse: mse(pred, ys),
      r2: r2Score(pred, ys),
      linf: linfError(pred, ys),
      pure_alu: alu,
      duration_ms: Date.now() - t0,
      c_export: toC(best, "spear_discovered", `const float ${vName}`),
    },
    null,
    2,
  ));
}

main();
