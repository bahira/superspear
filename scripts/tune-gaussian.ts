// Tune hand-shaped algebraic gaussian candidates against the gauss_shader
// evaluator using the engine's own constant refinement, then rank by
// (cost <= 9, MSE). Goal: a shader-ready exp-free kernel for variable blur.
// Run: npx tsx scripts/tune-gaussian.ts
process.env.SPEAR_TASKS = "gauss_shader";

async function main() {
  const { buildTasks } = await import("../src/lib/spear/benchmarks");
  const { makeNode, parseFormula, estimateCost, nodeToString, refineConstants } = await import("../src/lib/spear/engine");
  const def = buildTasks().find((t) => t.id === "gauss_shader")!;
  const v = "x";
  const c = (n: number) => makeNode("const", { value: n });
  const bin = (op: "add" | "sub" | "mul" | "pdiv", a: any, b: any) => makeNode(op, { children: [a, b] });
  const X = makeNode("var", { name: v });
  const x2 = makeNode("sq", { children: [X] });

  // candidate shapes (exp-free), constants are refinement starting points
  const shapes: [string, any][] = [
    ["student-t k2", bin("pdiv", c(1), makeNode("sq", { children: [bin("add", bin("mul", x2, c(0.25)), c(1))] }))],
    ["student-t k3", bin("pdiv", c(1), makeNode("cube", { children: [bin("add", bin("mul", x2, c(0.18)), c(1))] }))],
    ["pade 12/12", bin("pdiv", bin("add", c(12), bin("mul", x2, c(-5))), bin("add", c(12), bin("mul", x2, c(7))))],
    ["rational quartic", bin("pdiv", bin("add", c(1), bin("mul", x2, c(-0.45))), bin("add", c(1), bin("mul", x2, c(0.21))))],
    ["cos window", makeNode("cos", { children: [bin("mul", X, c(0.4))] })],
    ["tanh-sq", bin("sub", c(1), makeNode("sq", { children: [makeNode("relu", { children: [bin("sub", bin("mul", makeNode("abs", { children: [X] }), c(0.35)), c(0))] })] }))],
  ];

  console.log("shape            | cout tune | MSE tune     | formule raffinee");
  console.log("-----------------|-----------|--------------|----------------------------------");
  const rows: { name: string; cost: number; mse: number; str: string }[] = [];
  for (const [name, shape] of shapes) {
    try {
      const res = refineConstants(shape, (nd: any) => {
        const r = def.evaluate(nd);
        return Number.isFinite(r.metric) ? r.metric : 1e9;
      }, 300);
      const mse = def.evaluate(res.node).metric;
      const cost = estimateCost(res.node);
      rows.push({ name, cost, mse, str: nodeToString(res.node) });
      console.log(`${name.padEnd(16)} | ${String(cost).padStart(9)} | ${mse.toExponential(4).padStart(12)} | ${nodeToString(res.node).slice(0, 70)}`);
    } catch (e) {
      console.log(`${name.padEnd(16)} | ERREUR ${String(e).slice(0, 60)}`);
    }
  }

  console.log("\nSHADER-READY (cout <= 9 && MSE <= 5e-3):");
  let any = false;
  for (const r of rows.filter((r) => r.cost <= 9 && r.mse <= 5e-3).sort((a, b) => a.mse - b.mse)) {
    any = true;
    console.log(`  [${r.name}] cout ${r.cost}, MSE ${r.mse.toExponential(3)}\n    ${r.str}`);
  }
  if (!any) {
    const best = rows.filter((r) => r.cost <= 9).sort((a, b) => a.mse - b.mse)[0];
    if (best) console.log(`  aucun strict — meilleur sous-9-unités: [${best.name}] MSE ${best.mse.toExponential(3)}\n    ${best.str}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
