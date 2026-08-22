// Fast headless run: budget 500, print all task breakthroughs + WASM parity.
import { runGroundedLoop, type LoopTaskSnapshot } from "./src/lib/spear/loop";
import { instantiateSpearWasm, toWasmBytes } from "./src/lib/spear/wasm";
import { evaluateScalar } from "./src/lib/spear/engine";

function node(op: string, children: unknown[] = [], value = 0, name = "") {
  return { op, children, value, name };
}

async function opParity(label: string, root: ReturnType<typeof node>, scope: Record<string, number>) {
  const bytes = toWasmBytes(root as never);
  const spear = await instantiateSpearWasm(Buffer.from(bytes).toString("base64"));
  const wasmOut = spear(Object.values(scope));
  const jsOut = evaluateScalar(root as never, scope);
  const ok = Number.isFinite(wasmOut) && Math.abs(wasmOut - jsOut) < 1e-9;
  console.log(`  ${ok ? "PASS" : "FAIL"} ${label.padEnd(22)} wasm=${wasmOut.toExponential(4)} js=${jsOut.toExponential(4)}`);
  if (!ok) throw new Error(`op parity failed: ${label}`);
}

async function wasmParity(snap: LoopTaskSnapshot): Promise<string> {
  if (!snap.wasm) return "no-wasm";
  if (snap.family !== "activation") return "n/a(regression)";
  if (!snap.chart?.length) return "no-chart";
  try {
    const spear = await instantiateSpearWasm(snap.wasm);
    let maxDiff = 0;
    for (const p of snap.chart) maxDiff = Math.max(maxDiff, Math.abs(spear([p.x]) - p.predicted));
    return maxDiff < 1e-9 ? `parity ${maxDiff.toExponential(0)}` : `PARITY-DRIFT ${maxDiff.toExponential(1)}`;
  } catch (e) {
    return `WASM-ERR: ${(e as Error).message}`;
  }
}

async function main() {
  console.log("=== op-by-op WASM parity (incl. sin/cos) ===");
  await opParity("sin", node("sin", [node("var", [], 0, "x")]), { x: 1.0 });
  await opParity("cos", node("cos", [node("var", [], 0, "x")]), { x: -0.5 });
  await opParity("exp+sin+cos", node("add", [node("sin", [node("var", [], 0, "x")]), node("cos", [node("mul", [node("var", [], 0, "x"), node("const", [], 2)])])]), { x: 0.7 });
  await opParity("exp clamp+sin", node("add", [node("exp", [node("var", [], 0, "x")]), node("sin", [node("var", [], 0, "x")])]), { x: 100 });

  console.log("\n▶ loop budget=500...");
  const t0 = Date.now();
  const progress = await runGroundedLoop({ seed: 1337, budget: 500, deadlineMs: 45_000 });
  console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)}s · status=${progress.status}\n`);

  const rows: { lvl: number; id: string; metric: string; level: string; formula: string; wasm: string; beaten: string }[] = [];

  for (const snap of progress.tasks) {
    const best = snap.best;
    if (!best) continue;
    const deployable = snap.baselines.filter((b) => b.kind !== "oracle" && b.kind !== "transcendental");
    const beaten = deployable.filter((b) => b.beaten).map((b) => b.name).join(", ") || "—";
    const wasm = await wasmParity(snap);
    rows.push({
      lvl: best.level,
      id: snap.taskId,
      metric: snap.metricLabel + "=" + best.metric.toExponential(2),
      level: `L${best.level}`,
      formula: best.formula,
      wasm,
      beaten,
    });
  }

  rows.sort((a, b) => b.lvl - a.lvl);
  for (const r of rows) {
    console.log(`[${r.level}] ${r.id.padEnd(20)} ${r.metric.padEnd(18)} wasm:${r.wasm.padEnd(18)} beaten:[${r.beaten}]`);
    console.log(`       ${r.formula}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});