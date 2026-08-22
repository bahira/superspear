// EDGE CPU BENCH — production activation kernels vs SPEAR algebraic forms,
// compiled to WASM, timed on raw CPU. Includes quality gates (max abs error
// vs the transcendental reference on the activation domain) and an FFN-block
// simulation showing the activation share of a transformer feed-forward pass.
// Run: npx tsx scripts/bench-edge.ts
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { makeNode, evaluateNode, parseNode, type SpearNode } from "../src/lib/spear/engine";
import { toWasmBytes, instantiateSpearWasm, collectWasmVars } from "../src/lib/spear/wasm";

type Node = SpearNode;
const v = (name: string) => makeNode("var", { name });
const c = (value: number) => makeNode("const", { value });
const bin = (op: "add" | "sub" | "mul" | "pdiv", a: Node, b: Node) => makeNode(op, { children: [a, b] });

// ---- production references (what ships today) ------------------------------
// SiLU: x / (1 + e^-x)                      [LLaMA/Mistral/Qwen SwiGLU]
const siluRef = bin("pdiv", v("x"), bin("add", c(1), makeNode("exp", { children: [makeNode("neg", { children: [v("x")] })] })));
// Softplus: ln(1 + e^x)
const softplusRef = makeNode("log", { children: [bin("add", c(1), makeNode("exp", { children: [v("x")] }))] });
// Sigmoid: 1 / (1 + e^-x)
const sigmoidRef = bin("pdiv", c(1), bin("add", c(1), makeNode("exp", { children: [makeNode("neg", { children: [v("x")] })] })));
// GELU-tanh: industry-standard erf-free approximation
//   0.5x(1 + tanh(z)), z = 0.797885(x + 0.044715x³); tanh(z)=1-2/(e^{2z}+1)
const z = bin("mul", c(0.7978845608), bin("add", v("x"), bin("mul", c(0.044715), makeNode("cube", { children: [v("x")] }))));
const tanhZ = bin("sub", c(1), bin("pdiv", c(2), bin("add", makeNode("exp", { children: [bin("mul", c(2), z)] }), c(1))));
const geluRef = bin("mul", bin("mul", c(0.5), v("x")), bin("add", c(1), tanhZ));

async function compile(node: Node): Promise<(x: number) => number> {
  const fn = await instantiateSpearWasm(Buffer.from(toWasmBytes(node)).toString("base64"));
  const names = collectWasmVars(node);
  return (x: number) => fn(names.length ? names.map(() => x) : []);
}

function bench(fn: (x: number) => number, data: Float64Array, rounds = 40): number {
  const args = [0];
  for (let w = 0; w < 3; w++) for (let i = 0; i < data.length; i++) { args[0] = data[i]; fn(data[i]); }
  const t0 = performance.now();
  for (let r = 0; r < rounds; r++) for (let i = 0; i < data.length; i++) { args[0] = data[i]; fn(data[i]); }
  return ((performance.now() - t0) * 1e6) / (rounds * data.length); // ns/elem
}

async function main() {
  const ledger = JSON.parse(readFileSync(join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json"), "utf8")) as Record<string, { tree?: unknown }>;
  const N = 200_000;
  const xs = new Float64Array(N);
  for (let i = 0; i < N; i++) xs[i] = -6 + 12 * ((i * 0.6180339887) % 1);

  const pairs: [string, Node, Node][] = [
    ["SiLU  (SwiGLU)", siluRef, parseTree(ledger, "silu")],
    ["GELU  (tanh-ref)", geluRef, parseTree(ledger, "gelu")],
    ["Softplus", softplusRef, parseTree(ledger, "softplus")],
    ["Sigmoid (fast-slot)", sigmoidRef, bin("pdiv", v("x"), bin("add", makeNode("abs", { children: [v("x")] }), c(1)))],
  ];

  console.log(`\nEDGE CPU BENCH — ${N} elems, WASM f64, domaine [-6,6]\n`);
  console.log("kernel            | ref ns/el | spear ns/el | gain   | err.max abs");
  console.log("------------------|-----------|-------------|--------|------------");
  for (const [label, refNode, spearNode] of pairs) {
    const refFn = await compile(refNode);
    const spFn = await compile(spearNode);
    const refNs = bench(refFn, xs);
    const spNs = bench(spFn, xs);
    // quality gate: max abs error vs float64 reference evaluation
    const refArr = evaluateNode(refNode, { x: xs }, N);
    const spArr = evaluateNode(spearNode, { x: xs }, N);
    let maxErr = 0;
    for (let i = 0; i < N; i++) maxErr = Math.max(maxErr, Math.abs(refArr[i] - spArr[i]));
    const gain = refNs / spNs;
    console.log(`${label.padEnd(17)} | ${refNs.toFixed(1).padStart(9)} | ${spNs.toFixed(1).padStart(11)} | ×${gain.toFixed(2).padStart(5)} | ${maxErr.toExponential(2)}`);
  }

  // ---- FFN block simulation: d=2048 -> ffn=5632 -> d=2048 ------------------
  console.log("\nFFN BLOCK (decode, d=2048, ffn=5632) — matmuls naïfs JS + activation");
  const d = 2048, ffn = 5632;
  const W1 = new Float64Array(ffn * d).map(() => Math.random() * 0.08 - 0.04);
  const W2 = new Float64Array(d * ffn).map(() => Math.random() * 0.08 - 0.04);
  const xin = new Float64Array(d).map(() => Math.random() * 2 - 1);
  const mid = new Float64Array(ffn);

  function matvec(W: Float64Array, rows: number, cols: number, x: Float64Array, out: Float64Array): void {
    for (let r = 0; r < rows; r++) {
      let s = 0;
      const base = r * cols;
      for (let k = 0; k < cols; k++) s += W[base + k] * x[k];
      out[r] = s;
    }
  }
  const siluSp = await compile(parseTree(ledger, "silu"));
  function ffnPass(act: (x: number) => number, actIsWasm: boolean): number {
    matvec(W1, ffn, d, xin, mid);
    if (actIsWasm) for (let i = 0; i < ffn; i++) mid[i] = act(mid[i]);
    else for (let i = 0; i < ffn; i++) mid[i] = act(mid[i]);
    const out = new Float64Array(d);
    matvec(W2, d, ffn, mid, out);
    return out[0];
  }

  // timings: activation alone (JS-evaluated exact vs wasm-spear), then block
  const tActRef0 = performance.now();
  for (let r = 0; r < 30; r++) for (let i = 0; i < ffn; i++) { const x = mid[i]; void (x / (1 + Math.exp(-x))); }
  const actRefNs = ((performance.now() - tActRef0) * 1e6) / (30 * ffn);
  for (let i = 0; i < ffn; i++) mid[i] = siluSp(mid[i]); // warm
  const tActSp0 = performance.now();
  for (let r = 0; r < 30; r++) for (let i = 0; i < ffn; i++) mid[i] = siluSp(mid[i]);
  const actSpNs = ((performance.now() - tActSp0) * 1e6) / (30 * ffn);

  const t0 = performance.now();
  for (let r = 0; r < 10; r++) ffnPass((x) => x / (1 + Math.exp(-x)), false);
  const blockRefMs = (performance.now() - t0) / 10;
  const t1 = performance.now();
  for (let r = 0; r < 10; r++) ffnPass(siluSp, true);
  const blockSpMs = (performance.now() - t1) / 10;

  console.log(`activation seule : ref ${actRefNs.toFixed(1)} ns/el -> spear(wasm) ${actSpNs.toFixed(1)} ns/el  (×${(actRefNs / actSpNs).toFixed(2)})`);
  console.log(`bloc complet     : ref ${blockRefMs.toFixed(2)} ms -> spear ${blockSpMs.toFixed(2)} ms  (${(((blockRefMs - blockSpMs) / blockRefMs) * 100).toFixed(1)} % plus rapide)`);
}

function parseTree(ledger: Record<string, { tree?: unknown }>, id: string): Node {
  const e = ledger[id];
  if (!e?.tree) throw new Error(`pas d'arbre pour ${id}`);
  return parseNode(e.tree as Parameters<typeof parseNode>[0]);
}

main().catch((e) => { console.error(e); process.exit(1); });
