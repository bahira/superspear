// Per-pixel gaussian kernel speed: exp reference vs SPEAR student-t k3,
// both compiled to WASM. The variable-blur use case evaluates this per pixel.
// Run: npx tsx scripts/bench-gaussian.ts
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { makeNode } from "../src/lib/spear/engine";
import { toWasmBytes, instantiateSpearWasm } from "../src/lib/spear/wasm";

const v = (n: string) => makeNode("var", { name: n });
const c = (x: number) => makeNode("const", { value: x });
const bin = (op: "add" | "sub" | "mul" | "pdiv", a: any, b: any) => makeNode(op, { children: [a, b] });

// reference: e^(-x²/2)
const ref = makeNode("exp", { children: [bin("mul", c(-0.5), makeNode("sq", { children: [v("x")] }))] });
// SPEAR: (1.02232 / (0.207x² + 1))³
const spear = bin("pdiv", c(1.02232), makeNode("cube", { children: [bin("add", bin("mul", makeNode("sq", { children: [v("x")] }), c(0.207)), c(1))] }));

async function compile(node: any): Promise<(x: number) => number> {
  const fn = await instantiateSpearWasm(Buffer.from(toWasmBytes(node)).toString("base64"));
  return (x: number) => fn([x]);
}
function bench(fn: (x: number) => number, data: Float64Array, rounds = 50): number {
  for (let w = 0; w < 3; w++) for (let i = 0; i < data.length; i++) fn(data[i]);
  const t0 = performance.now();
  for (let r = 0; r < rounds; r++) for (let i = 0; i < data.length; i++) fn(data[i]);
  return ((performance.now() - t0) * 1e6) / (rounds * data.length);
}

async function main() {
  void readFileSync; void join;
  const N = 200_000;
  const xs = new Float64Array(N);
  for (let i = 0; i < N; i++) xs[i] = -4 + 8 * ((i * 0.6180339887) % 1);
  const refFn = await compile(ref);
  const spFn = await compile(spear);
  const refNs = bench(refFn, xs);
  const spNs = bench(spFn, xs);
  let maxErr = 0;
  for (let i = 0; i < N; i++) maxErr = Math.max(maxErr, Math.abs(refFn(xs[i]) - spFn(xs[i])));
  console.log(`GAUSSIENNE PAR PIXEL ([-4,4], ${N} pts, WASM f64)`);
  console.log(`  ref exp : ${refNs.toFixed(1)} ns/el`);
  console.log(`  SPEAR   : ${spNs.toFixed(1)} ns/el   gain ×${(refNs / spNs).toFixed(2)}   err.max ${maxErr.toExponential(2)}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
