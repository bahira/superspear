// Differential evaluator: float64 chain vs Math.fround(float32) chain,
// per-node, to locate where C(float32) diverges from wasm/JS(float64).
// Run: npx tsx scripts/probe-f32.ts [taskId=lennard_jones] [r=-1.6]
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseNode, nodeToString, type SpearNode } from "../src/lib/spear/engine";

type Val = { f64: number; f32: number };
const fr = Math.fround;

function ev(node: SpearNode, scope: Record<string, number>): Val {
  const kids = () => node.children.map((c) => ev(c, scope));
  switch (node.op) {
    case "var": { const v = scope[node.name]; return { f64: v, f32: fr(v) }; }
    case "const": return { f64: node.value, f32: fr(node.value) };
    case "add": { const k = kids(); return { f64: k[0].f64 + k[1].f64, f32: fr(fr(k[0].f32) + fr(k[1].f32)) }; }
    case "sub": { const k = kids(); return { f64: k[0].f64 - k[1].f64, f32: fr(fr(k[0].f32) - fr(k[1].f32)) }; }
    case "mul": { const k = kids(); return { f64: k[0].f64 * k[1].f64, f32: fr(fr(k[0].f32) * fr(k[1].f32)) }; }
    case "pdiv": {
      const k = kids();
      const step = (num: number, den: number): number => {
        let d = den;
        if (d > -1e-4 && d < 1e-4) d = d >= 0 ? 1e-4 : -1e-4;
        return Math.max(-1e4, Math.min(1e4, num / d));
      };
      return { f64: step(k[0].f64, k[1].f64), f32: fr(step(k[0].f32, k[1].f32)) };
    }
    case "neg": { const k = kids(); return { f64: -k[0].f64, f32: fr(-k[0].f32) }; }
    case "abs": { const k = kids(); return { f64: Math.abs(k[0].f64), f32: fr(Math.abs(k[0].f32)) }; }
    case "sq": { const k = kids(); return { f64: k[0].f64 * k[0].f64, f32: fr(k[0].f32 * k[0].f32) }; }
    case "cube": { const k = kids(); const a = k[0]; return { f64: a.f64 * a.f64 * a.f64, f32: fr(fr(fr(a.f32) * fr(a.f32)) * fr(a.f32)) }; }
    case "sqrt": { const k = kids(); const a = Math.abs(k[0].f64); const b = Math.abs(k[0].f32); return { f64: Math.sqrt(a), f32: fr(Math.sqrt(b)) }; }
    case "relu": { const k = kids(); return { f64: k[0].f64 > 0 ? k[0].f64 : 0, f32: fr(k[0].f32 > 0 ? k[0].f32 : 0) }; }
    case "exp": { const k = kids(); const c = (x: number) => Math.max(-50, Math.min(50, x)); return { f64: Math.exp(c(k[0].f64)), f32: fr(Math.exp(c(k[0].f32))) }; }
    case "sin": { const k = kids(); return { f64: Math.sin(k[0].f64), f32: fr(Math.sin(k[0].f32)) }; }
    case "cos": { const k = kids(); return { f64: Math.cos(k[0].f64), f32: fr(Math.cos(k[0].f32)) }; }
    case "log": { const k = kids(); const g = (x: number) => (x > 1e-30 ? x : 1e-30); return { f64: Math.log(g(k[0].f64)), f32: fr(Math.log(g(k[0].f32))) }; }
    case "atan": { const k = kids(); return { f64: Math.atan(k[0].f64), f32: fr(Math.atan(k[0].f32)) }; }
    case "asin": { const k = kids(); const g = (x: number) => Math.max(-1, Math.min(1, x)); return { f64: Math.asin(g(k[0].f64)), f32: fr(Math.asin(g(k[0].f32))) }; }
    case "tanh": { const k = kids(); return { f64: Math.tanh(k[0].f64), f32: fr(Math.tanh(k[0].f32)) }; }
    case "acos": { const k = kids(); const g = (x: number) => Math.max(-1, Math.min(1, x)); return { f64: Math.acos(g(k[0].f64)), f32: fr(Math.acos(g(k[0].f32))) }; }
    case "max": { const k = kids(); return { f64: Math.max(k[0].f64, k[1].f64), f32: fr(Math.max(k[0].f32, k[1].f32)) }; }
    case "min": { const k = kids(); return { f64: Math.min(k[0].f64, k[1].f64), f32: fr(Math.min(k[0].f32, k[1].f32)) }; }
    default: return { f64: NaN, f32: NaN };
  }
}

function main() {
  const id = process.argv[2] ?? "lennard_jones";
  const xName = process.argv[3] ?? "r";
  const xVal = Number(process.argv[4] ?? -1.6);
  const ledger = JSON.parse(readFileSync(join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json"), "utf8")) as Record<string, { tree?: unknown }>;
  const node = parseNode(ledger[id].tree as never);

  // collect var names in tree order
  const names: string[] = [];
  const walk = (n: SpearNode) => { if (n.op === "var" && !names.includes(n.name)) names.push(n.name); n.children.forEach(walk); };
  walk(node);
  const scope: Record<string, number> = {};
  for (const n of names) scope[n] = n === xName ? xVal : 0.7;

  console.log("formula:", nodeToString(node));
  console.log("vars:", JSON.stringify(scope));
  const report = (n: SpearNode, path: string): void => {
    const v = ev(n, scope);
    const rel = Math.abs(v.f64 - v.f32) / (Math.abs(v.f64) + 1e-300);
    if (rel > 1e-4 || !Number.isFinite(v.f32)) {
      console.log(`${path.padEnd(6)} ${n.op.padEnd(6)} f64=${v.f64.toExponential(8)}  f32=${v.f32.toExponential(8)}  rel=${rel.toExponential(2)}  [${nodeToString(n).slice(0, 60)}]`);
    }
    n.children.forEach((c, i) => report(c, `${path}.${i}`));
  };
  report(node, "root");
}

main();
