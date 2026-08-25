// Generates packages/spear-kernels â€” a self-contained npm package exposing
// every kernel from spear-hall-of-fame.json as plain JS + CUDA C + PyTorch +
// WASM (base64) with metadata. Run: npx tsx scripts/gen-kernels-package.ts
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import {
  parseNode,
  toC,
  toPython,
  type SerializedNode,
  type SpearNode,
} from "../src/lib/spear/engine";
import { collectWasmVars, toWasmBytes } from "../src/lib/spear/wasm";

interface LedgerSpeed {
  formulaCost?: number;
  exactCost?: number;
  speedup?: number;
  vsIterative?: number;
}

interface LedgerEntry {
  metric?: number;
  level?: number;
  formula?: string;
  speed?: LedgerSpeed;
  tree?: SerializedNode;
  fast?: { speed?: LedgerSpeed; tree?: SerializedNode };
  fastTree?: SerializedNode;
}

// ---------------------------------------------------------------- AST -> JS
function astToJs(node: SpearNode): string {
  const walk = (nd: SpearNode): string => {
    switch (nd.op) {
      case "var": return nd.name;
      case "const": return JSON.stringify(nd.value);
      case "add": return `(${walk(nd.children[0])} + ${walk(nd.children[1])})`;
      case "sub": return `(${walk(nd.children[0])} - ${walk(nd.children[1])})`;
      case "mul": return `(${walk(nd.children[0])} * ${walk(nd.children[1])})`;
      // protected division â€” bit-faithful to evaluateNode: denominator floored
      // at Â±1e-4 (sign preserved), result clamped Â±1e4.
      case "pdiv": {
        const a = walk(nd.children[0]);
        const b = walk(nd.children[1]);
        const db = `((${b}) < 1e-4 && (${b}) > -1e-4 ? ((${b}) >= 0 ? 1e-4 : -1e-4) : (${b}))`;
        return `Math.min(1e4, Math.max(-1e4, (${a}) / ${db}))`;
      }
      case "relu": return `Math.max(0, ${walk(nd.children[0])})`;
      case "abs": return `Math.abs(${walk(nd.children[0])})`;
      case "neg": return `(-${walk(nd.children[0])})`;
      case "sq": { const a = walk(nd.children[0]); return `(${a}) * (${a})`; }
      case "cube": { const a = walk(nd.children[0]); return `(${a}) * (${a}) * (${a})`; }
      case "sqrt": return `Math.sqrt(Math.abs(${walk(nd.children[0])}))`;
      case "max": return `Math.max(${walk(nd.children[0])}, ${walk(nd.children[1])})`;
      case "min": return `Math.min(${walk(nd.children[0])}, ${walk(nd.children[1])})`;
      // clamp matches evaluateNode and every other backend (C/Py/WASM)
      case "exp": return `Math.exp(Math.max(-50, Math.min(50, ${walk(nd.children[0])})))`;
      case "sin": return `Math.sin(${walk(nd.children[0])})`;
      case "cos": return `Math.cos(${walk(nd.children[0])})`;
      case "atan": return `Math.atan(${walk(nd.children[0])})`;
      case "asin": return `Math.asin(Math.max(-1, Math.min(1, ${walk(nd.children[0])})))`;
      case "log": return `Math.log(Math.max(1e-30, ${walk(nd.children[0])}))`;
      default: throw new Error(`astToJs: op inconnu "${nd.op}"`);
    }
  };
  const vars = collectWasmVars(node);
  return `((${vars.join(", ")}) => ${walk(node)})`;
}

interface Variant {
  js: string;
  c: string;
  py: string;
  wasmBase64: string;
}

function buildVariant(node: SpearNode): Variant {
  return {
    js: astToJs(node),
    c: toC(node),
    py: toPython(node),
    wasmBase64: Buffer.from(toWasmBytes(node)).toString("base64"),
  };
}

function emitVariant(v: Variant, ind: string): string {
  return [
    `${ind}{`,
    `${ind}  js: ${JSON.stringify(v.js)},`,
    `${ind}  eval: ${v.js},`,
    `${ind}  c: ${JSON.stringify(v.c)},`,
    `${ind}  py: ${JSON.stringify(v.py)},`,
    `${ind}  wasmBase64: ${JSON.stringify(v.wasmBase64)},`,
    `${ind}},`,
  ].join("\n");
}

// ---------------------------------------------------------------- generate
const here = import.meta.dirname ?? ".";
const root = join(here, "..");
const pkgDir = join(root, "packages", "spear-kernels");
const ledger = JSON.parse(readFileSync(join(root, "spear-hall-of-fame.json"), "utf8")) as Record<string, LedgerEntry>;

const skipped: string[] = [];
const blocks: string[] = [];
let fastCount = 0;

for (const [id, entry] of Object.entries(ledger)) {
  if (!entry.tree) { skipped.push(id); continue; }
  const precise = buildVariant(parseNode(entry.tree));
  const fastSrc = entry.fastTree ?? entry.fast?.tree ?? null;
  let fastBlock = "";
  if (fastSrc) {
    fastCount++;
    fastBlock = `    fast: ${emitVariant(buildVariant(parseNode(fastSrc)), "    ")}\n`;
  }
  const s = entry.speed ?? {};
  blocks.push([
    `  "${id}": {`,
    `    id: "${id}",`,
    `    precise: ${emitVariant(precise, "    ")}`,
    fastBlock +
    `    meta: {`,
    `      metric: ${JSON.stringify(entry.metric ?? null)},`,
    `      level: ${JSON.stringify(entry.level ?? null)},`,
    `      formulaCost: ${JSON.stringify(s.formulaCost ?? null)},`,
    `      exactCost: ${JSON.stringify(s.exactCost ?? null)},`,
    `      speedupVsExact: ${JSON.stringify(s.speedup ?? null)},`,
    `      vsIterative: ${JSON.stringify(s.vsIterative ?? null)},`,
    `    },`,
    `  },`,
  ].join("\n"));
}

const ids = Object.keys(ledger).filter((id) => !skipped.includes(id));
mkdirSync(pkgDir, { recursive: true });

const header = "// Auto-generated by scripts/gen-kernels-package.ts â€” DO NOT EDIT.\n";
writeFileSync(
  join(pkgDir, "index.js"),
  header +
  "// Self-contained runtime: the arrow functions below ARE the kernels.\n" +
  "export const kernels = {\n" + blocks.join("\n") + "\n};\n\n" +
  `export const kernelIds = ${JSON.stringify(ids)};\n\n` +
  "export default kernels;\n",
);

writeFileSync(
  join(pkgDir, "index.cjs"),
  header +
  "'use strict';\n" +
  "const kernels = {\n" + blocks.join("\n") + "\n};\n\n" +
  `const kernelIds = ${JSON.stringify(ids)};\n\n` +
  "module.exports = { kernels, kernelIds, default: kernels };\n",
);

writeFileSync(join(pkgDir, "index.d.ts"), `/** One executable bundle for a discovered formula. */
export interface KernelVariant {
  /** Standalone arrow-function source, e.g. \`"((x) => ...)"\`. */
  js: string;
  /** The same arrow, precompiled and ready to call. */
  eval: (...args: number[]) => number;
  /** CUDA C (\`__device__\`) source. */
  c: string;
  /** PyTorch (\`torch.*\`) source. */
  py: string;
  /** Compiled WebAssembly binary (f64 params -> f64 result), base64-encoded. */
  wasmBase64: string;
}

export interface KernelMeta {
  /** Fitness achieved by the discovered formula (lower is better). */
  metric: number | null;
  level: number | null;
  formulaCost: number | null;
  exactCost: number | null;
  speedupVsExact: number | null;
  vsIterative: number | null;
}

export interface Kernel {
  id: string;
  precise: KernelVariant;
  fast?: KernelVariant;
  meta: KernelMeta;
}

export declare const kernels: Record<string, Kernel>;
export declare const kernelIds: string[];
declare const _default: Record<string, Kernel>;
export default _default;
`);

writeFileSync(join(pkgDir, "package.json"), JSON.stringify({
  name: "spear-kernels",
  version: "1.5.0",
  description: "Closed-form kernels discovered by the SPEAR symbolic regression engine, shipped as JS, CUDA C, PyTorch and WebAssembly.",
  license: "MIT",
  type: "module",
  main: "./index.cjs",
  module: "./index.js",
  types: "./index.d.ts",
  exports: {
    ".": {
      types: "./index.d.ts",
      import: "./index.js",
      require: "./index.cjs",
    },
    "./package.json": "./package.json",
  },
  files: ["index.js", "index.cjs", "index.d.ts", "README.md"],
  sideEffects: false,
}, null, 2) + "\n");

writeFileSync(join(pkgDir, "README.md"), `# spear-kernels

Every closed-form kernel discovered by the [SPEAR](https://github.com/) symbolic
regression engine, published as a zero-dependency, fully self-contained package.
Each kernel ships four interchangeable backends plus metadata:

| Field | What it is |
|---|---|
| \`precise.js\` / \`precise.eval\` | standalone JS arrow-function (source string / compiled) |
| \`precise.c\` | CUDA C \`\_\_device\_\_\` source |
| \`precise.py\` | PyTorch source |
| \`precise.wasmBase64\` | compiled f64 WebAssembly module |
| \`fast\` | algebraic-only variant when one was discovered |
| \`meta\` | \`metric\`, \`level\`, \`formulaCost\`, \`exactCost\`, \`speedupVsExact\`, \`vsIterative\` |

## Usage

\`\`\`js
import { kernels, kernelIds } from "spear-kernels";

kernelIds;                       // list of all kernel ids

kernels.silu.precise.eval(2);    // ~1.762 (SiLU approximant)

if (kernels.sigmoid.fast) {
  kernels.sigmoid.fast.eval(3);  // algebraic fast path
}

kernels.gaussian_cdf.meta.speedupVsExact; // metadata access
\`\`\`

### WebAssembly backend

\`\`\`js
const bytes = Uint8Array.from(atob(kernels.silu.precise.wasmBase64), (c) => c.charCodeAt(0));
const env = {
  exp: Math.exp, sin: Math.sin, cos: Math.cos, atan: Math.atan,
  log: (v) => Math.log(v > 1e-30 ? v : 1e-30),
};
const { instance } = await WebAssembly.instantiate(bytes, { env });
instance.exports.spear(2); // same protected-division semantics as the JS backend
\`\`\`

Division is **protected** exactly like the engine: denominators floored at
Â±1e-4 (sign-preserving), results clamped to Â±1e4. Discovered formulas rely on
these rails â€” do not swap in bare \`/\`.
`);

console.log(`spear-kernels generated: ${ids.length} kernels (${ids.length} precise, ${fastCount} fast)`);
console.log(`skipped: ${skipped.length ? skipped.join(", ") : "none"}`);
