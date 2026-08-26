// SPEAR â†’ WebAssembly binary encoder.
// Compiles a SpearNode AST into a real .wasm module (no external toolchain).
// Function signature: f64 parameters (one per variable) â†’ f64 result.
// Ops emitted natively: add/sub/mul/div/min/max/abs/neg/sqrt (all core WASM f64).
// relu â†’ f64.max(0, x). sq/cube â†’ repeated multiply.
// exp â†’ imported from JS `env.exp` (WASM core has no exp instruction).

import { type SpearNode } from "./engine";
import { erf } from "./math-utils";

// ------------------------------------------------------------- tiny encoders
function u32(v: number): number[] {
  const out: number[] = [];
  let x = v >>> 0;
  do {
    let b = x & 0x7f;
    x >>>= 7;
    if (x) b |= 0x80;
    out.push(b);
  } while (x);
  return out;
}

function f64Const(v: number): number[] {
  const buf = new ArrayBuffer(8);
  new DataView(buf).setFloat64(0, v, true);
  return [...new Uint8Array(buf)];
}

function nameBytes(s: string): number[] {
  const b = [...s].map((c) => c.charCodeAt(0) & 0xff);
  return [b.length, ...b];
}

// ---------------------------------------------------------------- traversal
export function collectWasmVars(node: SpearNode): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  const walk = (n: SpearNode) => {
    if (n.op === "var" && !seen.has(n.name)) {
      seen.add(n.name);
      out.push(n.name);
    }
    n.children.forEach(walk);
  };
  walk(node);
  return out;
}

const IMPORTABLE: { op: string; name: string }[] = [
  { op: "exp", name: "exp" },
  { op: "sin", name: "sin" },
  { op: "cos", name: "cos" },
  { op: "atan", name: "atan" },
  { op: "asin", name: "asin" },
  { op: "log", name: "log" },
  { op: "tanh", name: "tanh" },
  { op: "acos", name: "acos" },
  { op: "erf", name: "erf" },
];

function neededImports(node: SpearNode): { op: string; name: string }[] {
  const used = new Set<string>();
  const walk = (n: SpearNode) => {
    if (n.op === "exp" || n.op === "sin" || n.op === "cos" || n.op === "log" || n.op === "atan" || n.op === "asin" || n.op === "tanh" || n.op === "acos" || n.op === "erf") used.add(n.op);
    n.children.forEach(walk);
  };
  walk(node);
  return IMPORTABLE.filter((imp) => used.has(imp.op));
}

// opcodes (f64): add a0 sub a1 mul a2 div a3 min a4 max a5 sqrt 9f abs 99 neg 9a
//               const 44 | local.get 20 | call 10 | end 0b
function emitNode(node: SpearNode, vars: string[], importIdx: Record<string, number>): number[] {
  switch (node.op) {
    case "var": return [0x20, ...u32(vars.indexOf(node.name))];
    case "const": return [0x44, ...f64Const(node.value)];
    case "add": return [...emitNode(node.children[0], vars, importIdx), ...emitNode(node.children[1], vars, importIdx), 0xa0];
    case "sub": return [...emitNode(node.children[0], vars, importIdx), ...emitNode(node.children[1], vars, importIdx), 0xa1];
    case "mul": return [...emitNode(node.children[0], vars, importIdx), ...emitNode(node.children[1], vars, importIdx), 0xa2];
    // protected division, bit-faithful to evaluateNode: denominators are
    // floored at Â±1e-4 (sign preserved via copysign) and results clamped to
    // Â±1e4. Discovered forms DO lean on these rails (gaussian_cdf plateau).
    // ponytail: child emitted twice to duplicate without locals â€” pure reads.
    case "pdiv":
      return [
        ...emitNode(node.children[0], vars, importIdx),
        ...emitNode(node.children[1], vars, importIdx), 0x99, 0x44, ...f64Const(1e-4), 0xa5,
        ...emitNode(node.children[1], vars, importIdx), 0xa6,
        0xa3,
        0x44, ...f64Const(-1e4), 0xa5,
        0x44, ...f64Const(1e4), 0xa4,
      ];
    case "relu": return [...emitNode(node.children[0], vars, importIdx), 0x44, ...f64Const(0), 0xa5];
    case "abs": return [...emitNode(node.children[0], vars, importIdx), 0x99];
    case "neg": return [...emitNode(node.children[0], vars, importIdx), 0x9a];
    case "sq": {
      const c = emitNode(node.children[0], vars, importIdx);
      return [...c, ...c, 0xa2];
    }
    case "cube": {
      const c = emitNode(node.children[0], vars, importIdx);
      return [...c, ...c, ...c, 0xa2, 0xa2];
    }
    case "sqrt": return [...emitNode(node.children[0], vars, importIdx), 0x99, 0x9f]; // f64.abs then f64.sqrt
    case "min": return [...emitNode(node.children[0], vars, importIdx), ...emitNode(node.children[1], vars, importIdx), 0xa4];
    case "max": return [...emitNode(node.children[0], vars, importIdx), ...emitNode(node.children[1], vars, importIdx), 0xa5];
    // clamp to [-50,50] then call env.exp â€” matches engine Math.exp(Math.max(-50,Math.min(50,x)))
    case "exp": return [...emitNode(node.children[0], vars, importIdx), 0x44, ...f64Const(-50), 0xa5, 0x44, ...f64Const(50), 0xa4, 0x10, ...u32(importIdx.exp)];
    case "sin": return [...emitNode(node.children[0], vars, importIdx), 0x10, ...u32(importIdx.sin)];
    case "atan": return [...emitNode(node.children[0], vars, importIdx), 0x10, ...u32(importIdx.atan)];
    // clamp to [-1,1] then call env.asin — matches engine Math.asin(clamp(x,-1,1))
    case "asin": return [...emitNode(node.children[0], vars, importIdx), 0x44, ...f64Const(-1), 0xa5, 0x44, ...f64Const(1), 0xa4, 0x10, ...u32(importIdx.asin)];
    case "cos": return [...emitNode(node.children[0], vars, importIdx), 0x10, ...u32(importIdx.cos)];
    case "tanh": return [...emitNode(node.children[0], vars, importIdx), 0x10, ...u32(importIdx.tanh)];
    // clamp to [-1,1] then call env.acos — matches engine Math.acos(clamp(x,-1,1))
    case "acos": return [...emitNode(node.children[0], vars, importIdx), 0x44, ...f64Const(-1), 0xa5, 0x44, ...f64Const(1), 0xa4, 0x10, ...u32(importIdx.acos)];
    case "erf": return [...emitNode(node.children[0], vars, importIdx), 0x10, ...u32(importIdx.erf)];
    // clamp low bound then call env.log â€” matches engine Math.log(max(x, 1e-30))
    case "log": return [...emitNode(node.children[0], vars, importIdx), 0x44, ...f64Const(1e-30), 0xa5, 0x10, ...u32(importIdx.log)];
    default: return [];
  }
}

// ------------------------------------------------------------- module build
export function toWasmBytes(node: SpearNode): Uint8Array {
  const vars = collectWasmVars(node);
  const imports = neededImports(node);
  const importIdx: Record<string, number> = {};
  imports.forEach((imp, i) => (importIdx[imp.op] = i));
  const nparams = vars.length;
  const funcIdx = imports.length; // each import occupies a function index

  const bytes: number[] = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]; // \0asm v1

  // Type section: type 0 = (params f64*n) -> (result f64) for the main
  // function; type 1 = (f64) -> (f64) for imported transcendental calls.
  // BUGFIX: imports previously referenced type 0, so multi-var modules with
  // sin/cos/exp declared imports expecting n params â†’ "not enough arguments".
  const unaryType: number[] = [0x60, ...u32(1), 0x7c, 0x01, 0x7c];
  const mainType: number[] = [0x60, ...u32(nparams), ...Array(nparams).fill(0x7c), 0x01, 0x7c];
  const typePayload: number[] = [0x02, ...mainType, ...unaryType];
  bytes.push(0x01, ...u32(typePayload.length), ...typePayload);

  // Import section: env.{exp,sin,cos} as needed (func indices 0..k)
  if (imports.length > 0) {
    const importPayload: number[] = [imports.length];
    for (const imp of imports) {
      importPayload.push(...nameBytes("env"), ...nameBytes(imp.name), 0x00, 0x01);
    }
    bytes.push(0x02, ...u32(importPayload.length), ...importPayload);
  }

  // Function section: one function (type 0)
  bytes.push(0x03, ...u32(2), 0x01, 0x00);

  // Export section: "spear" -> our function
  const exportPayload: number[] = [0x01, ...nameBytes("spear"), 0x00, ...u32(funcIdx)];
  bytes.push(0x07, ...u32(exportPayload.length), ...exportPayload);

  // Code section: body with 0 locals
  const body = emitNode(node, vars, importIdx);
  const code = [...u32(0), ...body, 0x0b];
  const codePayload: number[] = [0x01, ...u32(code.length), ...code];
  bytes.push(0x0a, ...u32(codePayload.length), ...codePayload);

  return new Uint8Array(bytes);
}

// ------------------------------------------------------------ client helpers
export function wasmBytesFromB64(b64: string): Uint8Array {
  return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}

/** Instantiate the compiled module. Returns the exported `spear` callable. */
export async function instantiateSpearWasm(
  b64: string,
): Promise<(args: number[]) => number> {
  const bytes = wasmBytesFromB64(b64);
  const importObject = { env: { exp: Math.exp, sin: Math.sin, cos: Math.cos, atan: Math.atan, asin: (v: number) => Math.asin(Math.max(-1, Math.min(1, v))), log: (v: number) => Math.log(v > 1e-30 ? v : 1e-30), tanh: Math.tanh, acos: (v: number) => Math.acos(Math.max(-1, Math.min(1, v))), erf } };
  const result = await WebAssembly.instantiate(bytes, importObject);
  // Node returns { module, instance }; browsers return the Instance directly.
  const wrapped = result as unknown as { instance: WebAssembly.Instance };
  const instance = wrapped.instance ?? (result as WebAssembly.Instance);
  const exports = instance.exports as Record<string, unknown>;
  const fn = exports["spear"] as (...a: number[]) => number;
  if (typeof fn !== "function") throw new Error("export `spear` absent du module WASM");
  // spread positional f64 params â€” passing an array would coerce to a single
  // (NaN for >1 element) value and silently break every multi-var module
  return (args: number[]) => fn(...args);
}
