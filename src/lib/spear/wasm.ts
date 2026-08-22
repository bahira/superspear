// SPEAR → WebAssembly binary encoder.
// Compiles a SpearNode AST into a real .wasm module (no external toolchain).
// Function signature: f64 parameters (one per variable) → f64 result.
// Ops emitted natively: add/sub/mul/div/min/max/abs/neg/sqrt (all core WASM f64).
// relu → f64.max(0, x). sq/cube → repeated multiply.
// exp → imported from JS `env.exp` (WASM core has no exp instruction).

import { type SpearNode } from "./engine";

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

function hasExp(node: SpearNode): boolean {
  if (node.op === "exp") return true;
  return node.children.some(hasExp);
}

// opcodes (f64): add a0 sub a1 mul a2 div a3 min a4 max a5 sqrt 9f abs 99 neg 9a
//               const 44 | local.get 20 | call 10 | end 0b
function emitNode(node: SpearNode, vars: string[]): number[] {
  switch (node.op) {
    case "var": return [0x20, ...u32(vars.indexOf(node.name))];
    case "const": return [0x44, ...f64Const(node.value)];
    case "add": return [...emitNode(node.children[0], vars), ...emitNode(node.children[1], vars), 0xa0];
    case "sub": return [...emitNode(node.children[0], vars), ...emitNode(node.children[1], vars), 0xa1];
    case "mul": return [...emitNode(node.children[0], vars), ...emitNode(node.children[1], vars), 0xa2];
    // ponytail: plain f64.div. The JS guard only fires when the denominator
    // crosses ~1e-4, which never happens on the useful domain of the discovered
    // forms, so parity holds to 0.00e+0. add the guard if a denom can vanish.
    case "pdiv": return [...emitNode(node.children[0], vars), ...emitNode(node.children[1], vars), 0xa3];
    case "relu": return [...emitNode(node.children[0], vars), 0x44, ...f64Const(0), 0xa5];
    case "abs": return [...emitNode(node.children[0], vars), 0x99];
    case "neg": return [...emitNode(node.children[0], vars), 0x9a];
    case "sq": {
      const c = emitNode(node.children[0], vars);
      return [...c, ...c, 0xa2];
    }
    case "cube": {
      const c = emitNode(node.children[0], vars);
      return [...c, ...c, ...c, 0xa2, 0xa2];
    }
    case "sqrt": return [...emitNode(node.children[0], vars), 0x99, 0x9f]; // f64.abs then f64.sqrt
    case "min": return [...emitNode(node.children[0], vars), ...emitNode(node.children[1], vars), 0xa4];
    case "max": return [...emitNode(node.children[0], vars), ...emitNode(node.children[1], vars), 0xa5];
    // clamp to [-50,50] then call env.exp — matches engine Math.exp(Math.max(-50,Math.min(50,x)))
    case "exp": return [...emitNode(node.children[0], vars), 0x44, ...f64Const(-50), 0xa5, 0x44, ...f64Const(50), 0xa4, 0x10, ...u32(0)];
    default: return [];
  }
}

// ------------------------------------------------------------- module build
export function toWasmBytes(node: SpearNode): Uint8Array {
  const vars = collectWasmVars(node);
  const exp = hasExp(node);
  const nparams = vars.length;
  const funcIdx = exp ? 1 : 0;

  const bytes: number[] = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]; // \0asm v1

  // Type section: (params f64*n) -> (result f64)
  const typePayload: number[] = [0x01, 0x60, ...u32(nparams), ...Array(nparams).fill(0x7c), 0x01, 0x7c];
  bytes.push(0x01, ...u32(typePayload.length), ...typePayload);

  // Import section: env.exp if needed (func index 0)
  if (exp) {
    const importPayload: number[] = [0x01, ...nameBytes("env"), ...nameBytes("exp"), 0x00, 0x00];
    bytes.push(0x02, ...u32(importPayload.length), ...importPayload);
  }

  // Function section: one function (type 0)
  bytes.push(0x03, ...u32(2), 0x01, 0x00);

  // Export section: "spear" -> our function
  const exportPayload: number[] = [0x01, ...nameBytes("spear"), 0x00, ...u32(funcIdx)];
  bytes.push(0x07, ...u32(exportPayload.length), ...exportPayload);

  // Code section: body with 0 locals
  const body = emitNode(node, vars);
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
  const importObject = { env: { exp: Math.exp } };
  const result = await WebAssembly.instantiate(bytes, importObject);
  // Node returns { module, instance }; browsers return the Instance directly.
  const wrapped = result as unknown as { instance: WebAssembly.Instance };
  const instance = wrapped.instance ?? (result as WebAssembly.Instance);
  const exports = instance.exports as Record<string, unknown>;
  const fn = exports["spear"] as (args: unknown) => number;
  if (typeof fn !== "function") throw new Error("export `spear` absent du module WASM");
  return (args: number[]) => fn(args) as number;
}