// WASM↔JS parity smoke test across every ledger champion with an AST.
// Catches codegen drift: import type signatures, protected-division rails,
// opcode swaps. Run: npx tsx scripts/test-wasm-parity.ts
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseNode, evaluateScalar } from "../src/lib/spear/engine";
import { toWasmBytes, instantiateSpearWasm, collectWasmVars } from "../src/lib/spear/wasm";

interface LedgerEntry { tree?: unknown }
type Ledger = Record<string, LedgerEntry>;

// deterministic probe points per task id (var -> value); any finite points work
const PROBES: Record<string, Record<string, number>> = {
  rope_rot: { x: 1.3, y: -0.7, th: 2.1 },
  pendulum_hybrid: { th: 0.5, d: 1 },
  kerr_spin: { b: 12, s: 0.4 },
  kerr: { b: 12, s: 0.4 },
  gaussian_cdf: { x: 0.8 },
  eigen3_sym: { t: 1.5, u: 0.5, w: -0.2 },
  ik_reach: { d: 3, l2: 2, l3: 2 },
  gemv4: { x0: 1, x1: -1, x2: 0.5, x3: 2 },
  kdv_soliton: { x: 2.5, t: 1.5 },
  gaussian_kernel: { x: 0.9 },
  silu: { x: 1.7 },
  softplus: { x: 0.6 },
};

async function main() {
  const ledger = JSON.parse(readFileSync(join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json"), "utf8")) as Ledger;
  let checked = 0;
  let failed = 0;
  for (const [id, entry] of Object.entries(ledger)) {
    if (!entry.tree) continue;
    const node = parseNode(entry.tree as never);
    const names = collectWasmVars(node);
    // probe point: use curated values when available, else all-vars = 0.7
    const scope: Record<string, number> = {};
    for (const v of names) scope[v] = PROBES[id]?.[v] ?? 0.7;
    try {
      const fn = await instantiateSpearWasm(Buffer.from(toWasmBytes(node)).toString("base64"));
      const js = evaluateScalar(node, scope);
      const wv = fn(names.map((k) => scope[k]));
      const rel = Math.abs(js - wv) / (Math.abs(js) + 1e-12);
      checked++;
      if (!(rel <= 1e-9)) {
        failed++;
        console.log(`✗ [${id}] rel=${rel.toExponential(2)} js=${js.toExponential(4)} wasm=${wv.toExponential(4)}`);
      } else {
        console.log(`✓ [${id}]`);
      }
    } catch (e) {
      failed++;
      console.log(`✗ [${id}] compile/instantiate: ${String(e).slice(30, 110)}`);
    }
  }
  console.log(`\nparity: ${checked - failed}/${checked} OK`);
  if (failed > 0) process.exit(1);
}

main().catch((e) => { console.error(e); process.exit(1); });
