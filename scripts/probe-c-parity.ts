// Reproduce export-audit's C↔WASM parity flow for ONE task, verbosely.
// Shows whether the emitted C carries the protected division and where the
// first divergence happens (audit uses all-equal probe values per call).
// Run: npx tsx scripts/probe-c-parity.ts [taskId=kerr_spin]
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { execFileSync } from "node:child_process";
import { parseNode, evaluateNode, toMisraC } from "../src/lib/spear/engine";
import { toWasmBytes, instantiateSpearWasm, collectWasmVars } from "../src/lib/spear/wasm";

async function main() {
  const id = process.argv[2] ?? "kerr_spin";
  const ledger = JSON.parse(readFileSync(join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json"), "utf8")) as Record<string, { tree?: unknown }>;
  const entry = ledger[id];
  if (!entry?.tree) { console.log(`no tree for ${id}`); return; }
  const node = parseNode(entry.tree as never);
  const vars = collectWasmVars(node);

  const c99 = toMisraC(node, `spear_${id}`, `const float32_t ${vars.join(", const float32_t ")}`);
  console.log(`copysignf in emitted C: ${c99.includes("copysignf")}`);

  // audit-style probes: every variable gets the SAME value
  const P = 12;
  const vals = Array.from({ length: P }, (_, i) => -6 + (12 * i) / (P - 1));
  const calls = vals
    .map((v) => `    printf("%.17g\\n", (double)spear_${id.replace(/[^a-z0-9_]/gi, "_")}(${vars.map(() => v.toFixed(17)).join(", ")}));`)
    .join("\n");
  const cSource = `${c99}\n\n#include <stdio.h>\n\nint main(void)\n{\n${calls}\n    return 0;\n}\n`;

  const dir = mkdtempSync(join(tmpdir(), "spear-probe-"));
  try {
    const cPath = join(dir, `${id}.c`);
    const exePath = join(dir, `${id}.exe`);
    writeFileSync(cPath, cSource);
    execFileSync("gcc", ["-std=c99", "-Wall", "-Wextra", "-pedantic", "-O2", cPath, "-o", exePath], { stdio: "pipe" });
    const out = execFileSync(exePath, { stdio: "pipe" }).toString().trim().split(/\r?\n/).map(Number);

    const wasmFn = await instantiateSpearWasm(Buffer.from(toWasmBytes(node)).toString("base64"));
    // JS reference via the vector evaluator (the scoring path)
    const arrays: Record<string, Float64Array> = {};
    for (const v of vars) arrays[v] = Float64Array.from(vals);
    const jsRef = evaluateNode(node, arrays, P);

    for (let i = 0; i < P; i++) {
      const wv = wasmFn(vars.map(() => vals[i]));
      const rel = Math.abs(out[i] - wv) / Math.max(1, Math.abs(wv));
      console.log(`v=${vals[i].toFixed(1).padStart(5)}  C=${out[i].toExponential(3).padStart(11)}  wasm=${wv.toExponential(3).padStart(11)}  js=${jsRef[i].toExponential(3).padStart(11)}  relC-W=${rel.toExponential(2)}`);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
