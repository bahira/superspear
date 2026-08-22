// Export audit: pour chaque découverte du grounded loop —
//   1. lint MISRA-C:2012 du code émis (branchless, zéro heap, types fixes)
//   2. compilation gcc réelle + exécution + parité numérique C ↔ WASM
//   3. benchmark wall-clock formule vs loi exacte (les deux en WASM)
// Usage: npx tsx scripts/export-audit.ts [seed] [budget]
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import { runGroundedLoop, type LoopTaskSnapshot } from "../src/lib/spear/loop";
import { lintMisraC } from "../src/lib/spear/engine";
import { instantiateSpearWasm, toWasmBytes } from "../src/lib/spear/wasm";
import { buildTasks } from "../src/lib/spear/benchmarks";

interface Row {
  id: string;
  misraOk: boolean;
  violations: string[];
  cCompiles: boolean;
  parityMaxDiff: number;
  benchNsFormula?: number;
  benchNsExact?: number;
  measuredSpeedup?: number;
  costModelSpeedup?: number;
}

function extractParams(c99: string): string[] {
  const sig = c99.match(/float32_t\s+\w+\(([^)]*)\)/);
  if (!sig) return [];
  return [...sig[1].matchAll(/const float32_t\s+([A-Za-z_]\w*)/g)].map((mm) => mm[1]);
}

async function benchWasm(b64: string, nVars: number, n = 200_000): Promise<number> {
  const fn = await instantiateSpearWasm(b64);
  const xs = new Float64Array(n);
  for (let i = 0; i < n; i++) xs[i] = -6 + (12 * i) / (n - 1);
  const args = new Array(nVars).fill(0);
  for (let w = 0; w < 3; w++) for (let i = 0; i < n; i++) { for (let k = 0; k < nVars; k++) args[k] = xs[i]; fn(args); }
  const t0 = performance.now();
  for (let i = 0; i < n; i++) { for (let k = 0; k < nVars; k++) args[k] = xs[i]; fn(args); }
  return (performance.now() - t0) * 1e6 / n; // ns/elem
}

async function main() {
  const seed = Number(process.argv[2] ?? 9999);
  const budget = Number(process.argv[3] ?? 1000);
  console.log(`▶ grounded loop seed=${seed} budget=${budget} (audit d'export)...`);
  const progress = await runGroundedLoop({ seed, budget, deadlineMs: 45_000 });
  const defs = new Map(buildTasks().map((t) => [t.id, t]));
  const dir = mkdtempSync(join(tmpdir(), "spear-misra-"));
  const rows: Row[] = [];

  for (const snap of progress.tasks as LoopTaskSnapshot[]) {
    if (!snap.best || !snap.c99) continue;
    const row: Row = { id: snap.taskId, misraOk: true, violations: [], cCompiles: false, parityMaxDiff: NaN };

    // ---- 1. lint statique MISRA
    const lint = lintMisraC(snap.c99);
    row.misraOk = lint.ok;
    row.violations = lint.violations;

    // ---- 2. compilation gcc + parité C ↔ WASM
    try {
      const params = extractParams(snap.c99);
      const P = 12;
      const vals = Array.from({ length: P }, (_, i) => -6 + (12 * i) / (P - 1));
      const calls = vals.map((v) => `    printf("%.17g\\n", (double)spear_${snap.taskId.replace(/[^a-z0-9_]/gi, "_")}(${params.map(() => v.toFixed(17)).join(", ")}));`).join("\n");
      const cSource = `${snap.c99}\n\n#include <stdio.h>\n\nint main(void)\n{\n${calls}\n    return 0;\n}\n`;
      const cPath = join(dir, `${snap.taskId}.c`);
      const exePath = join(dir, `${snap.taskId}.exe`);
      writeFileSync(cPath, cSource);
      execFileSync("gcc", ["-std=c99", "-Wall", "-Wextra", "-pedantic", "-O2", cPath, "-o", exePath], { stdio: "pipe" });
      row.cCompiles = true;
      const out = execFileSync(exePath, { stdio: "pipe" }).toString().trim().split(/\r?\n/).map(Number);
      const wasmFn = await instantiateSpearWasm(snap.wasm!);
      // float32 (C) vs float64 (wasm): tolerance must be RELATIVE to magnitude
      let maxRel = 0;
      let mismatched = false;
      for (let i = 0; i < P; i++) {
        const wv = wasmFn(new Array(params.length).fill(vals[i]));
        const cv = out[i];
        if (!Number.isFinite(cv) && !Number.isFinite(wv)) continue; // same-side overflow
        const rel = Math.abs(cv - wv) / Math.max(1, Math.abs(wv));
        if (Number.isFinite(rel)) maxRel = Math.max(maxRel, rel);
        else mismatched = true;
      }
      row.parityMaxDiff = mismatched ? Number.NaN : maxRel;
    } catch (e) {
      const msg = (e as Error).message;
      const stderr = (e as { stderr?: Buffer }).stderr?.toString().split(/\r?\n/).slice(0, 3).join(" | ") ?? "";
      row.violations.push(`C-stage: ${(stderr || msg).slice(0, 200)}`);
    }

    // ---- 3. bench wall-clock formule vs loi exacte (toutes deux en WASM)
    const def = defs.get(snap.taskId);
    const nVars = extractParams(snap.c99).length;
    if (def?.exactRefNode && snap.wasm && nVars > 0) {
      try {
        row.benchNsFormula = await benchWasm(snap.wasm, nVars);
        const exactB64 = Buffer.from(toWasmBytes(def.exactRefNode)).toString("base64");
        row.benchNsExact = await benchWasm(exactB64, def.variables.length);
        row.measuredSpeedup = row.benchNsExact / row.benchNsFormula;
        row.costModelSpeedup = snap.speed?.estimatedSpeedup;
      } catch (e) {
        console.log(`  [bench:${snap.taskId}] ${(e as Error).message.slice(0, 140)}`);
      }
    }
    rows.push(row);
  }

  console.log("\n══════════ AUDIT EXPORT MISRA-C / PARITÉ / BENCH ══════════");
  let allOk = true;
  for (const r of rows) {
    const misra = r.violations.length === 0 ? "MISRA✓" : `✗ ${r.violations.join(" | ").slice(0, 160)}`;
    const comp = r.cCompiles ? "gcc✓" : "gcc✗";
    const parity = Number.isFinite(r.parityMaxDiff) ? `parité ${r.parityMaxDiff.toExponential(1)}` : "parité DIVERGENTE";
    const bench = r.measuredSpeedup !== undefined
      ? `bench ×${r.measuredSpeedup.toFixed(2)} (${r.benchNsFormula!.toFixed(0)}/${r.benchNsExact!.toFixed(0)} ns/el)`
      : "";
    if (!r.misraOk || !r.cCompiles || !(Number.isFinite(r.parityMaxDiff) && r.parityMaxDiff <= 1e-4)) allOk = false;
    console.log(`[${r.id.padEnd(18)}] ${misra.padEnd(14)} ${comp} ${parity.padEnd(14)} ${bench}`);
  }
  console.log(`\n${allOk ? "✅ TOUS LES EXPORTS PASSENT" : "❌ DES VIOLATIONS EXISTENT"} — ${rows.length} tâches auditées`);
  rmSync(dir, { recursive: true, force: true });
  if (!allOk) process.exitCode = 1;
}

main().catch((e) => { console.error(e); process.exit(1); });
