// Calibrate OP_COST against real hardware.
//
// Every speedup in the ledger is derived from estimateCost(), whose per-op
// weights (mul/add = 1, pdiv = 4, sqrt = 2, transcendental = 20) were chosen a
// priori, never measured. bench-wallclock.ts then showed the consequence:
// logsumexp2 advertised ×8.57 and measured ×2.47, light_falloff_punctual
// advertised ×1.54 and measured ×0.73 — i.e. SLOWER. Both reference ASTs were
// verified to reproduce their targets, so the error is in the weights.
//
// This script measures each op directly instead of inferring it. For every op
// we time a kernel that applies it REPEAT times per element, and subtract a
// baseline kernel with the same loop structure and no op. The slope in ns is
// then normalised so that `mul` = 1, giving a cost in the same units OP_COST
// already uses.
//
// Honest-measurement precautions (same discipline as bench-wallclock):
//   • a dependency chain (x = op(x) + k) prevents -O2 from hoisting the op out
//     of the repeat loop or vectorising it away;
//   • the accumulator goes to a volatile sink so nothing is dead-code removed;
//   • kernels live in a separate translation unit, no LTO, so no inlining into
//     constants;
//   • warmup pass, then the MEDIAN of REPS timed reps;
//   • ops whose measured slope is below the timer noise floor are reported as
//     unresolved rather than rounded to a convenient number.
//
// RESULT — READ BEFORE USING THESE NUMBERS TO CHANGE OP_COST.
//
// The weights this script produces were tested end-to-end against the 22
// wall-clock-measured kernels and they predict real hardware WORSE than the
// a priori weights they would replace: median |log(modelled/measured)| went
// from 0.097 to 0.177. They were therefore NOT adopted.
//
// Why: this harness times each op in a serial dependency chain (x = op(x)),
// which measures LATENCY. Real kernels evaluate independent subexpressions
// that the CPU pipelines and vectorises, so what governs their runtime is
// THROUGHPUT. The two differ by an op-dependent factor — most visibly for
// sqrt/pdiv, which look ~5x more expensive under latency than they behave in
// practice. A latency table is the wrong instrument for pricing whole ASTs.
//
// This file is kept because the negative result is worth keeping: it documents
// that OP_COST must be fitted against whole-kernel measurements (see
// bench-wallclock.ts), not against isolated ops, and it stops the next person
// from redoing the same experiment and silently adopting the answer.
//
// Usage:
//   npx tsx scripts/calibrate-cost-model.ts          # table
//   npx tsx scripts/calibrate-cost-model.ts --md     # markdown report
//   npx tsx scripts/calibrate-cost-model.ts --json   # machine readable
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { OP_COST } from "../src/lib/spear/engine";

const N = 1 << 14; // elements
// Two repeat counts: cost is derived from the SLOPE between them, so the loop
// scaffolding, the taming line and the call overhead cancel exactly instead of
// being subtracted from a separate baseline kernel. A single-baseline
// subtraction gave physically impossible answers (log measured NEGATIVE, sin
// 4x cheaper than cos) because adding the op also changes the dependency chain
// the taming line sits on; differencing two repeat counts of the SAME kernel
// removes that confound.
const REPEAT_LO = 32;
const REPEAT_HI = 96;
const REPS = 9;
const INNER = 20;

// op -> C expression applying it to the running value `x`
const OPS: Record<string, string> = {
  add: "(x + 1.000001f)",
  sub: "(x - 0.999999f)",
  mul: "(x * 1.000001f)",
  pdiv: "(x / 1.000001f)",
  sqrt: "sqrtf(x)",
  abs: "fabsf(x)",
  neg: "(-x)",
  max: "fmaxf(x, 0.5f)",
  min: "fminf(x, 1.5f)",
  sq: "(x * x)",
  exp: "expf(x)",
  log: "logf(x)",
  sin: "sinf(x)",
  cos: "cosf(x)",
  atan: "atanf(x)",
  asin: "asinf(x)",
  acos: "acosf(x)",
  tanh: "tanhf(x)",
  erf: "erff(x)",
};

// Ops that need their input kept in a safe domain each iteration so we time the
// arithmetic and not a slow path (NaN/denormal/huge-argument reduction).
const TAME: Record<string, string> = {
  exp: "x = 0.5f + 0.25f * (x - floorf(x));",
  log: "x = 1.0f + 0.5f * (x - floorf(x));",
  asin: "x = 0.5f * (x - floorf(x));",
  acos: "x = 0.5f * (x - floorf(x));",
  sqrt: "x = 1.0f + 0.5f * (x - floorf(x));",
  sin: "x = 0.5f * (x - floorf(x));",
  cos: "x = 0.5f * (x - floorf(x));",
  atan: "x = 0.5f * (x - floorf(x));",
  tanh: "x = 0.5f * (x - floorf(x));",
  erf: "x = 0.5f * (x - floorf(x));",
  pdiv: "x = 1.0f + 0.5f * (x - floorf(x));",
};

function kernelSrc(expr: string, tame: string, repeat: number): string {
  return `#include <math.h>
void kern(const float *in, float *out, int n) {
  for (int i = 0; i < n; ++i) {
    float x = in[i];
    for (int r = 0; r < ${repeat}; ++r) {
      ${tame}
      x = ${expr};
    }
    out[i] = x;
  }
}
`;
}

const DRIVER = `#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void kern(const float *in, float *out, int n);
static volatile float sink;
int main(void) {
  int n = ${N};
  float *in = malloc(sizeof(float) * n);
  float *out = malloc(sizeof(float) * n);
  unsigned s = 12345u;
  for (int i = 0; i < n; ++i) { s = s * 1664525u + 1013904223u; in[i] = 0.5f + (float)(s >> 8) / (float)(1u << 25); }
  for (int w = 0; w < 3; ++w) { kern(in, out, n); sink = out[n - 1]; }
  double best[${REPS}];
  for (int rep = 0; rep < ${REPS}; ++rep) {
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (int k = 0; k < ${INNER}; ++k) { kern(in, out, n); sink = out[n - 1]; }
    clock_gettime(CLOCK_MONOTONIC, &b);
    double ns = (double)(b.tv_sec - a.tv_sec) * 1e9 + (double)(b.tv_nsec - a.tv_nsec);
    best[rep] = ns / ((double)n * ${INNER});
  }
  for (int i = 0; i < ${REPS}; ++i) for (int j = i + 1; j < ${REPS}; ++j) if (best[j] < best[i]) { double t = best[i]; best[i] = best[j]; best[j] = t; }
  printf("%.6f\\n", best[${REPS} / 2]);
  return 0;
}
`;

function timeKernel(dir: string, expr: string, tame: string, repeat: number, tag: string): number {
  const kc = join(dir, `k_${tag}.c`);
  const dc = join(dir, `d_${tag}.c`);
  const bin = join(dir, `b_${tag}`);
  writeFileSync(kc, kernelSrc(expr, tame, repeat));
  writeFileSync(dc, DRIVER);
  execFileSync("gcc", ["-O2", "-fno-lto", "-o", bin, kc, dc, "-lm"], { stdio: "pipe" });
  return Number(execFileSync(bin, { encoding: "utf8" }).trim());
}

function main(): void {
  const md = process.argv.includes("--md");
  const json = process.argv.includes("--json");
  const dir = mkdtempSync(join(tmpdir(), "spear-calib-"));
  try {
    const rows: { op: string; ns: number; model: number; measured: number | null }[] = [];
    for (const [op, expr] of Object.entries(OPS)) {
      const tame = TAME[op] ?? "";
      const lo = timeKernel(dir, expr, tame, REPEAT_LO, `${op}_lo`);
      const hi = timeKernel(dir, expr, tame, REPEAT_HI, `${op}_hi`);
      // slope = marginal cost of one extra application of this op
      const ns = (hi - lo) / (REPEAT_HI - REPEAT_LO);
      rows.push({ op, ns, model: OP_COST[op as keyof typeof OP_COST] ?? 1, measured: null });
    }

    // Normalise so mul = 1, matching the unit OP_COST is already expressed in.
    const mulNs = rows.find((r) => r.op === "mul")!.ns;
    // Resolution floor: ops whose delta is under this are indistinguishable
    // from zero given timer noise, so we refuse to publish a number for them.
    const floor = Math.max(Math.abs(mulNs) * 0.25, 0.05);
    for (const r of rows) r.measured = Math.abs(r.ns) < floor ? null : r.ns / mulNs;

    rows.sort((a, b) => (b.measured ?? -1) - (a.measured ?? -1));

    if (json) {
      console.log(JSON.stringify({ mulNs, floor, rows }, null, 2));
      return;
    }

    const out: string[] = [];
    const w = (s: string) => out.push(s);
    if (md) {
      w("# Cost-model calibration (measured)\n");
      w(`Generated by \`scripts/calibrate-cost-model.ts\`. gcc -O2, ${N} elems, slope between ${REPEAT_LO} and ${REPEAT_HI} applications, median of ${REPS}.\n`);
      w("`measured` is normalised so `mul` = 1, the same unit `OP_COST` uses. Ops below the");
      w("timer resolution floor are reported as `—` rather than rounded.\n");
      w("| op | modelled | measured | ns/op | verdict |");
      w("|---|---|---|---|---|");
    } else {
      w(`\n▶ cost-model calibration — gcc -O2, slope ${REPEAT_LO}→${REPEAT_HI}, mul = ${mulNs.toFixed(4)} ns, floor ${floor.toFixed(4)} ns\n`);
      w("op        modelled  measured   ns/op   verdict");
      w("-".repeat(58));
    }

    for (const r of rows) {
      const m = r.measured;
      const verdict =
        m === null ? "under resolution" : m > r.model * 1.5 ? "model UNDER-prices" : m < r.model / 1.5 ? "model OVER-prices" : "ok";
      if (md) {
        w(`| \`${r.op}\` | ${r.model} | ${m === null ? "—" : m.toFixed(2)} | ${r.ns.toFixed(4)} | ${verdict} |`);
      } else {
        w(
          `${r.op.padEnd(9)} ${String(r.model).padStart(8)} ${(m === null ? "—" : m.toFixed(2)).padStart(9)} ${r.ns
            .toFixed(4)
            .padStart(7)}   ${verdict}`,
        );
      }
    }
    if (md) w("");
    console.log(out.join("\n"));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

main();
