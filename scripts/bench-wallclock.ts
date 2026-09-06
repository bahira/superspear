// Wall-clock validation of the cost model.
//
// Every speedup in the ledger is a MODEL: estimateCost() counts ALU/SFU units
// (mul/add = 1, div = 4, sqrt = 2, transcendental = 20). That model is a
// hypothesis about real hardware, and an unvalidated hypothesis has no place
// backing a performance claim. This script tests it: champion and exact
// reference law are both emitted as C99, compiled with -O2, and timed on the
// same input buffer. We then report measured speedup vs modelled speedup.
//
// Why C and not WASM: the WASM backend imports exp/log/sin from JS, so timing
// it would mostly measure the JS<->WASM call boundary, not the kernel. C -O2 is
// also the actual deployment target for the MISRA path.
//
// Honest-measurement precautions:
//   • the same pseudo-random input buffer feeds both kernels;
//   • results are accumulated into a volatile sink so -O2 cannot delete the
//     loop, and the kernels are in a separate translation unit compiled
//     without LTO so the calls are not inlined away into constants;
//   • a warmup pass precedes timing; we take the MEDIAN of several reps;
//   • tasks whose champion IS the exact law are expected to land at ~1.0 —
//     that is a correctness check on the harness, not a win.
//
// Usage:
//   npx tsx scripts/bench-wallclock.ts            # all tasks with a reference
//   npx tsx scripts/bench-wallclock.ts gelu silu  # subset
//   npx tsx scripts/bench-wallclock.ts --md > REPORTS/wallclock-bench.md
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadLedger } from "../src/lib/spear/ledger";
import { buildTasks } from "../src/lib/spear/benchmarks";
import { parseNode, estimateCost, toMisraC, collectVarNames } from "../src/lib/spear/engine";
import type { SpearNode } from "../src/lib/spear/engine";

const N = 1 << 16;   // elements per pass
const REPS = 9;      // median over this many timed reps
const INNER = 40;    // passes per rep — amortises timer granularity

interface Row {
  id: string;
  bound: boolean;
  vars: number;
  modelFormula: number;
  modelExact: number;
  modelSpeedup: number;
  nsFormula: number;
  nsExact: number;
  measured: number;
  agreement: number; // measured / modelled
}

/**
 * C body for a node. We use the MISRA-C:2012 emitter, not toC(): toC() targets
 * CUDA (`__device__`, `1f` literals) and does not compile with a host gcc,
 * whereas the MISRA path is strict C99 — and it is the actual artifact SPEAR
 * ships for embedded deployment, so timing it measures what users run.
 */
function cFunction(node: SpearNode, name: string, params: string[]): string {
  const decl = params.map((p) => `const float32_t ${p}`).join(", ");
  return toMisraC(node, name, decl);
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[s.length >> 1];
}

const args = process.argv.slice(2);
const asMd = args.includes("--md");
const asJson = args.includes("--json");
const only = new Set(args.filter((a) => !a.startsWith("--")));

const ledger = loadLedger();
const defs = new Map(buildTasks().map((t) => [t.id, t]));
const rows: Row[] = [];
const skipped: string[] = [];

const dir = mkdtempSync(join(tmpdir(), "spear-wc-"));

/**
 * Measure the harness floor: an out-of-line function that just returns its
 * argument. Anything timed within ~2x of this is dominated by call + loop
 * overhead, so its "speedup" is an artifact of the harness rather than a
 * property of the kernel. Reporting x1.00 for those without saying so would
 * be exactly the kind of unearned claim this whole exercise is about removing.
 */
function measureFloor(): number {
  const c = `#include <stdio.h>
#include <time.h>
#define N ${N}
#define INNER ${INNER}
typedef float float32_t;
static float32_t buf[N];
volatile double sink = 0.0;
float32_t noop(const float32_t x) { return x; }
static double now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec*1e9+(double)t.tv_nsec;}
int main(void){
  for (int i=0;i<N;i++) buf[i]=(float32_t)(0.05+1.9*((double)i/N));
  double best=1e18;
  for (int r=0;r<${REPS};r++){
    double acc=0.0,t0=now_ns();
    for(int k=0;k<INNER;k++) for(int i=0;i<N;i++) acc+=noop(buf[i]);
    double t1=now_ns(); sink+=acc;
    double ns=(t1-t0)/((double)N*INNER); if(ns<best) best=ns;
  }
  printf("%.6f\\n",best);
  return 0;
}
`;
  const cp = join(dir, "floor.c");
  const bp = join(dir, "floor");
  writeFileSync(cp, c);
  execFileSync("gcc", ["-O2", "-fno-lto", "-o", bp, cp, "-lm"], { stdio: "pipe" });
  return Number(execFileSync(bp, { encoding: "utf8" }).trim());
}

const FLOOR = measureFloor();

for (const [id, entry] of Object.entries(ledger) as [string, any][]) {
  if (only.size && !only.has(id)) continue;
  const t = defs.get(id);
  if (!t || !entry.tree) continue;
  const ref: SpearNode | undefined = t.exactRefNode;
  // Need a reference AST to compile; tasks that only declare a scalar exactCost
  // have no compilable law, so a measured comparison is impossible.
  if (!ref) { skipped.push(`${id} (no exact reference AST to compile against)`); continue; }

  let champ: SpearNode;
  try { champ = parseNode(entry.tree); } catch { skipped.push(`${id} (unparseable AST)`); continue; }

  // Union of variables, so both kernels take the same signature.
  const params = [...new Set([...collectVarNames(champ), ...collectVarNames(ref)])];
  if (params.length === 0) { skipped.push(`${id} (constant formula)`); continue; }

  // the MISRA emitter re-emits headers + typedef per function; keep them once
  const stripPreamble = (src: string): string => src.slice(src.indexOf("float32_t "));
  const kernels = `${cFunction(champ, "spear_formula", params)}\n\n${stripPreamble(cFunction(ref, "spear_exact", params))}\n`;
  const argList = params.map((p, i) => `buf[(i + ${i * 7}) & MASK]`).join(", ");
  const main = `#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N ${N}
#define MASK (N - 1)
#define INNER ${INNER}
#define REPS ${REPS}

typedef float float32_t;
float32_t spear_formula(${params.map((p) => `const float32_t ${p}`).join(", ")});
float32_t spear_exact(${params.map((p) => `const float32_t ${p}`).join(", ")});

static float32_t buf[N];
volatile double sink = 0.0;

static double now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

int main(void) {
  /* deterministic inputs in a benign range, avoiding exact 0 for divisions */
  unsigned int s = 12345u;
  for (int i = 0; i < N; i++) {
    s = s * 1103515245u + 12345u;
    double u = (double)((s >> 8) & 0xFFFFFF) / (double)0xFFFFFF;
    buf[i] = (float32_t)(0.05 + 1.9 * u);
  }

  double tf[REPS], te[REPS];
  for (int r = 0; r < REPS + 1; r++) {           /* r == 0 is warmup */
    double acc = 0.0;
    double t0 = now_ns();
    for (int k = 0; k < INNER; k++)
      for (int i = 0; i < N; i++) acc += spear_formula(${argList});
    double t1 = now_ns();
    sink += acc;
    if (r) tf[r - 1] = (t1 - t0) / ((double)N * INNER);

    acc = 0.0;
    t0 = now_ns();
    for (int k = 0; k < INNER; k++)
      for (int i = 0; i < N; i++) acc += spear_exact(${argList});
    t1 = now_ns();
    sink += acc;
    if (r) te[r - 1] = (t1 - t0) / ((double)N * INNER);
  }
  for (int i = 0; i < REPS; i++) printf("%.6f %.6f\\n", tf[i], te[i]);
  return 0;
}
`;
  const kPath = join(dir, `${id}_k.c`);
  const mPath = join(dir, `${id}_m.c`);
  const bin = join(dir, id);
  writeFileSync(kPath, kernels);
  writeFileSync(mPath, main);
  try {
    // no -flto: keep the kernels opaque so the calls are really executed
    execFileSync("gcc", ["-O2", "-fno-lto", "-o", bin, mPath, kPath, "-lm"], { stdio: "pipe" });
  } catch (e) {
    skipped.push(`${id} (compile failed)`);
    continue;
  }
  const out = execFileSync(bin, { encoding: "utf8" }).trim().split("\n");
  const fs2: number[] = [];
  const es: number[] = [];
  for (const l of out) { const [a, b] = l.split(" ").map(Number); fs2.push(a); es.push(b); }
  const nsF = median(fs2);
  const nsE = median(es);
  // Run-to-run spread of each kernel's own timings. If the gap between the two
  // kernels is smaller than the noise of measuring one of them, the harness
  // simply cannot tell them apart and any ratio is invented precision.
  const spread = (xs: number[]): number => {
    const m = median(xs);
    return median(xs.map((v) => Math.abs(v - m))) * 1.4826; // robust sigma
  };
  const noise = Math.max(spread(fs2), spread(es), FLOOR * 0.02);
  const indistinguishable = Math.abs(nsF - nsE) < 3 * noise;
  const mf = estimateCost(champ);
  const me = t.exactCost ?? estimateCost(ref);
  rows.push({
    id,
    // unresolvable when the two kernels differ by less than measurement noise,
    // or when both sit in the call-overhead floor
    bound: indistinguishable || (nsF < FLOOR * 2.5 && nsE < FLOOR * 2.5),
    vars: params.length,
    modelFormula: mf,
    modelExact: me,
    modelSpeedup: me / Math.max(1, mf),
    nsFormula: nsF,
    nsExact: nsE,
    measured: nsE / nsF,
    agreement: (nsE / nsF) / (me / Math.max(1, mf)),
  });
}

rmSync(dir, { recursive: true, force: true });
rows.sort((a, b) => b.measured - a.measured);

if (asJson) {
  // Machine-readable form so measurements can be written back into the ledger
  // (scripts/write-measured-speed.ts) instead of living only in a report.
  console.log(JSON.stringify({ rows, skipped, floorNs: FLOOR }, null, 2));
} else if (asMd) {
  const lines: string[] = [];
  lines.push("# Wall-clock benchmark — does the cost model tell the truth?");
  lines.push("");
  lines.push(`Champion and exact reference law both emitted as C99, compiled \`gcc -O2\` (no LTO, volatile sink so nothing is optimised away), timed on the same ${N}-element buffer, median of ${REPS} reps × ${INNER} passes. Generated by \`npx tsx scripts/bench-wallclock.ts --md\`.`);
  lines.push("");
  lines.push("`agreement` = measured ÷ modelled. 1.00 means the ALU/SFU cost model predicted reality exactly; below 1 the model over-promises.");
  lines.push("");
  lines.push(`**Resolution limit.** An out-of-line function that merely returns its argument costs **${FLOOR.toFixed(3)} ns/elem** on this machine. Kernels close to that floor — or whose two variants differ by less than the run-to-run noise — are measuring call and loop overhead, not arithmetic. Those rows are marked ⚠ and their ratios are *not* evidence of anything. Trusting them would reintroduce exactly the kind of unearned number this audit removed.`);
  lines.push("");
  lines.push("| task | vars | ns/elem formula | ns/elem exact | measured ⚡ | modelled ⚡ | agreement |");
  lines.push("|---|---|---|---|---|---|---|");
  for (const r of rows) {
    lines.push(`| \`${r.id}\`${r.bound ? " ⚠" : ""} | ${r.vars} | ${r.nsFormula.toFixed(3)} | ${r.nsExact.toFixed(3)} | **×${r.measured.toFixed(2)}** | ×${r.modelSpeedup.toFixed(2)} | ${r.bound ? "n/a" : r.agreement.toFixed(2)} |`);
  }
  lines.push("");
  const solidMd = rows.filter((r) => !r.bound);
  const ag = solidMd.map((r) => r.agreement).sort((a, b) => a - b);
  const med = ag[ag.length >> 1];
  lines.push(`**Median agreement: ${med?.toFixed(2)}** across the ${solidMd.length} resolvable tasks (${rows.length - solidMd.length} of ${rows.length} were call-overhead bound and excluded).`);
  lines.push("");
  const over = solidMd.filter((r) => r.agreement < 0.5);
  if (over.length) {
    lines.push("## Where the model over-promises (agreement < 0.5)");
    lines.push("");
    for (const r of over) lines.push(`- \`${r.id}\`: model says ×${r.modelSpeedup.toFixed(2)}, hardware says ×${r.measured.toFixed(2)}`);
    lines.push("");
  }
  if (skipped.length) {
    lines.push("## Not measurable");
    lines.push("");
    for (const s of skipped) lines.push(`- ${s}`);
  }
  console.log(lines.join("\n"));
} else {
  console.log(`\n▶ wall-clock bench — gcc -O2, ${N} elems, median of ${REPS}×${INNER}`);
  console.log(`  empty-call floor: ${FLOOR.toFixed(3)} ns/elem — rows marked (!) are unresolvable (difference below measurement noise, or both at the call-overhead floor)\n`);
  console.log("task                     ns/f     ns/exact  measured  modelled  agree");
  console.log("-".repeat(72));
  for (const r of rows) {
    console.log(
      r.id.padEnd(24) +
        r.nsFormula.toFixed(3).padStart(8) +
        r.nsExact.toFixed(3).padStart(10) +
        ("×" + r.measured.toFixed(2)).padStart(10) +
        ("×" + r.modelSpeedup.toFixed(2)).padStart(10) +
        r.agreement.toFixed(2).padStart(7) +
        (r.bound ? "  (!)" : ""),
    );
  }
  const solid = rows.filter((r) => !r.bound);
  const ag = solid.map((r) => r.agreement).sort((a, b) => a - b);
  console.log(`\nmeasured: ${rows.length} | resolvable: ${solid.length} | median agreement (resolvable only): ${ag[ag.length >> 1]?.toFixed(2)}`);
  if (skipped.length) console.log(`skipped: ${skipped.length}`);
}
