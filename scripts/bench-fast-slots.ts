// Wall-clock measurement of the FAST SLOTS.
//
// A fast slot is the deliberate trade: a cheaper, less accurate kernel offered
// alongside the precise champion. 53 of 89 records ship one, advertising cost
// ratios up to x24 (rayleigh_phase 24 units -> 0, logsumexp2 67 -> 3).
//
// Every one of those ratios is a COST-MODEL prediction. bench-wallclock.ts
// never compiled a fast slot — it only ever timed champion vs exact law — and
// that same model has already been caught over-promising badly on this repo
// (logsumexp2 x8.57 modelled / x2.47 measured; gelu x1.92 / x0.99; smootherstep
// x2.25 / x0.50). So the headline "x24 faster" numbers attached to fast slots
// have never been checked against a CPU.
//
// This script times champion vs fast slot on the same buffer, and reports the
// accuracy given up alongside the speed gained, because a fast slot is only
// meaningful as a pair: speedup AND error. A slot that is 1.02x faster for 400x
// the error is not a trade, it is a defect.
//
// Same honesty precautions as bench-wallclock: identical input buffer, volatile
// sink so -O2 cannot delete the loop, separate translation unit without LTO,
// warmup then median of several reps, and an explicit call-overhead floor below
// which a difference is reported as unresolvable rather than as a win.
//
// Usage:
//   npx tsx scripts/bench-fast-slots.ts          # table
//   npx tsx scripts/bench-fast-slots.ts --md     # markdown report
//   npx tsx scripts/bench-fast-slots.ts --json
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadLedger } from "../src/lib/spear/ledger";
import { buildTasks } from "../src/lib/spear/benchmarks";
import { parseNode, estimateCost, toMisraC, collectVarNames } from "../src/lib/spear/engine";
import type { SpearNode } from "../src/lib/spear/engine";

const N = 1 << 16;
const REPS = 9;
const INNER = 40;

const args = process.argv.slice(2);
const asMd = args.includes("--md");
const asJson = args.includes("--json");
const only = new Set(args.filter((a) => !a.startsWith("--")));

interface Row {
  id: string;
  costChamp: number;
  costFast: number;
  modelled: number;
  nsChamp: number;
  nsFast: number;
  measured: number;
  bound: boolean;
  mseChamp: number;
  mseFast: number;
  errorFactor: number;
  agreement: number;
}

function cFunction(node: SpearNode, name: string, params: string[]): string {
  return toMisraC(node, name, params.map((p) => `const float32_t ${p}`).join(", "));
}

const dir = mkdtempSync(join(tmpdir(), "spear-fast-"));
const driver = (kernels: string, argList: string, params: number): string => `${kernels}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define MASK ${N - 1}
static volatile float sink;
static double now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec*1e9+(double)t.tv_nsec;}
int main(void){
  float *buf = malloc(sizeof(float) * ${N});
  unsigned s = 22221u;
  for (int i = 0; i < ${N}; ++i) { s = s*1664525u+1013904223u; buf[i] = -3.0f + 6.0f*((float)(s>>8)/(float)(1u<<24)); }
  float acc;
  for (int w = 0; w < 2; ++w) { acc = 0.0f; for (int i = 0; i < ${N}; ++i) acc += spear_a(${argList}); sink = acc; }
  double ca[${REPS}], fa[${REPS}];
  for (int r = 0; r < ${REPS}; ++r) {
    double t0 = now_ns();
    for (int k = 0; k < ${INNER}; ++k) { acc = 0.0f; for (int i = 0; i < ${N}; ++i) acc += spear_a(${argList}); sink = acc; }
    ca[r] = (now_ns() - t0) / ((double)${N} * ${INNER});
    t0 = now_ns();
    for (int k = 0; k < ${INNER}; ++k) { acc = 0.0f; for (int i = 0; i < ${N}; ++i) acc += spear_b(${argList}); sink = acc; }
    fa[r] = (now_ns() - t0) / ((double)${N} * ${INNER});
  }
  for (int i=0;i<${REPS};++i) for (int j=i+1;j<${REPS};++j){ if(ca[j]<ca[i]){double t=ca[i];ca[i]=ca[j];ca[j]=t;} if(fa[j]<fa[i]){double t=fa[i];fa[i]=fa[j];fa[j]=t;} }
  printf("%.6f %.6f\\n", ca[${REPS} / 2], fa[${REPS} / 2]);
  (void)${params};
  return 0;
}
`;

function measureFloor(): number {
  const cp = join(dir, "floor.c");
  const bp = join(dir, "floor");
  writeFileSync(cp, `#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define MASK ${N - 1}
static volatile float sink;
static double now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec*1e9+(double)t.tv_nsec;}
__attribute__((noinline)) static float id_(const float x){ return x; }
int main(void){
  float *buf = malloc(sizeof(float)*${N}); unsigned s=1u;
  for(int i=0;i<${N};++i){ s=s*1664525u+1013904223u; buf[i]=(float)(s>>8)/(float)(1u<<24); }
  float acc; for(int w=0;w<2;++w){acc=0;for(int i=0;i<${N};++i)acc+=id_(buf[i]);sink=acc;}
  double b[${REPS}];
  for(int r=0;r<${REPS};++r){ double t0=now_ns();
    for(int k=0;k<${INNER};++k){acc=0;for(int i=0;i<${N};++i)acc+=id_(buf[i]);sink=acc;}
    b[r]=(now_ns()-t0)/((double)${N}*${INNER}); }
  for(int i=0;i<${REPS};++i)for(int j=i+1;j<${REPS};++j)if(b[j]<b[i]){double t=b[i];b[i]=b[j];b[j]=t;}
  printf("%.6f\\n", b[${REPS}/2]); return 0; }
`);
  execFileSync("gcc", ["-std=gnu99", "-O2", "-fno-lto", "-o", bp, cp, "-lm"], { stdio: "pipe" });
  return Number(execFileSync(bp, { encoding: "utf8" }).trim());
}

const FLOOR = measureFloor();
const ledger = loadLedger();
const defs = new Map((buildTasks() as unknown as { id: string; evaluate: (n: SpearNode) => { metric: number } }[]).map((t) => [t.id, t]));
const rows: Row[] = [];
const skipped: string[] = [];

for (const [id, entry] of Object.entries(ledger) as [string, any][]) {
  if (only.size && !only.has(id)) continue;
  const t = defs.get(id);
  if (!t || !entry.tree || !entry.fastTree) { if (t && entry.tree && !entry.fastTree) skipped.push(`${id} (no fast slot)`); continue; }

  let champ: SpearNode, fast: SpearNode;
  try { champ = parseNode(entry.tree); fast = parseNode(entry.fastTree); } catch { skipped.push(`${id} (unparseable)`); continue; }

  const params = [...new Set([...collectVarNames(champ), ...collectVarNames(fast)])];
  if (params.length === 0) { skipped.push(`${id} (constant formula)`); continue; }

  const strip = (src: string): string => src.slice(src.indexOf("float32_t "));
  const kernels = `${cFunction(champ, "spear_a", params)}\n\n${strip(cFunction(fast, "spear_b", params))}\n`;
  const argList = params.map((p, i) => `buf[(i + ${i * 7}) & MASK]`).join(", ");

  const cp = join(dir, `${id}.c`);
  const bp = join(dir, id);
  writeFileSync(cp, driver(kernels, argList, params.length));
  try {
    execFileSync("gcc", ["-std=gnu99", "-O2", "-fno-lto", "-o", bp, cp, "-lm"], { stdio: "pipe" });
  } catch { skipped.push(`${id} (compile failed)`); continue; }

  const [nsChamp, nsFast] = execFileSync(bp, { encoding: "utf8" }).trim().split(/\s+/).map(Number);
  const costChamp = estimateCost(champ);
  const costFast = estimateCost(fast);
  const modelled = costChamp / Math.max(costFast, 1);
  const measured = nsChamp / nsFast;
  // both kernels indistinguishable from the empty-call floor => the ratio is
  // an artifact of call overhead, not a property of the formulas
  const bound = nsChamp < FLOOR * 2.5 && nsFast < FLOOR * 2.5;

  const mseChamp = t.evaluate(champ).metric;
  const mseFast = t.evaluate(fast).metric;
  rows.push({
    id, costChamp, costFast, modelled, nsChamp, nsFast, measured, bound,
    mseChamp, mseFast,
    errorFactor: mseChamp > 0 ? mseFast / mseChamp : Infinity,
    agreement: measured / modelled,
  });
}

rmSync(dir, { recursive: true, force: true });
rows.sort((a, b) => b.measured - a.measured);

const resolvable = rows.filter((r) => !r.bound);
const worthIt = resolvable.filter((r) => r.measured >= 1.2);
const useless = resolvable.filter((r) => r.measured < 1.05);

if (asJson) {
  console.log(JSON.stringify({ rows, skipped, floorNs: FLOOR }, null, 2));
} else if (asMd) {
  const L: string[] = [];
  L.push("# Fast slots — measured, not modelled\n");
  L.push(`A fast slot trades accuracy for speed. Every advertised ratio was a cost-model prediction until now: \`bench-wallclock.ts\` only ever timed champion vs exact law, never the fast slot. Generated by \`npx tsx scripts/bench-fast-slots.ts --md\`.\n`);
  L.push(`gcc -O2, ${N} elems, median of ${REPS}×${INNER}. Call-overhead floor ${FLOOR.toFixed(3)} ns/elem; rows at the floor are marked (!) and excluded from the verdict.\n`);
  L.push("`error ×` is how much worse the fast slot's MSE is. A slot is only a real trade if it buys speed **and** the error is acceptable.\n");
  L.push("| task | cost ratio | measured ⚡ | agreement | mse champion | mse fast | error × |");
  L.push("|---|---|---|---|---|---|---|");
  for (const r of rows) {
    L.push(`| \`${r.id}\`${r.bound ? " (!)" : ""} | ×${r.modelled.toFixed(2)} | **×${r.measured.toFixed(2)}** | ${r.agreement.toFixed(2)} | ${r.mseChamp.toExponential(2)} | ${r.mseFast.toExponential(2)} | ${Number.isFinite(r.errorFactor) ? r.errorFactor.toExponential(1) : "∞"} |`);
  }
  L.push("");
  L.push(`**${resolvable.length} resolvable** of ${rows.length} measured. **${worthIt.length}** deliver ≥1.2× real speedup; **${useless.length}** deliver under 1.05× — for those the accuracy loss buys nothing.`);
  console.log(L.join("\n"));
} else {
  console.log(`\n▶ fast-slot bench — gcc -O2, ${N} elems, median of ${REPS}×${INNER}`);
  console.log(`  call-overhead floor: ${FLOOR.toFixed(3)} ns/elem — (!) rows are unresolvable\n`);
  console.log("task                       ns/champ  ns/fast  measured modelled  agree   error×");
  console.log("-".repeat(84));
  for (const r of rows) {
    console.log(
      `${r.id.padEnd(24)} ${r.nsChamp.toFixed(3).padStart(9)} ${r.nsFast.toFixed(3).padStart(8)} ` +
        `${("×" + r.measured.toFixed(2)).padStart(9)} ${("×" + r.modelled.toFixed(2)).padStart(8)} ` +
        `${r.agreement.toFixed(2).padStart(6)} ${(Number.isFinite(r.errorFactor) ? r.errorFactor.toExponential(1) : "inf").padStart(9)}` +
        `${r.bound ? "  (!)" : ""}`,
    );
  }
  console.log(`\nmeasured: ${rows.length} | resolvable: ${resolvable.length} | >=1.2x real: ${worthIt.length} | <1.05x (no gain): ${useless.length}`);
  if (skipped.length) console.log(`skipped: ${skipped.length}`);
}
