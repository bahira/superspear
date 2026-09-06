// Regression test for snapConstants().
//
// Numeric refinement hill-climbs in f64, but every write went through
// roundConst() which truncated to 6 decimals. A constant converging on
// 15*pi/4 = 11.78097245... was stored as 11.780972, and that truncation — not
// the search — capped the achievable error. On kerr it cost six orders of
// magnitude (3.8e-17 instead of 1.7e-23).
//
// Two properties must hold, and both have already been violated during
// development, so both are asserted here:
//
//   1. SNAPPING RECOVERS EXACTNESS. Naively this looks easy, but scoring each
//      constant in isolation is misleading: on kerr, snapping 128/3 alone
//      makes the metric WORSE (5.8e-17 vs 3.8e-17) because the other constant
//      is still truncated and now dominates. Only 15*pi/4 AND 128/3 together
//      win. A greedy per-constant rule rejects both and silently regresses to
//      the old behaviour — this test catches exactly that.
//
//   2. SNAPPING IS NEVER HARMFUL. A fitted 3.1416 that is not really pi must
//      not be rewritten into something worse. snapConstants only commits a
//      substitution that scores better, so the metric must never degrade.
//
// Usage: npx tsx scripts/test-const-snapping.ts
import { buildTasks } from "../src/lib/spear/benchmarks";
import { parseFormula, snapConstants, roundConst } from "../src/lib/spear/engine";

interface Case {
  task: string;
  formula: string;
  /** metric that must be reached (or beaten) after snapping */
  target: number;
}

// Truncated constants, exactly as numeric refinement leaves them.
const RECOVERY: Case[] = [
  // 4/b + 15pi/4 /b^2 + 128/3 /b^3 — needs BOTH constants snapped together
  { task: "kerr", formula: "((4/b) + ((11.780972/(b)²) + (42.666667/(b)³)))", target: 1e-22 },
  // 2595/ln(10) * log(1 + f/700): the constant multiplies a log, so truncation
  // is amplified — this one is nine orders off before snapping
  { task: "mel_scale", formula: "((1126.994181) * log((1 + (x * 0.001428571428571429))))", target: 1e-24 },
];

// Constants that are NOT closed forms: snapping must not make things worse.
const SAFETY: { task: string; formula: string }[] = [
  { task: "gaussian_kernel", formula: "exp((-(3.1416 * (x)²)))" },
  { task: "gaussian_kernel", formula: "exp((-(0.51 * (x)²)))" },
  { task: "sigmoid", formula: "(1/(1 + exp((-(1.01 * x)))))" },
];

function main(): void {
  const tasks = buildTasks() as unknown as { id: string; evaluate: (n: unknown) => { metric: number } }[];
  const fail: string[] = [];

  for (const c of RECOVERY) {
    const t = tasks.find((x) => x.id === c.task);
    if (!t) { fail.push(`[${c.task}] no such task`); continue; }
    const node = parseFormula(c.formula);
    const before = t.evaluate(node).metric;
    const r = snapConstants(node, (cand) => t.evaluate(cand).metric);
    if (!(r.score <= c.target)) {
      fail.push(
        `[${c.task}] snapping did not recover exactness: ${before.toExponential(2)} -> ` +
          `${r.score.toExponential(2)}, expected <= ${c.target.toExponential(0)}`,
      );
    } else {
      console.log(`  ok  ${c.task.padEnd(14)} ${before.toExponential(2)} -> ${r.score.toExponential(2)}`);
    }
  }

  for (const c of SAFETY) {
    const t = tasks.find((x) => x.id === c.task);
    if (!t) { fail.push(`[${c.task}] no such task`); continue; }
    const node = parseFormula(c.formula);
    const before = t.evaluate(node).metric;
    const r = snapConstants(node, (cand) => t.evaluate(cand).metric);
    // allow exact equality; forbid any degradation
    if (r.score > before * (1 + 1e-12)) {
      fail.push(`[${c.task}] snapping DEGRADED a non-closed-form constant: ${before.toExponential(3)} -> ${r.score.toExponential(3)}`);
    } else {
      console.log(`  ok  ${c.task.padEnd(14)} safety: ${before.toExponential(2)} -> ${r.score.toExponential(2)} (never worse)`);
    }
  }

  // roundConst must not reintroduce f64 dust: preserving full precision for
  // closed forms let 2 drift to 2.0000000000000004, which desynchronised 17
  // stored formulas from their ASTs.
  for (const [v, want] of [[2.0000000000000004, 2], [3, 3], [-1.0000000000000002, -1]] as [number, number][]) {
    const got = roundConst(v);
    if (got !== want) fail.push(`roundConst(${v}) = ${got}, expected ${want} — f64 dust would desync stored formulas`);
  }

  if (fail.length) {
    console.error("\n✗ test-const-snapping FAILED\n");
    for (const f of fail) console.error("  " + f);
    process.exit(1);
  }
  console.log("test-const-snapping: all assertions passed");
}

main();
