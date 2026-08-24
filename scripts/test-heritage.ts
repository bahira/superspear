// Self-check for the heritage upgrade (OOD constraint + asin op + dual-pop
// loop plumbing). Run via: npx tsx scripts/test-heritage.ts
import {
  dominates,
  makeNode,
  nodeToString,
  parseFormula,
  evaluateScalar,
  toC,
} from "../src/lib/spear/engine";
import { compositeSeeds, makeOodProbe, OOD_KAPPA } from "../src/lib/spear/heritage";
import { buildTasks } from "../src/lib/spear/benchmarks";
import { runGroundedLoop } from "../src/lib/spear/loop";

let failures = 0;
function check(cond: boolean, label: string): void {
  if (!cond) {
    failures++;
    console.error(`FAIL: ${label}`);
  }
}

// ---- 1. asin op: eval clamp + round-trip through the printer/parser
const asinTree = parseFormula("asin(x)");
check(nodeToString(asinTree) === "asin(x)", `asin print round-trip: ${nodeToString(asinTree)}`);
check(Math.abs(evaluateScalar(asinTree, { x: 5 }) - Math.PI / 2) < 1e-12, "asin clamps domain at +1");
check(Math.abs(evaluateScalar(asinTree, { x: -1 }) + Math.PI / 2) < 1e-12, "asin(-1) = -pi/2");
check(toC(asinTree).includes("asinf"), "asin emits C");

// ---- 2. constrained NSGA-II semantics
check(
  dominates({ fitness: 1, size: 9 }, { fitness: 100, size: 1, violation: 0.2 }),
  "feasible must dominate infeasible",
);
check(
  !dominates({ fitness: 100, size: 1, violation: 0.2 }, { fitness: 1, size: 9 }),
  "infeasible must not dominate feasible",
);
check(
  dominates({ fitness: 1, size: 9, violation: 0.1 }, { fitness: 1, size: 9, violation: 0.5 }),
  "smaller violation wins among infeasible",
);

// ---- 3. OOD probe: honest shape passes, off-support blow-up dies
const grid = (lo: number, hi: number, n: number): Float64Array =>
  Float64Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
const trainX = grid(0, 3, 64);
const sinOf = (a: Float64Array): Float64Array => Float64Array.from(a, (v) => Math.sin(v));
const probe = makeOodProbe(
  { vars: { x: trainX }, y: sinOf(trainX), n: trainX.length },
  { vars: { x: grid(3, 4.5, 32) }, y: sinOf(grid(3, 4.5, 32)), n: 32 },
);
const good = parseFormula("sin(x)");
check(probe(good) === 0, `exact law must be feasible, got ${probe(good)}`);
const bad = parseFormula("(x * x * x * x)");
check(Number.isFinite(probe(bad)) === false || probe(bad) > 0, `x⁴ must fail OOD band, got ${probe(bad)}`);

// ---- 4. task wiring: probes present on both families, composites seeded
const defs = buildTasks();
const withOod = defs.filter((d) => d.ood);
check(defs.length > 40, `expected the full task set, got ${defs.length}`);
check(withOod.length >= defs.length * 0.6, `OOD coverage too thin: ${withOod.length}/${defs.length}`);
check(compositeSeeds("t").length === 5 && compositeSeeds("t")[0].size > 2, "composite seeds built");

// ---- 5. end-to-end loop: A/B populations + migration must not crash and
// must still produce valid champions
async function main(): Promise<void> {
  const loop = await runGroundedLoop({ seed: 42, budget: 60, deadlineMs: 25_000 });
  check(loop.status !== "running", `loop finished (${loop.status})`);
  check(loop.tasks.length === defs.length, "all tasks snapshotted");
  const solved = loop.tasks.filter((t) => t.best && Number.isFinite(t.best.metric)).length;
  check(solved >= defs.length * 0.5, `too few tasks produced a champion: ${solved}/${defs.length}`);
  console.log(`loop smoke: ${solved}/${defs.length} champions, ${loop.breakthroughs.length} breakthroughs`);

  if (failures > 0) {
    console.error(`test-heritage: ${failures} failure(s)`);
    process.exit(1);
  }
  console.log("test-heritage: all assertions passed");
}
void main();
