// Regression test for the reproducibility bug that made the hall of fame
// meaningless: dataset noise used to be drawn from the SEARCH PRNG, so
//   • every seed scored its champion on a different realisation of the data
//     ("best across all seeds" was partly a lottery on friendly noise), and
//   • re-evaluating a stored AST gave a different number than the record
//     (free_fall drifted across 2.9e-4 … 5.1e-4 for one fixed formula).
//
// Invariant enforced here: a champion's metric is a property of the formula
// alone — identical under any search seed, any construction order, any run.
import { setSeed, rand, parseNode } from "../src/lib/spear/engine";
import { setUniformSource } from "../src/lib/spear/math-utils";
import { buildTasks } from "../src/lib/spear/benchmarks";
import { loadLedger } from "../src/lib/spear/ledger";

const NOISY = ["free_fall", "kepler", "rc_circuit", "european_call", "kv_cache", "temporal_grad"];

function metricsUnderSeed(seed: number | null, ids: string[]): Map<string, number> {
  if (seed !== null) {
    setSeed(seed);
    // Deliberately hostile: point the injectable source at the search PRNG,
    // exactly as the loop does. Datasets must ignore it.
    setUniformSource(rand);
  }
  const ledger = loadLedger();
  const defs = new Map(buildTasks().map((t) => [t.id, t]));
  const out = new Map<string, number>();
  for (const id of ids) {
    const def = defs.get(id) as { evaluate: (n: never) => { metric: number } } | undefined;
    const entry = ledger[id] as { tree?: unknown } | undefined;
    if (!def || !entry?.tree) continue;
    out.set(id, def.evaluate(parseNode(entry.tree as never) as never).metric);
  }
  return out;
}

const ids = NOISY.filter((id) => loadLedger()[id]);
const ref = metricsUnderSeed(null, ids);
const failures: string[] = [];

// 1. invariance across search seeds + repeated construction
for (const seed of [4242, 777, 3333, 930202]) {
  const got = metricsUnderSeed(seed, ids);
  for (const [id, m] of ref) {
    const g = got.get(id);
    if (g === undefined) continue;
    const same = Object.is(m, g) || (Number.isFinite(m) && Number.isFinite(g) && Math.abs(m - g) <= Math.abs(m) * 1e-12);
    if (!same) failures.push(`[${id}] seed ${seed}: ${g} != ${m} — dataset depends on the search seed`);
  }
}

// 2. invariance across repeated buildTasks() in one process (construction order)
for (let pass = 0; pass < 3; pass++) {
  const got = metricsUnderSeed(null, ids);
  for (const [id, m] of ref) {
    const g = got.get(id)!;
    if (!Object.is(m, g)) failures.push(`[${id}] pass ${pass}: ${g} != ${m} — dataset depends on construction order`);
  }
}

if (failures.length) {
  console.error("dataset determinism FAILED:");
  for (const f of failures) console.error("  ✗ " + f);
  process.exit(1);
}
console.log(`test-dataset-determinism: ${ids.length} noisy tasks stable across 4 seeds and 3 rebuilds`);
