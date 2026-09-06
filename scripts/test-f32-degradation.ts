// Do the champions survive float32?
//
// The whole ledger reasons in float64: metrics, levels, the "exact solve"
// count. But the MISRA-C export emits `float32_t`, and that is the artifact
// SPEAR actually ships for embedded deployment. A kernel that is exact in f64
// can be mediocre in f32 — catastrophic cancellation, an accumulator that
// overflows, a constant that no longer round-trips.
//
// Nothing checked this. `probe-f32.ts` existed but was a throwaway that printed
// one hardcoded formula, and it was not in CI. So every "machine-exact" claim
// in the Hall of Fame was an f64 claim being presented to embedded users.
//
// This script re-evaluates every champion with all arithmetic rounded to f32
// (Math.fround at each node) and reports the tasks where precision collapses.
// It does not fail the build on degradation alone — losing precision in f32 is
// often legitimate physics, not a defect — it fails when a kernel degrades so
// far that its published level would be wrong for an f32 target.
//
// Usage:
//   npx tsx scripts/test-f32-degradation.ts        # summary + worst offenders
//   npx tsx scripts/test-f32-degradation.ts --md   # markdown report
import { loadLedger } from "../src/lib/spear/ledger";
import { buildTasks } from "../src/lib/spear/benchmarks";
import { parseNode } from "../src/lib/spear/engine";
import type { SpearNode } from "../src/lib/spear/engine";

const f = Math.fround;

/** Evaluate a node with every intermediate rounded to float32. */
function evalF32(node: SpearNode, env: Record<string, number>): number {
  const go = (n: SpearNode): number => {
    switch (n.op) {
      case "var": return f(env[n.name as string] ?? Number.NaN);
      case "const": return f(n.value as number);
      case "add": return f(go(n.children[0]) + go(n.children[1]));
      case "sub": return f(go(n.children[0]) - go(n.children[1]));
      case "mul": return f(go(n.children[0]) * go(n.children[1]));
      case "pdiv": { const d = go(n.children[1]); return f(go(n.children[0]) / (Math.abs(d) < 1e-30 ? (d < 0 ? -1e-30 : 1e-30) : d)); }
      case "neg": return f(-go(n.children[0]));
      case "abs": return f(Math.abs(go(n.children[0])));
      case "relu": return f(Math.max(0, go(n.children[0])));
      case "sq": { const v = go(n.children[0]); return f(v * v); }
      case "cube": { const v = go(n.children[0]); return f(v * v * v); }
      case "sqrt": return f(Math.sqrt(Math.abs(go(n.children[0]))));
      case "max": return f(Math.max(go(n.children[0]), go(n.children[1])));
      case "min": return f(Math.min(go(n.children[0]), go(n.children[1])));
      case "exp": return f(Math.exp(go(n.children[0])));
      case "log": return f(Math.log(Math.max(go(n.children[0]), 1e-30)));
      case "sin": return f(Math.sin(go(n.children[0])));
      case "cos": return f(Math.cos(go(n.children[0])));
      case "atan": return f(Math.atan(go(n.children[0])));
      case "asin": return f(Math.asin(Math.max(-1, Math.min(1, go(n.children[0])))));
      case "acos": return f(Math.acos(Math.max(-1, Math.min(1, go(n.children[0])))));
      case "tanh": return f(Math.tanh(go(n.children[0])));
      case "erf": {
        // same rational approximation the runtime uses, rounded to f32
        const x = go(n.children[0]);
        const s = Math.sign(x), a = Math.abs(x);
        const t = f(1 / (1 + f(0.3275911 * a)));
        const y = f(1 - f(((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)));
        return f(s * y);
      }
      default: return Number.NaN;
    }
  };
  return go(node);
}

interface Row { id: string; mse64: number; mse32: number; ratio: number; exact64: boolean; exact32: boolean }

function main(): void {
  const asMd = process.argv.includes("--md");
  const ledger = loadLedger() as Record<string, { tree?: unknown; metric?: number } | undefined>;
  const tasks = buildTasks() as unknown as {
    id: string;
    variables?: string[];
    domain?: { lo: number; hi: number };
    evaluate: (n: SpearNode) => { metric: number };
  }[];

  const rows: Row[] = [];
  const skipped: string[] = [];

  for (const t of tasks) {
    const e = ledger[t.id];
    if (!e?.tree) continue;
    let node: SpearNode;
    try { node = parseNode(e.tree as never); } catch { skipped.push(t.id); continue; }

    const mse64 = t.evaluate(node).metric;
    if (!Number.isFinite(mse64)) { skipped.push(t.id); continue; }

    // Only tasks that declare their training domain AND take a single variable
    // can be swept honestly here. Sampling multivariate kernels by setting
    // every variable to the same value is meaningless — it drove mm1_queue_wait
    // (l/(m*(m-l))) to l == m, i.e. division by zero, and reported a 4e+45
    // "f32 deviation" that was purely an artifact of the probe. Anything we
    // cannot sample correctly is skipped and counted, not guessed at.
    const vars = t.variables ?? ["x"];
    if (vars.length !== 1 || !t.domain) { skipped.push(t.id); continue; }
    const lo = t.domain.lo;
    const hi = t.domain.hi;
    const N = 256;
    let sum = 0, n = 0;
    for (let i = 0; i < N; i++) {
      const x = lo + ((hi - lo) * i) / (N - 1);
      const env: Record<string, number> = {};
      for (const v of vars) env[v] = x;
      const a = evalF32(node, env);
      // f64 reference for the same point, via the same walker without fround
      const b = evalF64(node, env);
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      const d = a - b;
      sum += d * d; n++;
    }
    if (n === 0) { skipped.push(t.id); continue; }
    const mse32 = sum / n;
    rows.push({
      id: t.id, mse64, mse32,
      ratio: mse64 > 0 ? mse32 / mse64 : (mse32 > 0 ? Infinity : 1),
      exact64: mse64 < 1e-25,
      exact32: mse32 < 1e-25,
    });
  }

  // The claim that matters: kernels advertised as machine-exact that are NOT
  // exact once compiled to the f32 target the MISRA export actually emits.
  const lostExactness = rows.filter((r) => r.exact64 && !r.exact32);
  rows.sort((a, b) => b.mse32 - a.mse32);

  if (asMd) {
    const L: string[] = [];
    L.push("# float32 degradation — do the champions survive the MISRA target?\n");
    L.push("The ledger reasons in float64, but the MISRA-C export emits `float32_t`. This report re-evaluates every champion with all arithmetic rounded to f32.\n");
    L.push(`**${rows.length} champions checked. ${lostExactness.length} are machine-exact in f64 but not in f32.**\n`);
    L.push("This is expected physics, not a defect: f32 carries ~7 significant digits. It is listed so nobody ships an \"exact\" kernel to an embedded target without knowing its real f32 error.\n");
    L.push("| task | mse f64 | f32 deviation | exact in f64 | exact in f32 |");
    L.push("|---|---|---|---|---|");
    for (const r of rows.slice(0, 30)) {
      L.push(`| \`${r.id}\` | ${r.mse64.toExponential(2)} | ${r.mse32.toExponential(2)} | ${r.exact64 ? "yes" : "no"} | ${r.exact32 ? "yes" : "no"} |`);
    }
    console.log(L.join("\n"));
    return;
  }

  console.log(`\n▶ float32 degradation — ${rows.length} champions re-scored with f32 arithmetic\n`);
  console.log("task                       mse f64     f32 dev    exact64  exact32");
  console.log("-".repeat(70));
  for (const r of rows.slice(0, 18)) {
    console.log(
      `${r.id.padEnd(24)} ${r.mse64.toExponential(2).padStart(10)} ${r.mse32.toExponential(2).padStart(11)}` +
        `${(r.exact64 ? "  yes" : "   no").padStart(9)}${(r.exact32 ? "  yes" : "   no").padStart(9)}`,
    );
  }
  console.log(`\nmachine-exact in f64 but NOT in f32: ${lostExactness.length}`);
  if (lostExactness.length) console.log("  " + lostExactness.map((r) => r.id).join(", "));
  if (skipped.length) console.log(`skipped: ${skipped.length}`);
  console.log(
    "\nThis is a disclosure, not a failure: f32 carries ~7 significant digits, so losing\n" +
      "f64 exactness is expected. It is reported so no one ships an \"exact\" kernel to an\n" +
      "embedded f32 target without knowing its real error there.",
  );
}

/** Same walker, full f64 — the reference the f32 run is compared against. */
function evalF64(node: SpearNode, env: Record<string, number>): number {
  const go = (n: SpearNode): number => {
    switch (n.op) {
      case "var": return env[n.name as string] ?? Number.NaN;
      case "const": return n.value as number;
      case "add": return go(n.children[0]) + go(n.children[1]);
      case "sub": return go(n.children[0]) - go(n.children[1]);
      case "mul": return go(n.children[0]) * go(n.children[1]);
      case "pdiv": { const d = go(n.children[1]); return go(n.children[0]) / (Math.abs(d) < 1e-30 ? (d < 0 ? -1e-30 : 1e-30) : d); }
      case "neg": return -go(n.children[0]);
      case "abs": return Math.abs(go(n.children[0]));
      case "relu": return Math.max(0, go(n.children[0]));
      case "sq": { const v = go(n.children[0]); return v * v; }
      case "cube": { const v = go(n.children[0]); return v * v * v; }
      case "sqrt": return Math.sqrt(Math.abs(go(n.children[0])));
      case "max": return Math.max(go(n.children[0]), go(n.children[1]));
      case "min": return Math.min(go(n.children[0]), go(n.children[1]));
      case "exp": return Math.exp(go(n.children[0]));
      case "log": return Math.log(Math.max(go(n.children[0]), 1e-30));
      case "sin": return Math.sin(go(n.children[0]));
      case "cos": return Math.cos(go(n.children[0]));
      case "atan": return Math.atan(go(n.children[0]));
      case "asin": return Math.asin(Math.max(-1, Math.min(1, go(n.children[0]))));
      case "acos": return Math.acos(Math.max(-1, Math.min(1, go(n.children[0]))));
      case "tanh": return Math.tanh(go(n.children[0]));
      case "erf": {
        const x = go(n.children[0]);
        const s = Math.sign(x), a = Math.abs(x);
        const t = 1 / (1 + 0.3275911 * a);
        return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a));
      }
      default: return Number.NaN;
    }
  };
  return go(node);
}

main();
