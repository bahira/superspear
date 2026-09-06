// Demote strawman vs-iterative multipliers.
//
// `iterativeBaseline` is contractually "how practitioners compute this WITHOUT
// a closed form". For 8 tasks that premise was simply untrue: a closed-form
// reference kernel exists and is the thing a real implementation calls. Pricing
// the champion against a 1000-draw Monte-Carlo or a 300-step RKF45 then buys an
// enormous multiplier from a baseline nobody would run.
//
// The two numbers are not even measuring comparable things: 1000-draw
// Monte-Carlo for Φ(x) lands at MSE ≈ 9e-5, the champion at 1e-34. A cost ratio
// between kernels separated by 29 orders of magnitude of accuracy is a category
// error, not a speedup.
//
// This pass does NOT delete the information — a solver comparison is legitimate
// context for someone who genuinely has no closed form. It moves it out of the
// headline `speed.vsIterative` slot into `speed.vsSolverContext`, annotated with
// the honest like-for-like ratio, so no generator or README can pick it up as
// the primary speed claim.
//
// Usage: npx tsx scripts/fix-strawman-baselines.ts [--dry]
import { loadLedger, saveLedger } from "../src/lib/spear/ledger";
import { buildTasks } from "../src/lib/spear/benchmarks";
import { estimateCost } from "../src/lib/spear/engine";

const dry = process.argv.includes("--dry");
const ledger = loadLedger();
const defs = new Map(buildTasks().map((t) => [t.id, t]));
const changes: string[] = [];

for (const [id, entry] of Object.entries(ledger) as [string, any][]) {
  const def = defs.get(id);
  if (!def?.iterativeBaseline) continue;
  const hasClosedForm = def.exactCost !== undefined || def.exactRefNode !== undefined;
  if (!hasClosedForm) continue; // legitimate: no closed form exists

  const exactCost = def.exactCost ?? (def.exactRefNode ? estimateCost(def.exactRefNode) : undefined);

  for (const slot of [entry.speed, entry.fast?.speed, entry.fast] as any[]) {
    if (!slot?.vsIterative) continue;
    const claimed = slot.vsIterative.speedup as number;
    const cost = slot.formulaCost ?? entry.speed?.formulaCost;
    const honest = exactCost && cost ? exactCost / Math.max(1, cost) : undefined;
    slot.vsSolverContext = {
      ...slot.vsIterative,
      note:
        "Context only — a closed-form reference exists for this task, so the solver is not what a real implementation runs. Not a like-for-like speedup: the solver's accuracy is orders of magnitude worse.",
      honestSpeedupVsReference: honest,
    };
    delete slot.vsIterative;
    changes.push(
      `[${id}] demoted ×${claimed.toFixed(1)} vs "${slot.vsSolverContext.label}" -> context (honest ×${honest?.toFixed(2) ?? "?"})`,
    );
  }
}

if (dry) {
  console.log(changes.join("\n"));
  console.log(`\n${changes.length} changes (dry run)`);
} else {
  saveLedger(ledger);
  console.log(changes.join("\n"));
  console.log(`\n${changes.length} changes written`);
}
