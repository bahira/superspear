// Write MEASURED wall-clock speedups into the ledger.
//
// Until now `speed.speedup` was purely a cost-model prediction, and the model
// is known to be wrong on specific kernels: logsumexp2 advertises ×8.57 but
// measures ×2.47, and light_falloff_punctual advertises ×1.54 but measures
// ×0.73 — i.e. the "fast" kernel is SLOWER than the law it replaces. Publishing
// only the model means publishing those two numbers as if they were results.
//
// This script runs the wall-clock harness and records, for every task where a
// real measurement exists, a `speed.measured` block. Consumers (README, npm
// package, audit) can then prefer measured over modelled, and the gap becomes
// visible instead of hidden.
//
// It never invents a measurement: tasks the harness skips (no compilable
// reference) or marks unresolvable (difference below the timer floor) are left
// with the model only, explicitly flagged as unvalidated.
//
// Usage: npx tsx scripts/write-measured-speed.ts [--dry]
import { execFileSync } from "node:child_process";
import { loadLedger, saveLedger } from "../src/lib/spear/ledger";

interface Measured {
  measuredSpeedup: number;
  nsFormula: number;
  nsExact: number;
  agreement: number;
  resolvable: boolean;
  harness: string;
}

function main(): void {
  const dry = process.argv.includes("--dry");

  const raw = execFileSync("npx", ["tsx", "scripts/bench-wallclock.ts", "--json"], {
    encoding: "utf8",
    cwd: process.cwd(),
    maxBuffer: 1 << 24,
  });
  const rows: {
    id: string;
    bound: boolean;
    nsFormula: number;
    nsExact: number;
    measured: number;
    modelSpeedup: number;
    agreement: number;
  }[] = JSON.parse(raw).rows;

  const ledger = loadLedger();
  const changes: string[] = [];

  for (const r of rows) {
    const entry = ledger[r.id];
    if (!entry?.speed) continue;
    const block: Measured = {
      measuredSpeedup: Number(r.measured.toFixed(4)),
      nsFormula: Number(r.nsFormula.toFixed(4)),
      nsExact: Number(r.nsExact.toFixed(4)),
      agreement: Number((r.measured / r.modelSpeedup).toFixed(4)),
      // `bound` = both kernels sit at the call-overhead floor, so the ratio is
      // an artifact of the harness, not a property of the formulas.
      resolvable: !r.bound,
      harness: "gcc -O2 · scalar x86-64 · median of 9×40 passes",
    };
    // Wall-clock numbers jitter run to run, so exact equality would make this
    // script report a change every time and turn the CI sync-check into noise.
    // Only a MATERIAL move counts. Threshold is set from OBSERVED run-to-run
    // jitter on this harness, not guessed: repeating the bench three times
    // moved lennard_jones by ~4% and kdv_soliton by ~3.5%, so 5% sat inside
    // the noise band and made the check flap. 12% clears measured jitter while
    // still catching the regressions that matter (logsumexp2 model-vs-measured
    // was a 3.5x discrepancy; kerr and light_falloff_punctual flipped sign).
    const prev = (entry.speed as Record<string, unknown>).measured as Measured | undefined;
    if (prev) {
      const drift = Math.abs(prev.measuredSpeedup - block.measuredSpeedup) / Math.max(prev.measuredSpeedup, 1e-9);
      // `resolvable` is NOT part of the stability test: call-overhead-bound
      // kernels sit exactly on the floor threshold and flip it on pure noise
      // (layernorm_scale, logistic_growth). What must stay stable is whether
      // the kernel is genuinely faster or slower than the law it replaces.
      // Deliberately NOT part of the stability test:
      //   • `resolvable` — call-overhead-bound kernels sit on the floor
      //     threshold and flip it on pure noise (layernorm_scale);
      //   • which side of 1.0 a kernel lands on — kernels measuring ~0.99-1.02
      //     straddle it every run.
      // Both made a byte-equality check flap between 1 and 4 "changes" on an
      // unmodified tree. Timing data cannot be gated by equality; the real
      // invariant (no kernel advertises a speedup it does not have) is enforced
      // by audit-champions' `speedup-not-real` rule instead.
      if (drift <= 0.12) continue;
    }
    (entry.speed as Record<string, unknown>).measured = block;
    const verdict = !block.resolvable
      ? "unresolvable (call-overhead bound)"
      : block.agreement > 1.5 || block.agreement < 0.667
        ? `MODEL OFF (agreement ${block.agreement})`
        : "ok";
    changes.push(
      `[${r.id}] measured ×${block.measuredSpeedup} vs modelled ×${r.modelSpeedup.toFixed(2)} — ${verdict}`,
    );
  }

  for (const c of changes) console.log(c);
  console.log(`\n${changes.length} entries updated${dry ? " (dry run, nothing written)" : ""}`);
  if (!dry && changes.length) saveLedger(ledger);
}

main();
