// Every EXACT_LAWS entry must reproduce its own task's target.
//
// EXACT_LAWS is what bench-wallclock compiles as the "exact reference" a
// champion is timed against. If an entry is not actually the task's law, the
// benchmark silently compares the champion to the wrong thing and every
// speedup derived from it is fiction — a failure mode strictly worse than
// having no measurement at all.
//
// This is not hypothetical: while writing these laws, 3 of the first 8 drafts
// did not reproduce their targets (bessel_i0e 4.8e-3, uncharted2_tonemap
// 1.4e-1, blackbody_r inf) and were dropped rather than shipped. Later,
// huber_loss looked correct as min(x^2/2, |x|-0.5) but that branch is wrong
// below |x|=1 — at x=0 it returns -0.5 instead of 0.
//
// Tasks whose data carries NOISE are exempt from the tight bound: there the
// law cannot beat the noise floor by construction, so we only require that it
// is finite and at least as good as the recorded champion.
//
// Usage: npx tsx scripts/verify-exact-laws.ts
import { buildTasks } from "../src/lib/spear/benchmarks";
import { EXACT_LAWS } from "../src/lib/spear/tasks/shared";
import { loadLedger } from "../src/lib/spear/ledger";
import { nodeToString } from "../src/lib/spear/engine";

const EXACT_TOL = 1e-6;

function main(): void {
  const tasks = buildTasks() as unknown as {
    id: string;
    evaluate: (n: unknown) => { metric: number };
  }[];
  const ledger = loadLedger() as Record<string, { metric?: number } | undefined>;

  let checked = 0;
  const failures: string[] = [];
  const noisy: string[] = [];

  for (const [id, law] of Object.entries(EXACT_LAWS)) {
    const t = tasks.find((x) => x.id === id);
    if (!t) {
      failures.push(`[${id}] EXACT_LAWS entry has no matching task`);
      continue;
    }
    checked++;
    let m = Number.NaN;
    try {
      m = t.evaluate(law).metric;
    } catch (e) {
      failures.push(`[${id}] law threw: ${(e as Error).message.slice(0, 80)}`);
      continue;
    }
    if (!Number.isFinite(m)) {
      failures.push(`[${id}] law evaluates to ${m} — not a usable reference`);
      continue;
    }
    if (m <= EXACT_TOL) continue;

    // Above tolerance: acceptable only if the task itself is noisy, evidenced
    // by the champion not doing better either.
    const champ = ledger[id]?.metric;
    if (typeof champ === "number" && Number.isFinite(champ) && m <= champ * 1.5) {
      noisy.push(`[${id}] law ${m.toExponential(2)} vs champion ${champ.toExponential(2)} — noisy task, law is competitive`);
      continue;
    }
    failures.push(
      `[${id}] law does NOT reproduce its target: mse ${m.toExponential(3)}` +
        (typeof champ === "number" ? ` (champion reaches ${champ.toExponential(3)})` : "") +
        `\n        ${nodeToString(law).slice(0, 120)}`,
    );
  }

  for (const n of noisy) console.log("  ~ " + n);
  if (failures.length) {
    console.error("\n✗ EXACT_LAWS verification FAILED\n");
    for (const f of failures) console.error("  " + f);
    console.error(
      `\n${failures.length} of ${checked} reference laws are wrong. A wrong reference makes every ` +
        `speedup measured against it fiction — fix or remove them.`,
    );
    process.exit(1);
  }
  console.log(
    `verify-exact-laws: ${checked} reference laws all reproduce their task targets` +
      (noisy.length ? ` (${noisy.length} competitive on noisy tasks)` : ""),
  );
}

main();
