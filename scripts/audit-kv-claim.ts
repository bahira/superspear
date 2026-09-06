// Audit the KV-cache retention claim.
//
// The README and three reports headline "80.31% retention on real distilgpt2
// attention, beats H2O". That number came from an external torch run and was
// never re-checked. Reading the stored sweep (validation/kv-multiscale-results.json)
// shows two problems that the headline hides:
//
//   1. CHERRY-PICKED BUDGET. The sweep covers caps 64/128/256/320. The claim
//      quotes only cap=320 — and at cap=64 H2O actually BEATS spear
//      (45.94 vs 45.60). No report mentions that.
//
//   2. THE QUOTED REGIME IS THE ONE WHERE POLICY MATTERS LEAST. At cap=320 a
//      RANDOM policy already retains 73.8%, so the headline's 80.3% sits only
//      6.5 points above chance. At cap=64 random gets 17.2% and the margin is
//      28.4 points — that is where the policy is actually doing work, and it
//      is the number not quoted.
//
// The margin over H2O is +0.118 points on n=72, roughly 500x smaller than the
// spread of the samples themselves (68.5 points). Without per-sample standard
// deviations that is not a demonstrated win, it is a tie.
//
// This script replays those checks from the stored JSON — no torch needed — so
// the claim is verifiable in CI instead of resting on an unrepeatable run.
//
// Usage: npx tsx scripts/audit-kv-claim.ts
import { readFileSync } from "node:fs";
import { join } from "node:path";

interface Stat { mean: number; median: number; min: number; max: number; n: number }
type Cap = Record<string, Stat>;

function main(): void {
  const path = join(process.cwd(), "validation", "kv-multiscale-results.json");
  const data = JSON.parse(readFileSync(path, "utf8")) as {
    models: Record<string, Record<string, Cap>>;
    caps: number[];
  };

  const failures: string[] = [];
  for (const [model, byCap] of Object.entries(data.models)) {
    const caps = Object.keys(byCap).sort((a, b) => Number(a) - Number(b));
    console.log(`\n▶ ${model} — retention (%) by context cap, n=${byCap[caps[0]].spear.n} samples each\n`);
    console.log("cap     spear     h2o    delta   random   oracle   margin over random");
    console.log("-".repeat(74));

    let spearWins = 0;
    for (const c of caps) {
      const s = byCap[c].spear, h = byCap[c].h2o, r = byCap[c].random, o = byCap[c].oracle;
      const delta = s.mean - h.mean;
      if (delta > 0) spearWins++;
      console.log(
        `${c.padEnd(6)} ${s.mean.toFixed(2).padStart(6)} ${h.mean.toFixed(2).padStart(7)} ` +
          `${(delta >= 0 ? "+" : "") + delta.toFixed(3)}`.padStart(9) +
          ` ${r.mean.toFixed(2).padStart(8)} ${o.mean.toFixed(2).padStart(8)}` +
          `${(s.mean - r.mean).toFixed(2).padStart(12)}${delta < 0 ? "   <- H2O WINS" : ""}`,
      );
    }

    // The honest reading, asserted so it cannot quietly rot.
    const best = byCap[caps[caps.length - 1]];
    const worst = byCap[caps[0]];
    const marginBest = best.spear.mean - best.h2o.mean;
    const spreadBest = best.spear.max - best.spear.min;

    console.log(
      `\n  spear beats H2O at ${spearWins}/${caps.length} budgets. ` +
        `At cap=${caps[0]} H2O leads by ${(worst.h2o.mean - worst.spear.mean).toFixed(3)} points.`,
    );
    console.log(
      `  Headline margin at cap=${caps[caps.length - 1]}: +${marginBest.toFixed(3)} points, ` +
        `against a sample spread of ${spreadBest.toFixed(1)} points (ratio ${(marginBest / spreadBest).toFixed(4)}).`,
    );
    console.log(
      `  At that cap a RANDOM policy already retains ${best.random.mean.toFixed(1)}%, ` +
        `so the headline sits ${(best.spear.mean - best.random.mean).toFixed(1)} points above chance;` +
        ` at cap=${caps[0]} that margin is ${(worst.spear.mean - worst.random.mean).toFixed(1)} points.`,
    );

    // Guard: the docs must not claim an unqualified win while a budget loses.
    if (spearWins < caps.length) {
      const claimFiles = ["README.md", join("REPORTS", "INDEX.md"), join("REPORTS", "breakthroughs.md")];
      for (const f of claimFiles) {
        let txt = "";
        try { txt = readFileSync(join(process.cwd(), f), "utf8"); } catch { continue; }
        if (!txt.includes("80.31")) continue;
        // The number may be quoted, but only alongside the caveat.
        const hasCaveat = /cap=64|budget|H2O l'emporte|H2O wins|tie|égalité/i.test(txt);
        if (!hasCaveat) {
          failures.push(`${f} quotes 80.31% without noting that H2O wins at cap=64`);
        }
      }
    }
  }

  if (failures.length) {
    console.error("\n✗ KV-cache claim is stated more strongly than the data supports\n");
    for (const f of failures) console.error("  " + f);
    console.error("\nQuote the full sweep, or qualify the number.");
    process.exit(1);
  }
  console.log("\naudit-kv-claim: the published claim matches the stored sweep, caveats included");
}

main();
