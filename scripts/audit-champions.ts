// Independent champion audit — recomputes EVERYTHING from the stored ASTs.
//
// For every ledger entry we re-derive, without trusting a single stored number:
//   • metric      : task.evaluate(tree)         vs entry.metric      (drift?)
//   • level       : milestones(metric)          vs entry.level       (inflated?)
//   • formula text: nodeToString(tree)          vs entry.formula     (desync?)
//   • cost/speed  : estimateCost(tree)          vs entry.speed.*     (stale?)
//   • holdout     : train/test split metric     -> overfit ratio
//   • ood         : extrapolation-band violation (finite? blows up?)
//   • r2          : deployability grade of the fast slot
//   • fast slot   : cheaper AND validated AND consistent with the champion
//
// Usage:
//   npx tsx scripts/audit-champions.ts              # table + summary
//   npx tsx scripts/audit-champions.ts --json       # machine-readable
//   npx tsx scripts/audit-champions.ts --strict     # exit 1 on any FAIL
//   npx tsx scripts/audit-champions.ts --md > REPORTS/champion-audit.md
import { loadLedger } from "../src/lib/spear/ledger";
import { EXACT_LAWS } from "../src/lib/spear/tasks/shared";
import { buildTasks } from "../src/lib/spear/benchmarks";
import { parseNode, estimateCost, nodeToString, countOps } from "../src/lib/spear/engine";
import type { SpearNode } from "../src/lib/spear/engine";
import type { TaskDef } from "../src/lib/spear/tasks/types";

const REL_TOL = 1e-6;    // relative tolerance on metric reproduction
const ABS_FLOOR = 1e-30; // below this, metrics are "exact" and only compared loosely

type Severity = "ok" | "info" | "warn" | "fail";

interface Issue { code: string; severity: Severity; detail: string }

interface Row {
  id: string;
  hasTask: boolean;
  storedMetric?: number;
  recomputedMetric?: number;
  metricDrift?: number;
  storedLevel?: number;
  recomputedLevel?: number;
  cost?: number;
  storedCost?: number;
  speedup?: number;
  /** "search" = found by the loop; "recovered-seed" = textually the seeded law. */
  provenance?: "search" | "recovered-seed";
  holdout?: number;
  overfitRatio?: number;
  ood?: number;
  r2?: number;
  fastCost?: number;
  fastMetric?: number;
  transcendental?: number;
  issues: Issue[];
}

function levelOf(t: TaskDef, metric: number): number {
  let lvl = 0;
  for (const m of t.milestones) if (Number.isFinite(metric) && m.test(metric)) lvl = Math.max(lvl, m.level);
  return lvl;
}

function relDiff(a: number, b: number): number {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return Number.isFinite(a) === Number.isFinite(b) ? 0 : Infinity;
  const scale = Math.max(Math.abs(a), Math.abs(b), ABS_FLOOR);
  return Math.abs(a - b) / scale;
}

function safe<T>(fn: () => T): T | undefined {
  try { return fn(); } catch { return undefined; }
}

function auditEntry(id: string, entry: Record<string, any>, def: TaskDef | undefined): Row {
  const row: Row = { id, hasTask: !!def, issues: [] };
  const push = (code: string, severity: Severity, detail: string) => row.issues.push({ code, severity, detail });

  if (!def) {
    push("orphan-entry", "warn", "ledger entry has no matching task definition — unverifiable, cannot be reproduced");
    return row;
  }
  if (!entry.tree) {
    push("no-ast", "fail", "no serialized AST — the record is prose, not a verifiable artifact");
    return row;
  }

  let node: SpearNode;
  try { node = parseNode(entry.tree); }
  catch (e) { push("bad-ast", "fail", `AST unparseable: ${(e as Error).message}`); return row; }

  // ---- formula text ↔ AST desync
  const text = nodeToString(node);
  if (entry.formula && entry.formula.replace(/\s/g, "") !== text.replace(/\s/g, "")) {
    push("formula-desync", "warn", `stored text != AST render\n      stored: ${entry.formula}\n      ast:    ${text}`);
  }

  // ---- metric reproduction
  row.storedMetric = entry.metric;
  const ev = safe(() => def.evaluate(node));
  if (!ev) { push("eval-throw", "fail", "task.evaluate() threw on the stored AST"); }
  else {
    row.recomputedMetric = ev.metric;
    if (!ev.finite) push("non-finite", "fail", "evaluation is not finite on the training grid");
    if (typeof entry.metric === "number") {
      const d = relDiff(entry.metric, ev.metric);
      row.metricDrift = d;
      if (d > 1e-2) push("metric-drift", "fail", `stored ${entry.metric.toExponential(3)} vs recomputed ${ev.metric.toExponential(3)} (rel ${d.toExponential(1)})`);
      else if (d > REL_TOL) push("metric-drift", "warn", `stored ${entry.metric.toExponential(3)} vs recomputed ${ev.metric.toExponential(3)} (rel ${d.toExponential(1)})`);
    }

    // ---- level integrity
    row.storedLevel = entry.level;
    row.recomputedLevel = levelOf(def, ev.metric);
    if (typeof entry.level === "number" && entry.level > row.recomputedLevel) {
      push("level-inflated", "fail", `claims L${entry.level}, milestones on the recomputed metric only grant L${row.recomputedLevel}`);
    } else if (typeof entry.level === "number" && entry.level < row.recomputedLevel) {
      push("level-stale", "warn", `claims L${entry.level} but earns L${row.recomputedLevel} — under-reported`);
    }
  }

  // ---- cost / speed integrity
  const cost = estimateCost(node);
  row.cost = cost;
  row.storedCost = entry.speed?.formulaCost;
  if (row.storedCost !== undefined && row.storedCost !== cost) {
    push("cost-stale", "warn", `speed.formulaCost=${row.storedCost} but AST costs ${cost}`);
  }
  const exactCost = def.exactCost ?? (def.exactRefNode ? estimateCost(def.exactRefNode) : undefined);
  if (exactCost) {
    const su = exactCost / Math.max(1, cost);
    row.speedup = su;
    // Compare like with like. Where a resolvable wall-clock measurement exists
    // it is deliberately published INSTEAD of the model (speedupSource =
    // "measured"), so diffing it against the model re-derivation flags the
    // intended substitution as staleness — it fired on all 12 measured
    // kernels. Only model-sourced figures are checked for drift here; the
    // model-vs-measured gap is reported separately by `speedup-model-off`.
    const isMeasured = (entry.speed as { speedupSource?: string } | undefined)?.speedupSource === "measured";
    if (!isMeasured && entry.speed?.speedup !== undefined && Math.abs(entry.speed.speedup - su) / Math.max(su, 1e-9) > 0.02) {
      push("speedup-stale", "warn", `stored ×${entry.speed.speedup} vs recomputed ×${su.toFixed(2)}`);
    }
  }
  row.transcendental = countOps(node).transcendental;

  // ---- vs-iterative baselines: is the comparison like-for-like?
  //
  // `iterativeBaseline` is contractually "the way practitioners compute this
  // WITHOUT a closed form". When the task ALSO ships a closed-form reference,
  // that premise is false: nobody runs 1000-draw Monte-Carlo for Φ(x) when
  // erf() exists. Pricing against the solver then buys a huge multiplier from
  // a strawman — gaussian_cdf advertised ×2000 while the honest number against
  // the real reference kernel is ×1.48, an inflation of ~1350×.
  //
  // Worse, the accuracies are not comparable either: 1000-draw Monte-Carlo
  // lands at MSE ≈ 9e-5, while the champion is at 1e-34. A cost ratio between
  // two kernels that differ by 29 orders of magnitude in accuracy is not a
  // speedup, it is a category error.
  const ib = def.iterativeBaseline;
  if (ib) {
    const hasClosedForm = def.exactCost !== undefined || def.exactRefNode !== undefined;
    const claimed = entry.speed?.vsIterative?.speedup as number | undefined;
    if (hasClosedForm && claimed !== undefined) {
      const honest = row.speedup;
      const inflation = honest && honest > 0 ? claimed / honest : undefined;
      push(
        "strawman-baseline",
        inflation !== undefined && inflation > 10 ? "fail" : "warn",
        `advertises ×${claimed.toFixed(1)} vs "${ib.label}", but a closed-form reference exists — honest ratio is ×${honest?.toFixed(2) ?? "?"}` +
          (inflation !== undefined ? ` (inflated ${inflation.toFixed(0)}×)` : ""),
      );
    }
  }

  // ---- did the search DISCOVER this, or recover a seeded law?
  // The task's reference law is in the seed pool (it was a real search wall:
  // the law beat the champion on 39 of 69 tasks before seeding). That is
  // legitimate — a seed still has to survive selection and constant refitting.
  // But a champion that is textually the seeded law is a RECOVERY, not a
  // discovery, and the two must never be reported as the same thing.
  const seededLaw = EXACT_LAWS[id];
  if (seededLaw) {
    const same = nodeToString(node) === nodeToString(seededLaw);
    row.provenance = same ? "recovered-seed" : "search";
    if (same) {
      push(
        "champion-is-seeded-law",
        "info",
        `champion is textually the seeded reference law — a recovery, not a discovery`,
      );
    }
  } else {
    row.provenance = "search";
  }

  // ---- is the fast slot actually fast?
  // A fast slot exists to trade accuracy for speed. If it is not measurably
  // faster, the accuracy loss buys nothing and it should not ship. Measured:
  // smootherstep's slot is x0.25 (four times SLOWER) because its min/max
  // branches cost more than the five pipelined multiplies they replace, which
  // the ALU cost model prices at 4 units vs 12.
  const fastMeasured = (entry.fast as { measured?: { measuredSpeedup: number; resolvable: boolean } } | undefined)?.measured;
  if (fastMeasured?.resolvable) {
    if (fastMeasured.measuredSpeedup < 1) {
      push(
        "fast-slot-slower",
        "fail",
        `fast slot MEASURES ×${fastMeasured.measuredSpeedup.toFixed(2)} — slower than the champion it is supposed to accelerate, while also being less accurate`,
      );
    } else if (fastMeasured.measuredSpeedup < 1.05) {
      push(
        "fast-slot-pointless",
        "warn",
        `fast slot measures ×${fastMeasured.measuredSpeedup.toFixed(2)} — no usable gain, so its accuracy loss buys nothing`,
      );
    }
  }

  // ---- the advertised speedup vs what the hardware actually did
  // A modelled multiplier is a hypothesis. Where write-measured-speed.ts has
  // recorded a real measurement, the model must agree with it; if the kernel
  // measured SLOWER than the law it claims to accelerate, the claim is simply
  // false and must not be shipped quietly.
  const measured = (entry.speed as { measured?: { measuredSpeedup: number; agreement: number; resolvable: boolean } } | undefined)
    ?.measured;
  // Even an UNRESOLVABLE measurement carries information: it means both kernels
  // sit at the call-overhead floor, so the speedup is ~1.0 whatever the model
  // predicts. gelu kept advertising ×1.92 while measuring ×0.99 because the
  // substitution only fires on resolvable rows. Publishing a multiplier the
  // hardware could not reproduce is exactly the class of claim this audit
  // exists to stop.
  if (measured && !measured.resolvable) {
    const claimed = entry.speed?.speedup;
    if (claimed !== undefined && claimed > 1.25 && measured.measuredSpeedup < 1.1) {
      push(
        "speedup-unmeasurable",
        "warn",
        `advertises ×${claimed.toFixed(2)} but both kernels sit at the call-overhead floor (measured ×${measured.measuredSpeedup.toFixed(2)}) — the gain is not observable`,
      );
    }
  }
  if (measured?.resolvable) {
    const claimed = entry.speed?.speedup;
    if (measured.measuredSpeedup < 1 && claimed !== undefined && claimed > 1) {
      push(
        "speedup-not-real",
        "fail",
        `advertises ×${claimed.toFixed(2)} but MEASURES ×${measured.measuredSpeedup.toFixed(2)} — slower than the law it replaces`,
      );
    } else if (measured.agreement > 1.5 || measured.agreement < 0.667) {
      push(
        "speedup-model-off",
        "warn",
        // `claimed` is already the measured value once substitution happened,
        // which printed "modelled ×0.84 vs measured ×0.84". Report the model
        // figure from speedupModelled so the gap is actually visible.
        `modelled ×${((entry.speed as { speedupModelled?: number } | undefined)?.speedupModelled ?? claimed ?? 0).toFixed(2)} vs measured ×${measured.measuredSpeedup.toFixed(2)} (agreement ${measured.agreement.toFixed(2)}) — cite the measured figure`,
      );
    }
  }

  // ---- generalisation: holdout
  if (def.holdout) {
    const h = safe(() => def.holdout!(node));
    if (h) {
      row.holdout = h.metric;
      const train = row.recomputedMetric ?? entry.metric;
      if (typeof train === "number" && train > 0 && Number.isFinite(h.metric)) {
        const ratio = h.metric / Math.max(train, ABS_FLOOR);
        row.overfitRatio = ratio;
        if (!Number.isFinite(h.metric)) push("holdout-blowup", "fail", "holdout metric is not finite");
        else if (ratio > 100) push("overfit", "fail", `holdout is ${ratio.toExponential(1)}× worse than train — memorised the grid`);
        else if (ratio > 10) push("overfit", "warn", `holdout ${ratio.toFixed(1)}× worse than train`);
      }
    }
  } else {
    push("no-holdout", "warn", "task exposes no holdout — the record is a train-set number only");
  }

  // ---- out-of-distribution behaviour
  if (def.ood) {
    const v = safe(() => def.ood!(node));
    if (v !== undefined) {
      row.ood = v;
      if (!Number.isFinite(v)) push("ood-blowup", "fail", "formula diverges off-distribution (ood = ∞)");
      else if (v > 0) push("ood-violation", "warn", `ood violation ${v.toExponential(2)} — extrapolates badly`);
    }
  }

  // ---- deployability grade
  if (def.r2) {
    const r = safe(() => def.r2!(node));
    if (r !== undefined) {
      row.r2 = r;
      if (Number.isFinite(r) && r < 0.98) push("low-r2", "warn", `r²=${r.toFixed(4)} < 0.98 — below deployable grade`);
    }
  }

  // ---- fast slot coherence
  if (entry.fastTree) {
    const fnode = safe(() => parseNode(entry.fastTree));
    if (!fnode) push("fast-bad-ast", "fail", "fast slot AST unparseable");
    else {
      const fcost = estimateCost(fnode);
      row.fastCost = fcost;
      if (fcost >= cost) push("fast-not-faster", "fail", `fast slot costs ${fcost} >= champion ${cost} — it is not a fast slot`);
      const fev = safe(() => def.evaluate(fnode));
      if (fev) {
        row.fastMetric = fev.metric;
        if (typeof entry.fast?.metric === "number" && relDiff(entry.fast.metric, fev.metric) > 1e-2) {
          push("fast-metric-drift", "fail", `fast stored ${entry.fast.metric.toExponential(3)} vs recomputed ${fev.metric.toExponential(3)}`);
        }
        const flvl = levelOf(def, fev.metric);
        if (typeof entry.fast?.level === "number" && entry.fast.level > Math.max(2, flvl) && !entry.fast.deploy) {
          push("fast-level-inflated", "warn", `fast claims L${entry.fast.level}, milestones grant L${flvl}`);
        }
        if (def.r2) {
          const fr2 = safe(() => def.r2!(fnode));
          if (fr2 !== undefined && Number.isFinite(fr2) && fr2 < 0.98 && entry.fast?.deploy) {
            push("fast-deploy-lowr2", "fail", `fast slot flagged deploy:true but r²=${fr2.toFixed(4)} < 0.98`);
          }
        }
      }
    }
  }

  // ---- reproducibility metadata
  if (entry.seed === undefined) push("no-seed", "warn", "no seed recorded — the record is not replayable");

  return row;
}

function fmt(v: number | undefined, digits = 2): string {
  if (v === undefined || !Number.isFinite(v)) return v === undefined ? "—" : "∞";
  if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5)) return v.toExponential(digits);
  return v.toFixed(digits);
}

async function main() {
  const args = process.argv.slice(2);
  const asJson = args.includes("--json");
  const asMd = args.includes("--md");
  const strict = args.includes("--strict");

  const ledger = loadLedger();
  const defs = new Map(buildTasks().map((t) => [t.id, t]));
  const rows: Row[] = [];
  for (const [id, entry] of Object.entries(ledger).sort(([a], [b]) => a.localeCompare(b))) {
    rows.push(auditEntry(id, entry as Record<string, any>, defs.get(id)));
  }
  // tasks with no champion at all
  const missing = [...defs.keys()].filter((id) => !ledger[id]).sort();

  const fails = rows.filter((r) => r.issues.some((i) => i.severity === "fail"));
  const warns = rows.filter((r) => r.issues.some((i) => i.severity === "warn") && !fails.includes(r));
  // `info` issues (provenance) are annotations, not defects: a record whose
  // only issue is "this is the seeded law" is still a clean record.
  const clean = rows.filter((r) => r.issues.every((i) => i.severity === "info"));
  const recovered = rows.filter((r) => r.provenance === "recovered-seed");

  if (asJson) {
    console.log(JSON.stringify({ rows, missing, summary: { total: rows.length, clean: clean.length, warn: warns.length, fail: fails.length, recoveredSeed: recovered.length, searchFound: rows.length - recovered.length } }, null, 2));
    if (strict && fails.length) process.exit(1);
    return;
  }

  const out: string[] = [];
  const line = (s = "") => out.push(s);

  if (asMd) {
    line("# Champion audit — recomputed from the stored ASTs");
    line();
    line(`Generated by \`npx tsx scripts/audit-champions.ts --md\`. Nothing here is read from the ledger's own numbers: every metric, level, cost and speedup below is recomputed from the serialized AST against the live task definition.`);
    line();
    line(`| entries | clean | warnings | failures | tasks without champion |`);
    line(`|---|---|---|---|---|`);
    line(`| ${rows.length} | ${clean.length} | ${warns.length} | ${fails.length} | ${missing.length} |`);
    line();
    line("## Per-champion");
    line();
    line("| task | metric (recomputed) | drift | L | cost | ⚡ | holdout | overfit | ood | r² | flags |");
    line("|---|---|---|---|---|---|---|---|---|---|---|");
    for (const r of rows) {
      const flags = r.issues.map((i) => (i.severity === "fail" ? `**${i.code}**` : i.code)).join(", ") || "—";
      line(`| \`${r.id}\` | ${fmt(r.recomputedMetric, 3)} | ${r.metricDrift === undefined ? "—" : fmt(r.metricDrift, 1)} | ${r.recomputedLevel ?? "—"} | ${r.cost ?? "—"} | ${r.speedup ? "×" + r.speedup.toFixed(2) : "—"} | ${fmt(r.holdout, 3)} | ${r.overfitRatio ? r.overfitRatio.toFixed(1) + "×" : "—"} | ${fmt(r.ood, 1)} | ${r.r2 === undefined ? "—" : r.r2.toFixed(4)} | ${flags} |`);
    }
    line();
    line("## Findings in detail");
    line();
    for (const r of rows) {
      if (!r.issues.length) continue;
      line(`### \`${r.id}\``);
      for (const i of r.issues) line(`- **${i.severity.toUpperCase()}** \`${i.code}\` — ${i.detail}`);
      line();
    }
    if (missing.length) {
      line("## Tasks with no champion in the ledger");
      line();
      line(missing.map((m) => `\`${m}\``).join(", "));
      line();
    }
    console.log(out.join("\n"));
    if (strict && fails.length) process.exit(1);
    return;
  }

  // ---- terminal report
  const pad = (s: string, n: number) => s.padEnd(n);
  console.log(`\n▶ champion audit — ${rows.length} ledger entries, ${defs.size} live tasks\n`);
  console.log(pad("task", 24) + pad("metric", 12) + pad("drift", 10) + pad("L", 4) + pad("cost", 6) + pad("holdout", 12) + pad("ood", 8) + "flags");
  console.log("-".repeat(110));
  for (const r of rows) {
    const worst = r.issues.some((i) => i.severity === "fail") ? "✗" : r.issues.length ? "!" : "✓";
    const flags = r.issues.map((i) => i.code).join(",") || "";
    console.log(`${worst} ` + pad(r.id, 22) + pad(fmt(r.recomputedMetric, 2), 12) + pad(r.metricDrift === undefined ? "—" : fmt(r.metricDrift, 1), 10) +
      pad(String(r.recomputedLevel ?? "—"), 4) + pad(String(r.cost ?? "—"), 6) + pad(fmt(r.holdout, 2), 12) + pad(fmt(r.ood, 1), 8) + flags);
  }
  console.log("\n== detail ==");
  for (const r of rows) {
    if (!r.issues.length) continue;
    console.log(`\n[${r.id}]`);
    for (const i of r.issues) console.log(`  ${i.severity === "fail" ? "✗" : "!"} ${i.code}: ${i.detail}`);
  }
  console.log(`\n== summary ==`);
  console.log(`  clean     : ${clean.length}`);
  console.log(`  warnings  : ${warns.length}`);
  console.log(`  failures  : ${fails.length}`);
  console.log(`  no champion: ${missing.length}${missing.length ? " → " + missing.join(", ") : ""}`);
  console.log(`  provenance : ${rows.length - recovered.length} search-found | ${recovered.length} recovered seeded law`);
  const byCode = new Map<string, number>();
  for (const r of rows) for (const i of r.issues) byCode.set(i.code, (byCode.get(i.code) ?? 0) + 1);
  console.log(`\n== issue histogram ==`);
  for (const [code, n] of [...byCode].sort((a, b) => b[1] - a[1])) console.log(`  ${String(n).padStart(4)}  ${code}`);

  if (strict && fails.length) process.exit(1);
}

main().catch((e) => { console.error(e); process.exit(1); });
