// Primitive-landing audit. Pattern proven twice this cycle: when a NodeOp
// lands in the engine, every benchmark task designed BEFORE it may carry an
// obsolete premise ("hard mode", "unservable") that hides cheap breakthroughs
// — grover_amplitude fell exactly this way once asin got served.
//
// What it does:
//   1. diffs served primitives (engine ALL_OPS) against the last-audited
//      snapshot (.spear-primitives.json, committed);
//   2. scans benchmarks.ts for hard-mode markers whose claimed-missing
//      primitive is NOW served → STALE (actionable);
//   3. lists open walls (ledger L0/L1) as re-seed candidates whenever any
//      primitive is new.
//
// Usage:
//   npx tsx scripts/audit-scaffolds.ts            # audit report
//   npx tsx scripts/audit-scaffolds.ts --update   # snapshot current ops
//   npx tsx scripts/audit-scaffolds.ts --strict   # exit 1 on stale markers (CI)
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { ALL_OPS } from "../src/lib/spear/engine";
import { buildTasks } from "../src/lib/spear/benchmarks";

interface Snapshot { ops: string[]; updatedAt: string }

const root = join(import.meta.dirname ?? ".", "..");
const snapPath = join(root, ".spear-primitives.json");
const benchPath = join(root, "src", "lib", "spear", "benchmarks.ts");

const MARKER = /unservable|hard mode|not served|missing\s+(inverse[ -]?trig\s+)?primitive/i;
// ops a marker may claim missing, by name as they'd appear in prose
const OP_WORDS: Record<string, RegExp> = {
  asin: /\basin\b/i, atan: /\batan\b/i, exp: /\bexp\b/i, log: /\blog\b/i,
  sin: /\bsin\b/i, cos: /\bcos\b/i, sqrt: /\bsqrt\b/i,
};

function main() {
  const dry = process.argv.includes("--dry");
  const strict = process.argv.includes("--strict");
  const update = process.argv.includes("--update");

  const current: string[] = [...ALL_OPS].sort();
  const snap: Snapshot = existsSync(snapPath)
    ? JSON.parse(readFileSync(snapPath, "utf8"))
    : { ops: [], updatedAt: "(none)" };

  if (update) {
    writeFileSync(snapPath, JSON.stringify({ ops: current, updatedAt: new Date().toISOString() }, null, 2) + "\n");
    console.log(`snapshot mis à jour: ${current.length} primitives (${current.join(", ")})`);
    return;
  }

  const prev = new Set(snap.ops);
  const added = current.filter((o) => !prev.has(o));
  const removed = [...prev].filter((o) => !current.includes(o));

  console.log(`primitives servies: ${current.length} (snapshot ${snap.updatedAt})`);
  console.log(added.length ? `  NOUVELLES depuis dernier audit: ${added.join(", ")}` : "  aucune nouvelle primitive depuis le dernier audit.");
  if (removed.length) console.log(`  RETIRÉES: ${removed.join(", ")}`);

  // ---- stale-marker scan over the task registry source
  const lines = readFileSync(benchPath, "utf8").split(/\r?\n/);
  const stale: { line: number; text: string }[] = [];
  lines.forEach((line, idx) => {
    if (!MARKER.test(line)) return;
    const window = lines.slice(Math.max(0, idx - 2), idx + 3).join("\n");
    for (const [op, wordRe] of Object.entries(OP_WORDS)) {
      if (!wordRe.test(window)) continue;
      if (current.includes(op)) {
        stale.push({ line: idx + 1, text: line.trim().slice(0, 90) });
        return;
      }
    }
  });
  console.log(stale.length
    ? `\nMARQUEurs PÉRIMÉS (${stale.length}) — primitive prétendue manquante mais servie :`
    : "\nmarqueurs hard-mode: tous cohérents avec les primitives servies.");
  for (const s of stale) console.log(`  benchmarks.ts:${s.line}  ${s.text}`);

  // ---- open walls become re-seed candidates whenever anything new landed
  if (added.length > 0 || process.argv.includes("--walls")) {
    try {
      const led = JSON.parse(readFileSync(join(root, "spear-hall-of-fame.json"), "utf8")) as Record<string, { level?: number }>;
      const defs = buildTasks();
      const walls = defs.filter((d) => (led[d.id]?.level ?? 0) <= 1 && d.id !== "bilinear_interp");
      console.log(`\nmurs ouverts (L≤1), candidats à un re-scaffold: ${walls.map((w) => w.id).join(", ")}`);
    } catch { /* pas de ledger — cold start */ }
  }

  if (added.length > 0) {
    console.log("\n→ action: relire chaque tâche touchée par les nouvelles primitives, re-seeder si la premise est tombée, puis --update.");
  }
  if (strict && stale.length > 0) process.exit(1);
}

main();
