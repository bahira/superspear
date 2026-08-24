// Composition & sweep audit — generalizes verify-inverse-pairs.ts to the whole ledger.
// a) inverse-pair composition [sigmoid, logit_ml] both directions on a dense grid
// b) finite-sweep + monotonic-where-expected check for every exact-grade (metric < 1e-6) single-var tree
// c) markdown report -> stdout + REPORTS/composition-audit.md
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { parseNode, evaluateScalar } from "../src/lib/spear/engine";

type Entry = {
  taskId?: string;
  metric?: number;
  formula?: string;
  tree?: unknown;
};

const EXACT_GRADE = 1e-6;
const N = 400;

// Sweep domain per exact-grade kernel. [0,1] for probability/weight tasks, else [-10,10].
// Monotone expectation only where mathematically certain; null = no global expectation.
const SWEEP_SPEC: Record<string, { lo: number; hi: number; monotone: "inc" | "dec" | null }> = {
  sigmoid: { lo: -10, hi: 10, monotone: "inc" },
  atan_unit: { lo: -10, hi: 10, monotone: "inc" },
  lambert_w: { lo: -10, hi: 10, monotone: "inc" }, // checked on [0,10] only, see below
  gaussian_kernel: { lo: -10, hi: 10, monotone: "dec" }, // even fn: checked on [0,10]
  gauss_shader: { lo: -10, hi: 10, monotone: "dec" },
  bilinear_interp: { lo: 0, hi: 1, monotone: "dec" },
  ema_smooth: { lo: -10, hi: 10, monotone: null }, // clamped rational fit, no global monotonicity claim
};

function sweepRange(id: string): [number, number, number, number] {
  const s = SWEEP_SPEC[id];
  if (!s) return [-10, 10, -10, 10];
  // For even/clamped kernels with a half-line expectation, test monotonicity only on the valid half.
  if (id === "lambert_w") return [s.lo, s.hi, 0, s.hi];
  if (id === "gaussian_kernel" || id === "gauss_shader") return [s.lo, s.hi, 0, s.hi];
  return [s.lo, s.hi, s.lo, s.hi];
}

function varsOf(node: unknown): Set<string> {
  const out = new Set<string>();
  (function walk(n: unknown) {
    if (n && typeof n === "object") {
      const o = n as Record<string, unknown>;
      if (o.o === "var" && typeof o.n === "string") out.add(o.n);
      for (const c of Object.values(o)) walk(c);
    }
  })(node);
  return out;
}

function main() {
  const root = join(import.meta.dirname ?? ".", "..");
  const led = JSON.parse(readFileSync(join(root, "spear-hall-of-fame.json"), "utf8")) as Record<string, Entry>;
  const lines: string[] = [];
  const say = (l: string) => { lines.push(l); console.log(l); };

  say(`# Composition Audit — ${new Date().toISOString()}`);
  say("");

  // ---- a) known inverse pair sigmoid <-> logit_ml, both directions ----
  const sig = parseNode(led["sigmoid"].tree as never);
  const lg = parseNode(led["logit_ml"].tree as never);
  let worst1 = 0, worst2 = 0; // worst1: sigmoid∘logit over prob domain; worst2: logit∘sigmoid over logit domain
  for (let i = 0; i <= N; i++) {
    const p = 0.02 + (0.96 * i) / N;
    worst1 = Math.max(worst1, Math.abs(evaluateScalar(sig, { x: evaluateScalar(lg, { x: p }) }) - p));
    const y = -6 + (12 * i) / N;
    worst2 = Math.max(worst2, Math.abs(evaluateScalar(lg, { x: evaluateScalar(sig, { x: y }) }) - y));
  }
  say("## Inverse pair [sigmoid, logit_ml]");
  say("");
  say("| composition | domain | max error |");
  say("|---|---|---|");
  say(`| sigmoid∘logit_ml | x ∈ [0.02, 0.98] | ${worst1.toExponential(3)} |`);
  say(`| logit_ml∘sigmoid | y ∈ [-6, 6] | ${worst2.toExponential(3)} |`);
  say("");
  say(worst1 < 0.02 && worst2 < 0.05 ? "**PAIRE INVERSE VALIDÉE** — les deux formes sont correctes ensemble" : "**PAIRE DÉGRADÉE** — au moins une dérive");
  say("");

  // ---- b) exact-grade single-var kernels: finite sweep + monotonicity ----
  say(`## Exact-grade sweep (metric < ${EXACT_GRADE}, single-var trees, ${N} points)`);
  say("");
  say("| task | metric | sweep | output range | finite | monotone (expected dir) | anomalies |");
  say("|---|---|---|---|---|---|---|");

  const findings: string[] = [];
  const ids = Object.keys(led);
  let audited = 0;
  for (const id of ids) {
    const e = led[id];
    const metric = e.metric ?? Infinity;
    if (!(metric < EXACT_GRADE)) continue;
    if (!e.tree) {
      say(`| ${id} | ${metric} | — | — | — | — | skipped: no top-level tree |`);
      findings.push(`**${id}**: metric=${metric} but no \`tree\` field in ledger (fastTree only) — excluded from sweep.`);
      continue;
    }
    const vars = varsOf(e.tree);
    if (vars.size !== 1) {
      say(`| ${id} | ${metric} | — | — | — | — | n/a: multi-var (${[...vars].join(", ")}) |`);
      findings.push(`**${id}**: exact-grade but multi-var (${[...vars].join(", ")}) — single-axis sweep not applicable.`);
      continue;
    }
    const v = [...vars][0];
    const node = parseNode(e.tree as never);
    const [lo, hi, mLo, mHi] = sweepRange(id);
    const spec = SWEEP_SPEC[id];
    let nonFinite = 0, monoViolations = 0, fMin = Infinity, fMax = -Infinity, prev = NaN;
    for (let i = 0; i <= N; i++) {
      const x = lo + ((hi - lo) * i) / N;
      const inWin = spec?.monotone ? x >= mLo && x <= mHi : false;
      const f = evaluateScalar(node, { [v]: x });
      if (!Number.isFinite(f)) { nonFinite++; prev = NaN; continue; }
      fMin = Math.min(fMin, f); fMax = Math.max(fMax, f);
      if (inWin && Number.isFinite(prev)) {
        const step = f - prev;
        if ((spec.monotone === "inc" && step < -1e-9) || (spec.monotone === "dec" && step > 1e-9)) monoViolations++;
      }
      prev = inWin ? f : NaN;
    }
    audited++;
    const okFinite = nonFinite === 0;
    const monoCell = !spec?.monotone ? "n/a" : monoViolations === 0 ? `OK (${spec.monotone})` : `FAIL (${monoViolations})`;
    const anom = nonFinite + monoViolations;
    say(`| ${id} | ${metric.toExponential(3)} | [${lo}, ${hi}] | [${fMin.toExponential(3)}, ${fMax.toExponential(3)}] | ${okFinite ? "OK" : `${nonFinite} NaN/Inf`} | ${monoCell} | ${anom} |`);

    // Out-of-domain / extrapolation warnings: clamped or rational forms fitted on a narrow band.
    const fml = e.formula ?? "";
    if (/\b(min|max|relu)\(/.test(fml)) {
      findings.push(`**${id}**: formula contains clamp constructs (${(fml.match(/\b(min|max|relu)\(/g) ?? []).map(s => s.slice(0, -1)).join(", ")}) — piecewise fit; extrapolation beyond training range unreliable (sweep outputs reached [${fMin.toExponential(3)}, ${fMax.toExponential(3)}]).`);
    }
    if (!okFinite || monoViolations > 0) {
      findings.push(`**${id}**: ANOMALY — non-finite=${nonFinite}, monotonicity violations=${monoViolations}.`);
    }
  }
  say("");
  say(`Audited ${audited} exact-grade single-var kernels out of ${ids.length} ledger entries.`);
  say("");

  // ---- c) findings ----
  say("## Findings");
  say("");
  if (!(worst1 < 0.02 && worst2 < 0.05)) {
    say(`- **[sigmoid, logit_ml]**: PAIRE DÉGRADÉE — logit_ml est une approximation (metric=${led["logit_ml"].metric?.toExponential(3)}); l'erreur de composition explose aux extrêmes de probabilité (logit∘sigmoid max ${worst2.toExponential(3)} sur [-6,6]). Restreindre l'usage au domaine d'entraînement ou re-miner logit_ml.`);
  }
  for (const f of findings.length ? findings : ["No issues found."]) say(`- ${f}`);
  say("");

  const reportPath = join(root, "REPORTS", "composition-audit.md");
  if (!existsSync(join(root, "REPORTS"))) mkdirSync(join(root, "REPORTS"));
  writeFileSync(reportPath, lines.join("\n"), "utf8");
  console.log(`\nReport written to REPORTS/composition-audit.md (${lines.length} lines).`);
}

main();
