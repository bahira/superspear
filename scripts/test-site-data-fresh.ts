// The public landing page must not show stale numbers.
//
// docs/kernel-data.js is a GENERATED snapshot of the ledger that powers the
// GitHub Pages site. Nothing ever re-ran its generator, so it drifted: 46 of
// the 89 kernels on the public page disagreed with the ledger — the shop
// window was advertising metrics the project had already superseded.
//
// This is the worst class of bug in a project whose whole premise is "every
// number reproduces", because it is the only number most visitors ever see.
//
// Usage: npx tsx scripts/test-site-data-fresh.ts
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { loadLedger } from "../src/lib/spear/ledger";

function main(): void {
  const root = process.cwd();
  const raw = readFileSync(join(root, "docs", "kernel-data.js"), "utf8");
  const m = raw.match(/window\.SPEAR_KERNELS\s*=\s*(\[[\s\S]*\]);?\s*$/);
  if (!m) {
    console.error("✗ docs/kernel-data.js is not in the expected `window.SPEAR_KERNELS = [...]` form");
    process.exit(1);
  }
  const site = JSON.parse(m[1]) as { i: string; m: number; l: number }[];
  const ledger = loadLedger() as Record<string, { metric?: number; level?: number } | undefined>;

  const stale: string[] = [];
  const missing: string[] = [];

  for (const row of site) {
    const e = ledger[row.i];
    if (!e) { missing.push(`${row.i} (on the site but not in the ledger)`); continue; }
    const lm = e.metric;
    if (typeof lm === "number" && Number.isFinite(lm)) {
      // relative comparison: these are metrics spanning 1e-35 .. 1e2
      const drift = Math.abs(lm - row.m) / Math.max(Math.abs(lm), 1e-300);
      if (drift > 1e-6) {
        stale.push(`${row.i}: site ${row.m.toExponential(3)} vs ledger ${lm.toExponential(3)}`);
      }
    }
    if (typeof e.level === "number" && e.level !== row.l) {
      stale.push(`${row.i}: site level L${row.l} vs ledger L${e.level}`);
    }
  }

  for (const id of Object.keys(ledger)) {
    if (!site.some((r) => r.i === id)) missing.push(`${id} (in the ledger but absent from the site)`);
  }

  if (stale.length || missing.length) {
    console.error("✗ the public site data is out of sync with the ledger\n");
    for (const s of stale.slice(0, 20)) console.error("  stale   " + s);
    if (stale.length > 20) console.error(`  … and ${stale.length - 20} more`);
    for (const s of missing.slice(0, 10)) console.error("  missing " + s);
    console.error(`\n${stale.length} stale, ${missing.length} missing — run: npx tsx scripts/gen-site-data.ts`);
    process.exit(1);
  }

  console.log(`test-site-data-fresh: ${site.length} kernels on the public site match the ledger`);
}

main();
