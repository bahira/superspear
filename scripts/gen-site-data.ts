// Generate docs/kernels.js — the compact dataset behind the project site's
// interactive registry. Node-native UTF-8: replaces an earlier PowerShell
// extraction whose ANSI reads mojibaked every non-ASCII title/formula.
//
// Usage: npx tsx scripts/gen-site-data.ts
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

interface Speed { formulaCost?: number; exactCost?: number; speedup?: number; vsIterative?: { label: string; speedup: number } }
interface FastEntry { formula: string; metric: number; level?: number; formulaCost?: number; vsIterative?: { label: string; speedup: number } }
interface Finding {
  taskId: string;
  title: string;
  metric: number;
  level: number;
  formula?: string;
  speed?: Speed;
  fast?: FastEntry;
}
type Ledger = Record<string, Finding>;

const root = join(import.meta.dirname ?? ".", "..");
const ledger = JSON.parse(readFileSync(join(root, "spear-hall-of-fame.json"), "utf8")) as Ledger;

const r1 = (v: number): number => Math.round(v * 10) / 10;

const rows = Object.entries(ledger)
  .sort(([a], [b]) => a.localeCompare(b))
  .map(([id, v]) => {
    const o: Record<string, unknown> = { i: id, t: v.title, m: v.metric, l: v.level };
    if (v.formula !== undefined) o.f = v.formula;
    if (v.speed) {
      if (v.speed.formulaCost !== undefined && v.speed.formulaCost !== null) o.fc = v.speed.formulaCost;
      if (v.speed.exactCost !== undefined && v.speed.exactCost !== null) o.xc = v.speed.exactCost;
      if (v.speed.speedup !== undefined && v.speed.speedup !== null) o.sp = r1(v.speed.speedup);
      if (v.speed.vsIterative) o.vi = { l: v.speed.vsIterative.label, s: r1(v.speed.vsIterative.speedup) };
    }
    if (v.fast) {
      o.ff = v.fast.formula;
      o.fm = v.fast.metric;
      if (v.fast.formulaCost !== undefined && v.fast.formulaCost !== null) o.fs = v.fast.formulaCost;
      if (v.fast.vsIterative) o.fvi = { l: v.fast.vsIterative.label, s: r1(v.fast.vsIterative.speedup) };
    }
    return o;
  });

const js = `window.SPEAR_KERNELS=${JSON.stringify(rows)};\n`;
const out = join(root, "docs", "kernel-data.js");
writeFileSync(out, js, "utf8");
console.log(`kernel-data.js: ${rows.length} kernels`);
