// Repair mojibake (UTF-8 bytes previously decoded as Windows-1252) in ledger
// text fields. Some historical writes corrupted titles/formulas ("â€”", "Ã‚Â²").
// Decoder: map every char back to its CP1252 byte, re-interpret as UTF-8,
// accept only when the result is valid and measurably cleaner. Iterates up to
// 3 times for doubly-corrupted strings.
//
// Usage: npx tsx scripts/fix-ledger-encoding.ts [--dry]
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// CP1252 high-byte table — keyed by UNICODE value → byte (reverse lookup).
// Keys must be the rendered glyphs' codepoints (€ = U+20AC → byte 0x80, …):
// object-literal numeric keys are decimal, so the direction matters.
const HI: Record<number, number> = {
  0x20ac: 0x80, 0x201a: 0x82, 0x0192: 0x83, 0x201e: 0x84, 0x2026: 0x85,
  0x2020: 0x86, 0x2021: 0x87, 0x02c6: 0x88, 0x2030: 0x89, 0x0160: 0x8a,
  0x2039: 0x8b, 0x0152: 0x8c, 0x017d: 0x8e, 0x2018: 0x91, 0x2019: 0x92,
  0x201c: 0x93, 0x201d: 0x94, 0x2022: 0x95, 0x2013: 0x96, 0x2014: 0x97,
  0x02dc: 0x98, 0x2122: 0x99, 0x0161: 0x9a, 0x203a: 0x9b, 0x0153: 0x9c,
  0x017e: 0x9e, 0x0178: 0x9f,
};
const UNDEFINED_CP1252 = new Set([0x81, 0x8d, 0x8f, 0x90, 0x9d]);

/** One inverse step: chars → CP1252 bytes → UTF-8 string. null if impossible. */
function decodeOnce(s: string): string | null {
  const bytes: number[] = [];
  for (const ch of s) {
    const cp = ch.codePointAt(0)!;
    const hi = HI[cp];
    // chars above 0xFF are only survivable through the CP1252 high table
    if (hi === undefined && cp >= 0x100) return null;
    if (UNDEFINED_CP1252.has(cp)) return null;
    bytes.push(hi !== undefined ? hi : cp);
  }
  const out = Buffer.from(bytes).toString("utf8");
  return out.includes("\uFFFD") ? null : out;           // invalid UTF-8 — reject
}

/** How mojibake-ish a string looks: lower is better. */
function garbageScore(s: string): number {
  return (s.match(/[\u00c3\u00c2\u00e2\u00e5\u0152\u017d]/g)?.length ?? 0);
}

function repair(s: string): { text: string; changed: boolean } {
  let cur = s;
  let curScore = garbageScore(s);
  for (let i = 0; i < 3; i++) {
    const cand = decodeOnce(cur);
    if (cand === null) break;
    const candScore = garbageScore(cand);
    // accept strictly-improving decodes; equal scores keep the shorter (bytes
    // collapse when multi-byte UTF-8 folds into single glyphs)
    if (candScore > curScore || (candScore === curScore && cand.length >= cur.length)) break;
    if (!/[\u0080-\uFFFF]/.test(cand) && !/[\u0080-\uFFFF]/.test(cur)) break;
    cur = cand;
    curScore = candScore;
  }
  return { text: cur, changed: cur !== s };
}

async function main() {
  const dry = process.argv.includes("--dry");
  const path = join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json");
  type Finding = { taskId: string; title?: string; formula?: string; fast?: { formula?: string } | null };
  const ledger = JSON.parse(readFileSync(path, "utf8")) as Record<string, Finding>;

  let fixed = 0;
  const samples: string[] = [];
  for (const e of Object.values(ledger)) {
    for (const key of ["title", "formula"] as const) {
      const r = repair(e[key] ?? "");
      if (r.changed && e[key]) {
        samples.push(`[${e.taskId}] ${key}: ${JSON.stringify(e[key])} -> ${JSON.stringify(r.text)}`);
        e[key] = r.text;
        fixed++;
      }
    }
    if (e.fast?.formula) {
      const r = repair(e.fast.formula);
      if (r.changed) {
        samples.push(`[${e.taskId}] fast: ${JSON.stringify(e.fast.formula)} -> ${JSON.stringify(r.text)}`);
        e.fast.formula = r.text;
        fixed++;
      }
    }
  }

  // titles have a clean upstream source — the live registry. Sync over whatever
  // survived decoding; this erases even historically FFFD-baked titles.
  const { buildTasks } = await import("../src/lib/spear/benchmarks");
  const defs = new Map(buildTasks().map((t) => [t.id, t]));
  let synced = 0;
  for (const [id, e] of Object.entries(ledger)) {
    const clean = defs.get(id)?.title;
    if (clean && e.title !== clean) {
      samples.push(`[${id}] title synced -> ${JSON.stringify(clean)}`);
      e.title = clean;
      synced++;
    }
  }

  console.log(`champs réparés: ${fixed} · titres synchronisés: ${synced}`);
  for (const s of samples.slice(0, 12)) console.log("  " + s);

  // final polish: reprint every display string from its authoritative AST —
  // heals characters no decoder can recover (e.g. U+FFFD-baked superscripts)
  const { parseNode, nodeToString } = await import("../src/lib/spear/engine");
  let reprinted = 0;
  for (const e of Object.values(ledger)) {
    const t = e as Finding & { tree?: unknown; fastTree?: unknown };
    if (t.tree) {
      const s = nodeToString(parseNode(t.tree as never));
      if (e.formula !== s) { e.formula = s; reprinted++; }
    }
    if (t.fastTree && e.fast) {
      const s = nodeToString(parseNode(t.fastTree as never));
      if (e.fast.formula !== s) { e.fast.formula = s; reprinted++; }
    }
  }
  console.log(`formules réimprimées depuis les arbres: ${reprinted}`);

  // honest residue report: what NO decoder could recover
  const bad = /[\u00c3][\u0080-\u00bf]|\uFFFD|â€/;
  const residue: string[] = [];
  for (const e of Object.values(ledger)) {
    for (const v of [e.title, e.formula, e.fast?.formula]) {
      if (v && bad.test(v)) { residue.push(`[${e.taskId}] ${JSON.stringify(v.slice(0, 70))}`); break; }
    }
  }
  console.log(residue.length ? `résidus suspects (${residue.length}):` : "aucun résidu suspect.");
  for (const r of residue.slice(0, 10)) console.log("  " + r);
  if (!dry) {
    writeFileSync(path, JSON.stringify(ledger, null, 2));
    console.log("ledger écrit.");
  }
}

main().catch((err) => { console.error(err); process.exit(1); });
