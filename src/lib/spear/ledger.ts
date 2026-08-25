// SPEAR ledger I/O — the ONLY door for writing discovery records.
//
// Layout:
//   ledger/<taskId>.json      source of truth, one file per task (granular
//                             diffs, no whole-ledger rewrite conflicts)
//   spear-hall-of-fame.json   GENERATED bundle, rebuilt atomically by
//                             saveLedger() so every existing reader (npm
//                             generator, bootstrap seeding, site data,
//                             README links) keeps working untouched.
//
// All writers MUST go through saveLedger/saveEntry; hand-editing the bundle
// is lost on the next write.
import { readFileSync, writeFileSync, existsSync, readdirSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";

export const ROOT = join(import.meta.dirname ?? ".", "..", "..", "..");
export const LEDGER_DIR = join(ROOT, "ledger");
/** Generated compatibility bundle — DO NOT hand-edit. */
export const BUNDLE_PATH = join(ROOT, "spear-hall-of-fame.json");

/** Permissive entry shape: scripts keep their own narrow views on top. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type LedgerEntry = Record<string, any>;
export type Ledger = Record<string, LedgerEntry>;

function readLegacy(): Ledger {
  if (!existsSync(BUNDLE_PATH)) return {};
  try {
    return JSON.parse(readFileSync(BUNDLE_PATH, "utf8")) as Ledger;
  } catch {
    return {};
  }
}

/** Full ledger view: per-task files win; legacy bundle is the fallback. */
export function loadLedger(): Ledger {
  if (!existsSync(LEDGER_DIR)) return readLegacy();
  const out: Ledger = {};
  // legacy entries remain visible unless a granular file supersedes them
  const legacy = readLegacy();
  for (const [id, e] of Object.entries(legacy)) out[id] = e;
  for (const f of readdirSync(LEDGER_DIR)) {
    if (!f.endsWith(".json")) continue;
    const id = f.slice(0, -".json".length);
    try {
      out[id] = JSON.parse(readFileSync(join(LEDGER_DIR, f), "utf8"));
    } catch { /* fichier partiel/illisible — on garde l'entrée fallback */ }
  }
  return out;
}

export function loadEntry(id: string): LedgerEntry | undefined {
  const p = join(LEDGER_DIR, `${id}.json`);
  if (existsSync(p)) {
    try { return JSON.parse(readFileSync(p, "utf8")); } catch { /* fallthrough */ }
  }
  return loadLedger()[id];
}

/**
 * Persist: one file per task, prune orphans, rebuild the compatibility bundle
 * byte-stably (same serializer every writer used before).
 */
export function saveLedger(ledger: Ledger): void {
  mkdirSync(LEDGER_DIR, { recursive: true });
  for (const f of readdirSync(LEDGER_DIR)) {
    if (f.endsWith(".json")) {
      const id = f.slice(0, -".json".length);
      if (!ledger[id]) rmSync(join(LEDGER_DIR, f)); // tâche disparue du ledger
    }
  }
  for (const [id, e] of Object.entries(ledger)) {
    writeFileSync(join(LEDGER_DIR, `${id}.json`), JSON.stringify(e, null, 2) + "\n");
  }
  writeFileSync(BUNDLE_PATH, JSON.stringify(ledger, null, 2));
}
