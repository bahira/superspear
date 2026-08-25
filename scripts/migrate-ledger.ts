// One-shot migration: split the legacy monolithic spear-hall-of-fame.json
// into ledger/<taskId>.json files, then rebuild the bundle through saveLedger
// (byte-stable output). Idempotent: re-running on a migrated repo is a no-op.
//
// Usage: npx tsx scripts/migrate-ledger.ts
import { existsSync } from "node:fs";
import { join } from "node:path";
import { LEDGER_DIR, loadLedger, saveLedger } from "../src/lib/spear/ledger";

if (existsSync(LEDGER_DIR)) {
  console.log("ledger/ existe déjà — migration déjà faite, rien à faire.");
} else {
  const ledger = loadLedger();
  saveLedger(ledger);
  console.log(`migré: ${Object.keys(ledger).length} entrées -> ledger/`);
}
