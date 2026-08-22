// One-off: remove corrupted ledger keys (titles used as ids by an old bug).
import { readFileSync, writeFileSync } from "node:fs";
const p = "spear-hall-of-fame.json";
const ledger = JSON.parse(readFileSync(p, "utf8")) as Record<string, unknown>;
let n = 0;
for (const k of Object.keys(ledger)) {
  // real task ids are [a-z0-9_] only — anything else is a stale shell
  if (!/^[a-z0-9_]+$/.test(k)) {
    delete ledger[k];
    n++;
    console.log("removed:", JSON.stringify(k));
  }
}
writeFileSync(p, JSON.stringify(ledger, null, 2));
console.log("done. entries:", Object.keys(ledger).length, "| removed:", n);
