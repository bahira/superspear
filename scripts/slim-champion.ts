import { loadLedger, saveLedger } from "../src/lib/spear/ledger";
// Champion compression: iteratively prune + refine a ledger champion to shed
// cost while holding near-exact accuracy. Targets scaffold-exact solutions
// that resist plain slimming (eigen 120u, logit 32u...).
//
// Usage: npx tsx scripts/slim-champion.ts <taskId> [--max-mse-rel 1e-6]
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseNode, simplify, prune, refineConstants, nodeToString, estimateCost } from "../src/lib/spear/engine";
import { buildTasks } from "../src/lib/spear/benchmarks";

async function main() {
  const id = process.argv[2] ?? "eigen3_sym";
  const maxRelArgIdx = process.argv.indexOf("--max-mse-rel");
  const maxRel = maxRelArgIdx !== -1 ? Number(process.argv[maxRelArgIdx + 1]) : 1e-6;

  const ledgerPath = join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json");
  const led = loadLedger();
  const entry = led[id];
  if (!entry?.tree) { console.log(`no tree for ${id}`); return; }

  const def = buildTasks().find((t) => t.id === id);
  if (!def) { console.log(`no task ${id}`); return; }

  const scoreOf = (n: any) => {
    try { const m = def.evaluate(n).metric; return Number.isFinite(m) ? m : 1e9; }
    catch { return 1e9; }
  };

  let node = parseNode(entry.tree);
  let cost0 = estimateCost(node);
  let mse0 = scoreOf(node);

  console.log(`[${id}] départ: cost=${cost0} mse=${mse0.toExponential(4)}`);

  for (let round = 1; round <= 12; round++) {
    const before = estimateCost(node);
    // prune subtrees that don't hurt much (tolerance tight)
    const pruned = prune(node, scoreOf, 1.0005);
    node = pruned.node;
    node = simplify(node);
    const tuned = refineConstants(node, scoreOf, 120);
    node = tuned.node;
    const after = estimateCost(node);
    const mse = scoreOf(node);
    console.log(`round ${round}: cost ${before} -> ${after} | mse ${mse.toExponential(4)}`);
    if (after === before && mse > maxRel) break;
    if (mse / Math.max(mse0, 1e-300) > Math.max(maxRel, 1e-300) * 50 && round >= 3) {
      // drifted too far from the original quality — stop
      break;
    }
  }

  const finalCost = estimateCost(node);
  const finalMse = scoreOf(node);
  const relDeg = finalMse / Math.max(mse0, 1e-300);
  console.log(`\nfinal: cost ${cost0} -> ${finalCost} | mse x${relDeg.toExponential(2)} vs départ`);

  if (finalCost < cost0 && relDeg <= 50) {
    entry.formula = nodeToString(node);
    entry.tree = JSON.parse(JSON.stringify(node));
    entry.metric = finalMse;
    if (entry.speed?.formulaCost) entry.speed.formulaCost = finalCost;
    saveLedger(led);
    console.log(`✓ ledger mis à jour (${id}: ${cost0} -> ${finalCost} unités)`);
  } else {
    console.log("aucun gain net — ledger inchangé");
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
