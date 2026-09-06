// Ledger repair — rewrite every record so its stored numbers are the numbers
// you get by re-evaluating its stored AST. Nothing is invented and nothing is
// improved: this pass only makes the ledger tell the truth about what it holds.
//
// What it fixes, and why the damage existed:
//
//  1. UNSHAPED FAST SLOTS. The loop paired a raw AST with the metric of its
//     affine-wrapped ("shaped") form, so the ledger stored e.g. `atan(x)`
//     labelled MSE 3.6e-4 while `atan(x)` actually scores 8.9e-1. The shaped
//     formula `0.351631·atan(x) + 0.5` is the thing that earns 3.6e-4. Repair:
//     recover the shaped node via evaluateScored and store THAT.
//  2. FAST SLOTS THAT AREN'T FAST. Once re-shaped (or once the champion
//     improved) some "fast" variants cost >= the champion. A fast slot that is
//     not cheaper has no reason to exist: dropped.
//  3. STALE METRICS / LEVELS / COSTS. Re-derived from the AST.
//
// Usage:
//   npx tsx scripts/repair-ledger.ts --dry    # report only
//   npx tsx scripts/repair-ledger.ts          # write
import { loadLedger, saveLedger } from "../src/lib/spear/ledger";
import { buildTasks } from "../src/lib/spear/benchmarks";
import { parseNode, serializeNode, estimateCost, nodeToString, canonicalKey } from "../src/lib/spear/engine";
import type { SpearNode } from "../src/lib/spear/engine";
import type { TaskDef } from "../src/lib/spear/tasks/types";

const dry = process.argv.includes("--dry");

function levelOf(t: TaskDef, metric: number): number {
  let lvl = 0;
  for (const m of t.milestones) if (Number.isFinite(metric) && m.test(metric)) lvl = Math.max(lvl, m.level);
  return lvl;
}

/** The self-contained node whose true metric is the reported one. */
function shape(t: TaskDef, node: SpearNode): { node: SpearNode; metric: number } {
  if (t.evaluateScored) {
    const s = t.evaluateScored(node);
    if (Number.isFinite(s.metric)) return { node: s.node, metric: s.metric };
  }
  return { node, metric: t.evaluate(node).metric };
}

const changes: string[] = [];
const ledger = loadLedger();
const defs = new Map(buildTasks().map((t) => [t.id, t]));

for (const [id, entry] of Object.entries(ledger) as [string, any][]) {
  const t = defs.get(id);
  if (!t || !entry.tree) continue;

  // ---------- champion ----------
  let champNode: SpearNode;
  try { champNode = parseNode(entry.tree); } catch { continue; }
  const direct = t.evaluate(champNode).metric;
  const shaped = shape(t, champNode);
  // Only re-shape the champion if shaping is what makes the stored metric true;
  // a champion that already reproduces its number is left byte-identical.
  const storedM = entry.metric as number;
  const closeTo = (v: number) => Number.isFinite(v) && Number.isFinite(storedM) &&
    Math.abs(v - storedM) <= Math.max(1e-12, Math.abs(storedM) * 1e-3);

  if (!closeTo(direct)) {
    if (closeTo(shaped.metric) && canonicalKey(shaped.node) !== canonicalKey(champNode)) {
      entry.tree = serializeNode(shaped.node);
      entry.formula = nodeToString(shaped.node);
      changes.push(`[${id}] champion re-shaped to its self-contained form (metric ${storedM.toExponential(3)} now reproduces)`);
      champNode = shaped.node;
    } else {
      // the stored number simply is not what the AST scores — trust the AST
      const truth = t.evaluate(champNode).metric;
      changes.push(`[${id}] champion metric corrected ${storedM.toExponential(3)} -> ${truth.toExponential(3)}`);
      entry.metric = truth;
    }
  }
  // formula text must always render the AST
  const txt = nodeToString(champNode);
  if (entry.formula !== txt) { entry.formula = txt; }
  const trueMetric = t.evaluate(champNode).metric;
  entry.metric = trueMetric;
  const lvl = levelOf(t, trueMetric);
  if (entry.level !== lvl) {
    changes.push(`[${id}] level ${entry.level} -> ${lvl}`);
    entry.level = lvl;
  }
  // Costs must be priced under ONE profile. Farm children ran with the GPU
  // profile (transcendental = 1) and wrote formulaCost under it, while
  // exactCost came from the default CPU profile (transcendental = 20) — so a
  // formula full of exp/log could advertise a speedup it never had (tanh_sat:
  // `tanh(x)` claiming ×20 against the exact law `tanh(x)`). Everything is
  // re-priced here on the default profile and the profile is recorded.
  const cost = estimateCost(champNode);
  const exactCost = t.exactCost ?? (t.exactRefNode ? estimateCost(t.exactRefNode) : undefined);
  if (entry.speed) {
    if (entry.speed.formulaCost !== cost) changes.push(`[${id}] cost ${entry.speed.formulaCost} -> ${cost} (re-priced on the default profile)`);
    entry.speed.formulaCost = cost;
    entry.speed.costProfile = "cpu-transcendental-20";
    if (exactCost) {
      const su = exactCost / Math.max(1, cost);
      if (entry.speed.speedup !== undefined && Math.abs(entry.speed.speedup - su) / Math.max(su, 1e-9) > 0.02) {
        changes.push(`[${id}] speedup ×${Number(entry.speed.speedup).toFixed(2)} -> ×${su.toFixed(2)}`);
      }
      entry.speed.exactCost = exactCost;
      entry.speed.speedup = su;
    }
  }

  // ---------- fast slot ----------
  if (entry.fastTree) {
    let fnode: SpearNode | null = null;
    try { fnode = parseNode(entry.fastTree); } catch { fnode = null; }
    if (!fnode) {
      delete entry.fast; delete entry.fastTree;
      changes.push(`[${id}] fast slot dropped (unparseable AST)`);
    } else {
      const fdirect = t.evaluate(fnode).metric;
      const fstored = entry.fast?.metric as number | undefined;
      const fshaped = shape(t, fnode);
      const fclose = (v: number) => fstored !== undefined && Number.isFinite(v) &&
        Math.abs(v - fstored) <= Math.max(1e-12, Math.abs(fstored) * 1e-3);

      if (!fclose(fdirect) && fclose(fshaped.metric)) {
        changes.push(`[${id}] fast slot re-shaped: '${nodeToString(fnode)}' -> '${nodeToString(fshaped.node)}' (was claiming ${fstored!.toExponential(3)} while scoring ${fdirect.toExponential(3)})`);
        fnode = fshaped.node;
      }
      const fmetric = t.evaluate(fnode).metric;
      const fcost = estimateCost(fnode);
      if (!Number.isFinite(fmetric) || fcost >= cost) {
        delete entry.fast; delete entry.fastTree;
        changes.push(`[${id}] fast slot dropped (cost ${fcost} >= champion ${cost}: not a fast slot)`);
      } else {
        const flvl = levelOf(t, fmetric);
        const deploy = flvl >= 2 ? undefined : (t.r2 && t.r2(fnode) >= 0.98 ? true : undefined);
        if (flvl < 2 && !deploy) {
          delete entry.fast; delete entry.fastTree;
          changes.push(`[${id}] fast slot dropped (L${flvl} and r² below deployable grade — not validated)`);
        } else {
          entry.fastTree = serializeNode(fnode);
          entry.fast = {
            ...(entry.fast ?? {}),
            formula: nodeToString(fnode),
            metric: fmetric,
            level: Math.max(2, flvl),
            formulaCost: fcost,
            ...(deploy ? { deploy: true } : {}),
          };
          if (!deploy) delete entry.fast.deploy;
        }
      }
    }
  }
}

if (dry) {
  console.log(changes.join("\n"));
  console.log(`\n${changes.length} changes (dry run, nothing written)`);
} else {
  saveLedger(ledger);
  console.log(changes.join("\n"));
  console.log(`\n${changes.length} changes written`);
}
