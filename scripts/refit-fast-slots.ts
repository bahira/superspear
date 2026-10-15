// Refit des fast slots contre la mÃ©trique de LEUR tÃ¢che.
//
// Deux voies produisent un fast slot :
//   - fill-fast-slots.ts (approximant algÃ©brique) : les constantes sont dÃ©jÃ 
//     tunÃ©es contre la mÃ©trique de la tÃ¢che (`refineConstants(node, score)`);
//   - la voie Â« donation Â» (transplanter le fast slot d'une AUTRE tÃ¢che,
//     ex. swiglu_fused <- silu) : les constantes viennent de la tÃ¢che DONNEUSE
//     et ne sont pas re-tunÃ©es â€” c'est lÃ  qu'un gain dort.
// Constat mesurÃ© : refit de swiglu_fused 9.011e-3 -> 7.628e-3 (-15 %) Ã  coÃ»t
// Ã©gal, sans toucher Ã  la structure.
//
// refineConstants ne fait que 6 tours et n'accepte QU'UN pas amÃ©liorant par
// constante et par tour : une invocation unique n'atteint jamais l'optimum
// local (mesurÃ© : passes successives 13 -> 8 -> 7 slots encore amÃ©liorables).
// On boucle donc par slot jusqu'Ã  convergence (< 0.01 % de gain), puis on
// n'Ã©crit que si le gain total dÃ©passe 0.5 % et que le niveau de milestone
// reste >= 2 (coÃ»t inchangÃ©, structure inchangÃ©e).
// Run: npx tsx scripts/refit-fast-slots.ts
import { loadLedger, saveLedger } from "../src/lib/spear/ledger";
import {
  parseNode,
  estimateCost,
  refineConstants,
  serializeNode,
  nodeToString,
  type SpearNode,
} from "../src/lib/spear/engine";
import { buildTasks } from "../src/lib/spear/benchmarks";

type Def = {
  id: string;
  evaluate: (n: SpearNode) => { metric: number };
  milestones: { level: number; test: (m: number) => boolean }[];
};

const led = loadLedger();
const defs = new Map((buildTasks() as unknown as Def[]).map((t) => [t.id, t]));
const MARGIN = 0.005; // < 0.5 % = bruit de coordonnÃ©e, on ne touche pas
const CONVERGED = 5e-5; // gain < 0.01 % entre deux appels = point fixe
const MAX_CALLS = 60; // garde-fou ; le crawl mesurÃ© converge en ~5-10

let improved = 0;
for (const [id, e] of Object.entries(led)) {
  if (!e.fastTree || !e.fast || !e.tree) continue;
  const t = defs.get(id);
  if (!t) continue;
  const score = (n: SpearNode): number => {
    try {
      const m = t.evaluate(n).metric;
      return Number.isFinite(m) ? m : 1e9;
    } catch {
      return 1e9;
    }
  };
  const levelOf = (m: number): number => {
    let l = 0;
    for (const ms of t.milestones) if (Number.isFinite(m) && ms.test(m)) l = Math.max(l, ms.level);
    return l;
  };

  const cur = parseNode(e.fastTree as never);
  const m0 = score(cur);
  if (!Number.isFinite(m0) || m0 <= 0) continue; // exact ou cassÃ© : rien Ã  gagner

  // descente jusqu'au point fixe de la structure (constantes seules)
  let best = parseNode(e.fastTree as never);
  let bestM = m0;
  let calls = 0;
  while (calls < MAX_CALLS) {
    const r = refineConstants(best, score, 600);
    calls++;
    const m = score(r.node);
    if (Number.isFinite(m) && m < bestM * (1 - CONVERGED)) {
      best = r.node;
      bestM = m;
    } else {
      break;
    }
  }
  if (!(bestM < m0 * (1 - MARGIN))) continue;
  const cost = estimateCost(best);
  if (cost >= estimateCost(parseNode(e.tree as never))) continue; // sÃ©curitÃ© : toujours strictement moins cher
  const lvl = levelOf(bestM);
  if (lvl < 2) continue;
  e.fast = { ...e.fast, formula: nodeToString(best), metric: bestM, level: Math.max(2, lvl), formulaCost: cost };
  e.fastTree = serializeNode(best);
  improved++;
  console.log(`âœ“ [${id}] ${m0.toExponential(3)} -> ${bestM.toExponential(3)} (${calls} appels) | coÃ»t ${cost} | L${e.fast.level} | ${nodeToString(best)}`);
}

if (improved > 0) saveLedger(led);
console.log(`\nfast slots amÃ©liorÃ©s: ${improved} â€” ledger ${improved > 0 ? "Ã©crit" : "inchangÃ©"}`);
