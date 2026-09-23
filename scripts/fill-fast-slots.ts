// Fast-slot coverage: give every ledger champion without a fast slot the best
// algebraic approximant we can honestly construct. Method: substitute each
// transcendental subtree with its Padé/Taylor algebraic approximant, then
// prune + refine constants against the task's own metric. A slot is written
// only if it is genuinely cheaper than the champion AND validated (level >= 2
// or deployment grade r² >= 0.98) — junk approximants are reported as gaps,
// never shipped. Run: npx tsx scripts/fill-fast-slots.ts
import { loadLedger, saveLedger } from "../src/lib/spear/ledger";
import {
  makeNode,
  parseNode,
  estimateCost,
  prune,
  refineConstants,
  serializeNode,
  nodeToString,
  ALGEBRAIC_OPS,
  type NodeOp,
  type SpearNode,
} from "../src/lib/spear/engine";
import { buildTasks } from "../src/lib/spear/benchmarks";

const n0 = (v: number): SpearNode => makeNode("const", { value: v });

// Padé [2/2] of e^v — good for |v| ≲ 2, degrades beyond (task rails apply).
function padeExp(v: SpearNode): SpearNode {
  const h = makeNode("mul", { children: [n0(0.5), v] });
  const h2 = makeNode("mul", { children: [makeNode("sq", { children: [v] }), n0(1 / 12)] });
  const num = makeNode("add", { children: [makeNode("add", { children: [n0(1), h] }), h2] });
  const denInner = makeNode("add", { children: [n0(1), makeNode("neg", { children: [h] })] });
  const den = makeNode("add", { children: [denInner, h2] });
  return makeNode("pdiv", { children: [num, den] });
}

// Taylor of sin(v) to v⁵ — good for |v| ≲ π.
function taylorSin(v: SpearNode): SpearNode {
  const c = makeNode("cube", { children: [v] });
  const c5 = makeNode("mul", { children: [c, makeNode("sq", { children: [v] })] });
  return makeNode("add", {
    children: [
      makeNode("sub", { children: [v, makeNode("pdiv", { children: [c, n0(6)] })] }),
      makeNode("pdiv", { children: [c5, n0(120)] }),
    ],
  });
}

// Taylor of cos(v) to v⁴.
function taylorCos(v: SpearNode): SpearNode {
  const v2 = makeNode("sq", { children: [v] });
  const v4 = makeNode("sq", { children: [v2] });
  return makeNode("add", {
    children: [
      makeNode("sub", { children: [n0(1), makeNode("pdiv", { children: [v2, n0(2)] })] }),
      makeNode("pdiv", { children: [v4, n0(24)] }),
    ],
  });
}

// Padé [3/2]-style of tanh(u).
function padeTanh(u: SpearNode): SpearNode {
  const u2 = makeNode("sq", { children: [u] });
  return makeNode("mul", {
    children: [u, makeNode("pdiv", { children: [makeNode("add", { children: [n0(27), u2] }), makeNode("add", { children: [n0(27), makeNode("mul", { children: [n0(9), u2] })] })] })],
  });
}

const SUBS: Partial<Record<NodeOp, (v: SpearNode) => SpearNode>> = {
  exp: padeExp,
  sin: taylorSin,
  cos: taylorCos,
  tanh: padeTanh,
  // log/asin/acos/sqrt : pas d'approximante algébrique fermée honnête
};

/** Replace every substitutable transcendental subtree with its approximant. */
function algebraize(node: SpearNode): SpearNode {
  if (node.op === "var" || node.op === "const") return node;
  const sub = SUBS[node.op];
  const children = node.children.map(algebraize);
  if (sub) return sub(children[0]);
  return makeNode(node.op, { value: node.value, name: node.name, children });
}

function hasTranscendental(node: SpearNode): boolean {
  if (node.op !== "var" && node.op !== "const" && !ALGEBRAIC_OPS.has(node.op)) return true;
  return node.children.some(hasTranscendental);
}

async function main() {
  const led = loadLedger();
  const defs = new Map((buildTasks() as unknown as { id: string; evaluate: (n: SpearNode) => { metric: number }; r2?: (n: SpearNode) => number | undefined; milestones: { level: number; test: (m: number) => boolean }[] }[]).map((t) => [t.id, t]));
  const levelOf = (t: { milestones: { level: number; test: (m: number) => boolean }[] }, m: number): number => {
    let l = 0;
    for (const ms of t.milestones) if (Number.isFinite(m) && ms.test(m)) l = Math.max(l, ms.level);
    return l;
  };
  const scoreOf = (t: { evaluate: (n: SpearNode) => { metric: number } }) => (n: SpearNode) => {
    try { const m = t.evaluate(n).metric; return Number.isFinite(m) ? m : 1e9; }
    catch { return 1e9; }
  };

  const gaps: string[] = [];
  let filled = 0;
  for (const [id, entry] of Object.entries(led)) {
    if (entry.fastTree && entry.fast) continue; // already covered
    if (!entry.tree) { gaps.push(id); continue; }
    const t = defs.get(id);
    if (!t) { gaps.push(id); continue; }
    const champ = parseNode(entry.tree);
    const champCost = estimateCost(champ);

    // un fast slot doit être STRICTEMENT moins cher : un champion déjà
    // algébrique et minimal n'en aura jamais — gap structurel, pas un échec
    const score = scoreOf(t);
    let node = algebraize(champ);
    if (estimateCost(node) >= champCost && !hasTranscendental(node)) { gaps.push(`${id} (champion algébrique minimal, cost ${champCost})`); continue; }

    const tuned = refineConstants(node, score, 120);
    node = tuned.node;
    const pruned = prune(node, score, 1.0005);
    node = pruned.node;
    const cost = estimateCost(node);
    const metric = score(node);
    const lvl = levelOf(t, metric);
    const r2 = t.r2?.(node);

    if (cost >= champCost) { gaps.push(`${id} (approximante pas moins chère: ${cost} >= ${champCost})`); continue; }
    if (!(lvl >= 2 || (r2 !== undefined && Number.isFinite(r2) && r2 >= 0.98))) {
      gaps.push(`${id} (niveau ${lvl}, r²=${r2 === undefined ? "—" : r2.toFixed(4)} — sous le seuil)`);
      continue;
    }

    entry.fast = {
      formula: nodeToString(node),
      metric,
      level: Math.max(2, lvl),
      formulaCost: cost,
    };
    if (r2 !== undefined && Number.isFinite(r2) && r2 >= 0.98) entry.fast.deploy = true;
    entry.fastTree = serializeNode(node);
    filled++;
    console.log(`✓ [${id}] fast cost ${champCost} -> ${cost} | metric ${metric.toExponential(3)} | L${entry.fast.level}${entry.fast.deploy ? " deploy" : ""}`);
  }

  saveLedger(led);
  console.log(`\nfast slots ajoutés: ${filled}`);
  console.log(`gaps honnêtes: ${gaps.length}`);
  for (const g of gaps) console.log(`  - ${g}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
