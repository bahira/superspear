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
  roundConst,
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

// ------------------------------------------------ sister fast-slot donation
// A fast slot learned on a one-variable task is reusable by any sister task
// whose subtree has the SAME shape (variables renamed). silu owns a 3-unit
// relu slot; swiglu_fused = u·silu(g) inherits it for 5 units instead of the
// 23 a local Padé on exp costs. Same knowledge, no new search, no external
// baseline: the donor is our own validated ledger entry.

/** Canonical key with variables renamed to v0, v1… in first-encounter order. */
function shapeKey(n: SpearNode, names = new Map<string, string>()): string {
  if (n.op === "var") {
    let slot = names.get(n.name);
    if (slot === undefined) { slot = `v${names.size}`; names.set(n.name, slot); }
    return `v:${slot}`;
  }
  if (n.op === "const") return `c:${roundConst(n.value)}`;
  return `${n.op}(${n.children.map((c) => shapeKey(c, names)).join(",")})`;
}

function distinctVars(n: SpearNode, out = new Set<string>()): Set<string> {
  if (n.op === "var") out.add(n.name);
  n.children.forEach((c) => distinctVars(c, out));
  return out;
}

function renameVar(n: SpearNode, from: string, to: string): SpearNode {
  return makeNode(n.op, {
    value: n.value,
    name: n.op === "var" && n.name === from ? to : n.name,
    children: n.children.map((c) => renameVar(c, from, to)),
  });
}

/** Replaces every subtree whose shape key is in `donors`; null if none matched. */
function donate(n: SpearNode, donors: Map<string, SpearNode>): SpearNode | null {
  let hits = 0;
  const walk = (node: SpearNode): SpearNode => {
    const rep = donors.get(shapeKey(node));
    if (rep && !hasTranscendental(rep)) {
      hits++;
      const targetVars = [...distinctVars(node)];
      const donorVars = [...distinctVars(rep)];
      if (targetVars.length !== 1 || donorVars.length !== 1) return rep;
      return renameVar(rep, donorVars[0], targetVars[0]);
    }
    return makeNode(node.op, { value: node.value, name: node.name, children: node.children.map(walk) });
  };
  const out = walk(n);
  return hits > 0 ? out : null;
}

/** shapeKey -> cheapest validated one-variable fast slot we already own. */
function harvestDonors(led: Record<string, { tree?: unknown; fastTree?: unknown }>): Map<string, SpearNode> {
  const donors = new Map<string, SpearNode>();
  for (const e of Object.values(led)) {
    if (!e.tree || !e.fastTree) continue;
    try {
      const champ = parseNode(e.tree as never);
      const fast = parseNode(e.fastTree as never);
      if (distinctVars(champ).size !== 1 || distinctVars(fast).size !== 1) continue;
      if (hasTranscendental(fast)) continue;
      if (estimateCost(fast) >= estimateCost(champ)) continue;
      const key = shapeKey(champ);
      const prev = donors.get(key);
      if (!prev || estimateCost(fast) < estimateCost(prev)) donors.set(key, fast);
    } catch { /* AST illisible : on l'ignore */ }
  }
  return donors;
}

async function main() {
  const led = loadLedger();
  const defs = new Map((buildTasks() as unknown as { id: string; variables: string[]; evaluate: (n: SpearNode) => { metric: number }; r2?: (n: SpearNode) => number | undefined; milestones: { level: number; test: (m: number) => boolean }[] }[]).map((t) => [t.id, t]));
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

  // ---- pass 2: un slot mieux qu'un slot existant, via donation entre sœurs
  const donors = harvestDonors(led);
  let upgraded = 0;
  for (const [id, entry] of Object.entries(led)) {
    if (!entry.tree) continue;
    const t = defs.get(id);
    if (!t) continue;
    const champ = parseNode(entry.tree);
    if (!hasTranscendental(champ)) continue;
    const candidate = donate(champ, donors);
    if (!candidate) continue;
    // un candidat qui parle une variable inconnue ne peut que tomber à 1e9 :
    // on le dit au lieu de le laisser dies dans le silence de la métrique
    const known = new Set(t.variables as unknown as string[]);
    const stray = [...distinctVars(candidate)].filter((v) => !known.has(v));
    if (stray.length) { gaps.push(`${id} (donation : variable inconnue « ${stray.join(", ")} »)`); continue; }
    const cost = estimateCost(candidate);
    const current = entry.fastTree ? parseNode(entry.fastTree) : null;
    const currentCost = current ? estimateCost(current) : estimateCost(champ);
    if (cost >= currentCost) continue; // aucun gain réel
    const score = scoreOf(t);
    const metric = score(candidate);
    const lvl = levelOf(t, metric);
    const r2 = t.r2?.(candidate);
    if (!(lvl >= 2 || (r2 !== undefined && Number.isFinite(r2) && r2 >= 0.98))) continue;
    entry.fast = { formula: nodeToString(candidate), metric, level: Math.max(2, lvl), formulaCost: cost };
    if (r2 !== undefined && Number.isFinite(r2) && r2 >= 0.98) entry.fast.deploy = true;
    entry.fastTree = serializeNode(candidate);
    upgraded++;
    console.log(`↑ [${id}] fast ${currentCost} -> ${cost} u (donation) | metric ${metric.toExponential(3)} | L${entry.fast.level}${entry.fast.deploy ? " deploy" : ""}`);
  }
  if (upgraded) {
    saveLedger(led);
    console.log(`\nfast slots améliorés par donation: ${upgraded}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
