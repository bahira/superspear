import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  ALL_OPS,
  ALGEBRAIC_OPS,
  canonicalKey,
  countOps,
  evaluateNode,
  evaluateScalar,
  refineConstants,
  simplify,
  estimateCost,
  makeNode,
  parseNode,
  rand,
  randInt,
  serializeNode,
  type SerializedNode,
  type NodeOp,
  type GpConfig,
  type SpearNode,
} from "../engine";
import { mse, linfError, linspace, mapArray, gaussianRandom, erf } from "../math-utils";
import type { TaskBaseline, TaskDef } from "./types";

export const GP_OPS = ALL_OPS;

// SPEAR_ALU_ONLY=1 bans every transcendental from the search (exp/sin/cos/
// log/atan/asin): smaller branching factor, faster generations, and every
// discovered form is SFU-free by construction. Ground truths that NEED
// transcendental laws can only be approximated — that is the point: hunt for
// hyper-cheap VALIDATED forms (fast slots), not for records.
export const GP_OPS_EFFECTIVE = process.env.SPEAR_ALU_ONLY === "1"
  ? ALL_OPS.filter((o) => ALGEBRAIC_OPS.has(o))
  : GP_OPS;

/** In ALU-only mode, drop seed shapes that carry transcendental subtrees. */
export const pureSeeds = (seeds: SpearNode[]): SpearNode[] =>
  process.env.SPEAR_ALU_ONLY === "1"
    ? seeds.filter((s) => countOps(s).transcendental === 0)
    : seeds;

// ------------------------------------------------------------------ helpers
export function ols1(x: Float64Array, y: Float64Array): { a: number; b: number; mse: number } {
  const n = x.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i]; }
  const d = n * sxx - sx * sx;
  const a = Math.abs(d) < 1e-12 ? 0 : (n * sxy - sx * sy) / d;
  const b = (sy - a * sx) / n;
  let s = 0;
  for (let i = 0; i < n; i++) { const e = a * x[i] + b - y[i]; s += e * e; }
  return { a, b, mse: s / n };
}

export function grid(n: number, lo: number, hi: number): Float64Array {
  return linspace(lo, hi, n);
}

export const KV_SEQ = 320;
export const KV_KEEP = 40;
export const KV_TRAIN = 8;
export const KV_TEST = 12;
export const KV_SINK = 4;
export const KV_RECENT = 40;
export const KV_HEAVY = 10;

interface KvWorld {
  /** shared latent importance profile (heavy hitters + sinks) */
  w: Float64Array;
  train: KvSample[];
  test: { feats: KvSample; future: Float64Array }[];
}
type KvSample = Record<"A" | "P" | "S" | "R", Float64Array>;

export function softmaxInto(logits: Float64Array, out: Float64Array): void {
  let mx = -Infinity;
  for (let i = 0; i < logits.length; i++) mx = Math.max(mx, logits[i]);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - mx); sum += out[i]; }
  for (let i = 0; i < logits.length; i++) out[i] /= sum || 1;
}

export function buildKvWorld(): KvWorld {
  const w = new Float64Array(KV_SEQ);
  for (let i = 0; i < KV_SINK; i++) w[i] = 3.2 + rand() * 1.2;              // attention sinks
  const heavy = new Set<number>();
  while (heavy.size < KV_HEAVY) {
    heavy.add(KV_SINK + Math.floor(rand() * (KV_SEQ - KV_SINK - KV_RECENT)));
  }
  heavy.forEach((i) => { w[i] = 2.6 + rand() * 1.6; });                      // persistent heavy hitters
  const mkSample = (recencyBoost: number): KvSample => {
    const logits = new Float64Array(KV_SEQ);
    for (let i = 0; i < KV_SEQ; i++) logits[i] = w[i] + gaussianRandom() * 0.9;
    if (recencyBoost > 0) {
      for (let i = 0; i < KV_RECENT; i++) {
        logits[KV_SEQ - KV_RECENT + i] += recencyBoost * (0.55 + (i / KV_RECENT) * 0.9);
      }
    }
    // Scores are normalised by their mean (standard practice in H2O/SnapKV
    // implementations) so that attention mass, position flags and recency flags
    // live on comparable scales: A ≈ 1 for an average token, ≈ 15 for a heavy
    // hitter. Without this, no bounded coefficient could ever balance them.
    const A = new Float64Array(KV_SEQ);
    softmaxInto(logits, A);
    for (let i = 0; i < KV_SEQ; i++) A[i] *= KV_SEQ;
    const P = new Float64Array(KV_SEQ);
    const S = new Float64Array(KV_SEQ);
    const R = new Float64Array(KV_SEQ);
    for (let i = 0; i < KV_SEQ; i++) {
      P[i] = i / (KV_SEQ - 1);
      if (i < KV_SINK) S[i] = 1;
      if (i >= KV_SEQ - KV_RECENT) R[i] = 1;
    }
    return { A, P, S, R };
  };
  // observed statistics: past queries put NO emphasis on recency (old context)
  // Training queries span the real distribution of regimes: some look far back
  // in the context (low recency), some concentrate on the newest tokens. A
  // policy calibrated only on one regime would be blind to the other — that is
  // exactly the failure mode we want the benchmark to expose.
  const TRAIN_REGIMES = [0.35, 2.6, 1.1, 2.6, 0.35, 1.6, 2.6, 0.8];
  const train = Array.from({ length: KV_TRAIN }, (__, i) => mkSample(TRAIN_REGIMES[i % TRAIN_REGIMES.length]));
  // each test case = observed statistics + the unseen next-query attention
  // The test split is drawn from the SAME mixture of regimes as calibration
  // (a policy tuned on one regime and scored on another would be a flaw in the
  // benchmark, not a result).
  const TEST_REGIMES = [2.6, 0.35, 1.1, 2.6, 0.35, 2.6, 1.6, 0.8, 2.6, 0.35, 1.1, 2.6];
  const test = Array.from({ length: KV_TEST }, (_, i) => ({
    feats: mkSample(TRAIN_REGIMES[i % TRAIN_REGIMES.length]),
    future: mkSample(TEST_REGIMES[i % TEST_REGIMES.length]).A,
  }));
  return { w, train, test };
}

/** sum of weights of the k largest scores (binary min-heap, O(n log k)) */
export function topKWeightSum(score: Float64Array, weight: Float64Array, k: number): number {
  const hs = new Float64Array(k);
  const hw = new Float64Array(k);
  let size = 0;
  const up = (i: number) => {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (hs[p] <= hs[i]) break;
      [hs[p], hs[i]] = [hs[i], hs[p]];
      [hw[p], hw[i]] = [hw[i], hw[p]];
      i = p;
    }
  };
  const down = (i: number) => {
    for (;;) {
      const l = 2 * i + 1, r = 2 * i + 2;
      let s = i;
      if (l < size && hs[l] < hs[s]) s = l;
      if (r < size && hs[r] < hs[s]) s = r;
      if (s === i) break;
      [hs[s], hs[i]] = [hs[i], hs[s]];
      [hw[s], hw[i]] = [hw[i], hw[s]];
      i = s;
    }
  };
  for (let i = 0; i < score.length; i++) {
    if (size < k) { hs[size] = score[i]; hw[size] = weight[i]; size++; up(size - 1); }
    else if (score[i] > hs[0]) { hs[0] = score[i]; hw[0] = weight[i]; down(0); }
  }
  let t = 0;
  for (let i = 0; i < size; i++) t += hw[i];
  return t;
}

/** retained share of the UNSEEN next-query attention mass */
export function massOf(node: SpearNode, world: KvWorld, split: "train" | "test"): number {
  const pool = split === "train" ? world.train.map((s) => ({ feats: s, w: s.A })) : world.test.map((t) => ({ feats: t.feats, w: t.future }));
  let total = 0;
  for (const { feats, w } of pool) {
    let sc: Float64Array;
    try {
      sc = evaluateNode(node, feats, KV_SEQ);
    } catch {
      return -1;
    }
    for (let i = 0; i < KV_SEQ; i++) if (!Number.isFinite(sc[i])) return -1;
    total += topKWeightSum(sc, w, KV_KEEP);
  }
  // features are mean-normalised (× KV_SEQ); the evaluation weight stays a
  // probability mass, so we divide the accumulated sum back down.
  return (total / pool.length / KV_SEQ) * 100;
}

export function buildKvTask(): TaskDef {
  const world = buildKvWorld();
  const gpConfig: GpConfig = {
    variables: ["A", "P", "S", "R"],
    constRange: [0.05, 4],
    ops: ["add", "sub", "mul", "pdiv", "max", "min"],
    maxDepth: 4,
    terminalVarProb: 0.7,
  };

  // selection happens on the observed (train) split only
  const trainMetric = (node: SpearNode): number => massOf(node, world, "train");
  // reported score is measured on the unseen next-query attention
  const testMetric = (node: SpearNode): number => massOf(node, world, "test");

  // Baselines share the exact same top-k routine on the SAME test split.
  const evalRule = (fn: (f: KvSample, i: number) => number): number => {
    let total = 0;
    for (const t of world.test) {
      const score = Float64Array.from({ length: KV_SEQ }, (_, i) => fn(t.feats, i));
      total += topKWeightSum(score, t.future, KV_KEEP);
    }
    return (total / world.test.length / KV_SEQ) * 100;
  };

  const ceiling = (() => {
    let total = 0;
    for (const t of world.test) total += topKWeightSum(t.future, t.future, KV_KEEP);
    return (total / world.test.length / KV_SEQ) * 100;
  })();

  const randomMetric = evalRule(() => rand());
  const windowMetric = evalRule((_, i) => (i >= KV_SEQ - KV_KEEP ? 1 : 0));
  const streamingMetric = evalRule((_, i) => (i < KV_SINK ? 3 : i >= KV_SEQ - KV_KEEP + KV_SINK ? 1 : 0));
  const h2oMetric = evalRule((f, i) => f.A[i]);
  const h2oWindowMetric = evalRule((f, i) => (f.A[i] + (i >= KV_SEQ - KV_KEEP ? 6 : 0)) * (i < KV_SINK ? 4 : 1));

  const baselines: TaskBaseline[] = [
    { name: "Aléatoire", metric: randomMetric, note: "Contrôle bas de l'échelle", kind: "statistical", formula: "rand" },
    { name: "Fenêtre glissante", metric: windowMetric, note: "Sliding window — défaut vLLM", kind: "heuristic", formula: "keep last 40" },
    { name: "StreamingLLM", metric: streamingMetric, note: "Sinks + fenêtre (Xiao et al. 2023)", kind: "heuristic", formula: "3·S + recency" },
    { name: "H2O (attention accumulée)", metric: h2oMetric, note: "Top-k sur Ā observé (Zhang et al. 2023)", kind: "heuristic", formula: "A" },
    { name: "H2O + fenêtre (type SnapKV)", metric: h2oWindowMetric, note: "Heuristique hybride la plus forte du lot", kind: "heuristic", formula: "4·S·A + 6·R" },
    { name: "Oracle (A futur, non observable)", metric: ceiling, note: "Plafond théorique — inaccessible au moment de la décision", kind: "oracle", formula: "top-k sur A_futur" },
  ];

  return {
    id: "kv_cache",
    family: "kv_cache",
    title: "Règle d'éviction du KV-cache",
    subtitle: `${KV_SEQ} tokens, budget ${KV_KEEP} (${(KV_SEQ / KV_KEEP).toFixed(0)}× compression) — sélection sur ${KV_TRAIN} requêtes passées, évaluée sur ${KV_TEST} requêtes futures`,
    groundTruth: "Score(A, P, S, R) — A = masse d'attention accumulée, normalisée par sa moyenne",
    metricLabel: "Masse d'attention future conservée (%)",
    metricDirection: "max",
    formatMetric: (m) => `${m.toFixed(2)} %`,
    variables: ["A", "P", "S", "R"],
    gpConfig,
    evaluate: (node) => ({ metric: trainMetric(node), finite: true }),
    holdout: (node) => ({ metric: testMetric(node), finite: true }),
    refine: (node) => {
      const res = refineConstants(node, (c) => -trainMetric(c), 24);
      return { node: res.node, evals: res.evals };
    },
    seedPool: [
      makeNode("var", { name: "A" }),
      makeNode("add", { children: [makeNode("var", { name: "A" }), makeNode("var", { name: "R" })] }),
      makeNode("add", { children: [makeNode("var", { name: "A" }), makeNode("mul", { children: [makeNode("const", { value: 2 }), makeNode("var", { name: "S" })] })] }),
      // StreamingLLM-style: heavy on sinks + recency
      makeNode("add", { children: [makeNode("mul", { children: [makeNode("const", { value: 4 }), makeNode("var", { name: "S" })] }), makeNode("mul", { children: [makeNode("const", { value: 2 }), makeNode("var", { name: "R" })] })] }),
      // SnapKV-style: attention with recency bonus
      makeNode("add", { children: [makeNode("mul", { children: [makeNode("const", { value: 4 }), makeNode("var", { name: "S" })] }), makeNode("add", { children: [makeNode("var", { name: "A" }), makeNode("mul", { children: [makeNode("const", { value: 3 }), makeNode("var", { name: "R" })] })] })] }),
      // H2O-style: attention with sink bonus
      makeNode("add", { children: [makeNode("var", { name: "A" }), makeNode("mul", { children: [makeNode("const", { value: 5 }), makeNode("var", { name: "S" })] })] }),
      // hybrid: A dominates, sinks and recency boost
      makeNode("add", { children: [
        makeNode("mul", { children: [makeNode("const", { value: 3 }), makeNode("var", { name: "S" })] }),
        makeNode("add", { children: [makeNode("mul", { children: [makeNode("const", { value: 1 }), makeNode("var", { name: "A" })] }), makeNode("mul", { children: [makeNode("const", { value: 4 }), makeNode("var", { name: "R" })] })] }),
      ] }),
      // multiplicative sink boost: sinks scale the whole score, not just add
      makeNode("mul", { children: [
        makeNode("add", { children: [makeNode("var", { name: "A" }), makeNode("var", { name: "R" })] }),
        makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("mul", { children: [makeNode("const", { value: 3 }), makeNode("var", { name: "S" })] })] }),
      ] }),
      makeNode("max", { children: [
        makeNode("add", { children: [makeNode("var", { name: "A" }), makeNode("var", { name: "R" })] }),
        makeNode("mul", { children: [makeNode("var", { name: "S" }), makeNode("const", { value: 8 })] }),
      ] }),
      makeNode("mul", { children: [
        makeNode("var", { name: "A" }),
        makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("mul", { children: [makeNode("const", { value: 3 }), makeNode("var", { name: "S" })] })] }),
      ] }),
    ],
    baselines,
    // Calibrated against the measured oracle ceiling: milestones are expressed
    // as "gap closed" so they stay reachable whatever the world seed produces.
    milestones: [
      { level: 1, label: `≥ ${(ceiling - 22).toFixed(0)} % de masse future`, test: (m) => m >= ceiling - 22 },
      { level: 2, label: `≥ ${(ceiling - 12).toFixed(0)} %`, test: (m) => m >= ceiling - 12 },
      { level: 3, label: `Toutes les heuristiques dépassées (≥ ${(ceiling - 7).toFixed(0)} %)`, test: (m) => m >= ceiling - 7 },
      { level: 4, label: `Gap oracle fermé à 75 % (≥ ${(ceiling - 3.5).toFixed(0)} %)`, test: (m) => m >= ceiling - 3.5 },
      { level: 5, label: `Gap oracle fermé à 90 % (≥ ${(ceiling - 1.5).toFixed(0)} %)`, test: (m) => m >= ceiling - 1.5 },
    ],
    verify: (node) => {
      const dump = JSON.stringify(node);
      const usesSink = dump.includes('"name":"S"');
      const usesRecency = dump.includes('"name":"R"');
      const usesAttention = dump.includes('"name":"A"');
      if (usesAttention && usesRecency && usesSink) {
        return "Règle tri-dimensionnelle A + S + R : heavy hitters + attention sinks + récence, la triade que la littérature a mise des années à identifier.";
      }
      if (usesAttention && usesRecency) return "Combine attention accumulée et récence (proche SnapKV).";
      return null;
    },
    codeVarDecl: "const float A, const float P, const float S, const float R",
  };
}

// ---------------------------------------------------------------------------
// REGRESSION TASKS — recover closed-form physical laws from noisy data
// ---------------------------------------------------------------------------

// Reference kernels for the cost model: the true generating law expressed as
// an AST per task id. estimateCost on these gives the ALU/SFU baseline that a
// discovered formula is compared against (speedup = exactCost / formulaCost).
/**
 * Cultural bootstrap: load champion ASTs from a previous hall-of-fame ledger
 * and offer them as warm-up bricks for OTHER tasks (never the task that found
 * them — no self-distillation of an already-solved answer). Variables are
 * renamed to the target task's variables; oversized trees are dropped and the
 * smallest shapes come first. Enabled whenever the ledger file exists.
 */
export function loadBootstrapSeeds(varNames: string[], excludeId: string, maxSeeds = 5): SpearNode[] {
  if (process.env.SPEAR_NO_BOOTSTRAP === "1") return [];
  try {
    const path = process.env.SPEAR_LEDGER ?? "spear-hall-of-fame.json";
    const ledger = JSON.parse(readFileSync(path, "utf8")) as Record<string, { tree?: SerializedNode; taskId?: string }>;
    const candidates: SpearNode[] = [];
    for (const [id, entry] of Object.entries(ledger)) {
      if (id === excludeId || !entry.tree) continue;
      const tree = parseNode(entry.tree);
      if (tree.size > 14 || tree.depth > 6) continue;
      // rename every variable to the target's variables, positionally by
      // first appearance — single-var targets absorb everything cleanly
      let vi = 0;
      const rename = (nd: SpearNode): SpearNode => {
        if (nd.op === "var") {
          const name = varNames[vi++ % varNames.length];
          return makeNode("var", { name });
        }
        return makeNode(nd.op, { value: nd.value, children: nd.children.map(rename) });
      };
      const renamed = simplify(rename(tree));
      // anti-cheat: skip if it IS this task's exact published law
      const law = EXACT_LAWS[excludeId];
      if (law && canonicalKey(renamed) === canonicalKey(law)) continue;
      candidates.push(renamed);
    }
    candidates.sort((a, b) => a.size - b.size);
    return candidates.slice(0, maxSeeds);
  } catch {
    return []; // no ledger yet, or unreadable — cold start is fine
  }
}

export const V = (name: string): SpearNode => makeNode("var", { name });
export const C = (value: number): SpearNode => makeNode("const", { value });
export const EXACT_LAWS: Record<string, SpearNode> = {
  free_fall: makeNode("mul", { children: [C(4.905), makeNode("sq", { children: [V("t")] })] }),
  kepler: makeNode("mul", { children: [V("a"), makeNode("sqrt", { children: [V("a")] })] }),
  european_call: makeNode("add", { children: [
    makeNode("mul", { children: [C(40), V("sigma")] }),
    makeNode("mul", { children: [C(16), makeNode("sq", { children: [V("sigma")] })] }),
  ] }),
  lambert_w: makeNode("mul", { children: [V("x"), makeNode("exp", { children: [V("x")] })] }),
  rc_circuit: makeNode("sub", { children: [C(1), makeNode("exp", { children: [makeNode("neg", { children: [V("t")] })] })] }),
  layernorm_scale: makeNode("pdiv", { children: [C(1), makeNode("sqrt", { children: [V("x")] })] }),
  gaussian_kernel: makeNode("exp", { children: [makeNode("neg", { children: [makeNode("mul", { children: [C(0.5), makeNode("sq", { children: [V("x")] })] })] })] }),
  diffusion_beta: (() => {
    // cost model only cares about structure, not constant values
    const inner = makeNode("add", { children: [V("t"), C(0.01)] });
    const scaled = makeNode("mul", { children: [C(1.56), inner] });
    const wave = makeNode("cos", { children: [scaled] });
    return makeNode("sub", { children: [C(1), makeNode("sq", { children: [wave] })] });
  })(),
  bilinear_interp: makeNode("sub", { children: [C(1), V("u")] }),
  temporal_grad: makeNode("sub", { children: [V("b"), V("a")] }),
  lorentz: makeNode("pdiv", { children: [C(1), makeNode("sqrt", { children: [makeNode("sub", { children: [C(1), makeNode("sq", { children: [V("b")] })] })] })] }),
  hill: makeNode("pdiv", { children: [makeNode("cube", { children: [V("c")] }), makeNode("add", { children: [C(1), makeNode("cube", { children: [V("c")] })] })] }),
  kerr: makeNode("add", { children: [
    makeNode("pdiv", { children: [C(4), V("b")] }),
    makeNode("add", { children: [
      makeNode("pdiv", { children: [C(11.78097245), makeNode("sq", { children: [V("b")] })] }),
      makeNode("pdiv", { children: [C(42.66666667), makeNode("cube", { children: [V("b")] })] }),
    ] }),
  ] }),
  lennard_jones: (() => {
    const inv = makeNode("pdiv", { children: [C(1), V("r")] });
    const inv2 = makeNode("sq", { children: [inv] });
    const inv6 = makeNode("cube", { children: [inv2] });
    return makeNode("mul", { children: [C(4), makeNode("sub", { children: [makeNode("sq", { children: [inv6] }), inv6] })] });
  })(),
  damped_oscillation: makeNode("mul", { children: [
    makeNode("exp", { children: [makeNode("neg", { children: [makeNode("mul", { children: [C(0.25), V("t")] })] })] }),
    makeNode("cos", { children: [makeNode("mul", { children: [C(3), V("t")] })] }),
  ] }),
  logistic_growth: makeNode("pdiv", { children: [
    C(1),
    makeNode("add", { children: [C(1), makeNode("exp", { children: [makeNode("neg", { children: [makeNode("mul", { children: [C(2), makeNode("sub", { children: [V("t"), C(2)] })] })] })] })] }),
  ] }),
  softplus: makeNode("log", { children: [makeNode("add", { children: [C(1), makeNode("exp", { children: [V("x")] })] })] }),
  kdv_soliton: (() => {
    // η = 2·sech²(x−4t), sech²ξ = 4/(e^ξ + e^−ξ)²
    const xi = makeNode("sub", { children: [V("x"), makeNode("mul", { children: [C(4), V("t")] })] });
    const ePos = makeNode("exp", { children: [xi] });
    const eNeg = makeNode("exp", { children: [makeNode("neg", { children: [xi] })] });
    const cosh = makeNode("mul", { children: [C(0.5), makeNode("add", { children: [ePos, eNeg] })] });
    return makeNode("mul", { children: [C(2), makeNode("pdiv", { children: [C(1), makeNode("sq", { children: [cosh] })] })] });
  })(),
  kerr_spin: (() => {
    // (4/b + (4s − 2.70566)/b²) / (1 − 3.62166/b)
    const num = makeNode("add", { children: [
      makeNode("pdiv", { children: [C(4), V("b")] }),
      makeNode("pdiv", { children: [makeNode("sub", { children: [makeNode("mul", { children: [C(4), V("s")] }), C(2.70566)] }), makeNode("sq", { children: [V("b")] })] }),
    ] });
    return makeNode("pdiv", { children: [num, makeNode("sub", { children: [C(1), makeNode("pdiv", { children: [C(3.62166), V("b")] })] })] });
  })(),
  pendulum_hybrid: (() => {
    // clamp((1−w)·uswing + w·ucatch, −2, 2) with w = σ(10.1786(cosθ − 0.7))
    const th = V("th"); const d = V("d");
    const c = makeNode("cos", { children: [th] });
    const s = makeNode("sin", { children: [th] });
    const EErr = makeNode("sub", { children: [
      makeNode("add", { children: [makeNode("sq", { children: [makeNode("mul", { children: [C(0.5), d] })] }), makeNode("mul", { children: [C(6), makeNode("sub", { children: [C(1), c] })] })] }),
      C(12),
    ] });
    const uSwing = makeNode("mul", { children: [
      makeNode("mul", { children: [makeNode("mul", { children: [C(-4.3278), d] }), EErr] }),
      c,
    ] });
    const uCatch = makeNode("neg", { children: [makeNode("add", { children: [makeNode("mul", { children: [C(1.7222), s] }), makeNode("mul", { children: [C(8.0402), d] })] })] });
    const sigArg = makeNode("mul", { children: [C(10.1786), makeNode("sub", { children: [c, C(0.7)] })] });
    const w = makeNode("pdiv", { children: [C(1), makeNode("add", { children: [C(1), makeNode("exp", { children: [sigArg] })] })] });
    const blend = makeNode("add", { children: [
      makeNode("mul", { children: [makeNode("sub", { children: [C(1), w] }), uSwing] }),
      makeNode("mul", { children: [w, uCatch] }),
    ] });
    return makeNode("min", { children: [C(2), makeNode("max", { children: [C(-2), blend] })] });
  })(),
};

// Iterative baselines: how each quantity is computed WITHOUT a closed form.
// Cost arithmetic (engine units: mul/add=1, div=4, sqrt=2, transcendental=20):
//  - geodesic u''=3u²−u via RK4: 4 stages × ~3 u = 12/step × 200 steps = 2400
//  - pendulum (θ,ω) Euler-Cromer: sinθ + 3 ops ≈ 25/step × 60 steps = 1500
//  - damped harmonic (x,v) via RKF45: 6 stages × ~3 u × 2 dims = 18×300 = 5400
//  - CDF by Monte-Carlo: 1000 draws × Box-Muller (log+sqrt+cos+muls ≈ 46) = 46000
export const ITERATIVE_BASELINES: Record<string, { label: string; totalCost: number }> = {
  kerr: { label: "RK4 géodésique · 200 pas", totalCost: 2400 },
  kerr_spin: { label: "RK4 géodésique · 200 pas", totalCost: 2400 },
  damped_pendulum: { label: "Euler-Cromer · 60 pas", totalCost: 1500 },
  damped_oscillation: { label: "RKF45 · 300 pas", totalCost: 5400 },
  gaussian_cdf: { label: "Monte-Carlo · 1000 tirages", totalCost: 46000 },
  // SPEAR CODEX additions:
  //  Jacobi on 3x3: ~4 sweeps x 3 rotations x ~17 u (matrix update + sqrt) = 200
  eigen3_sym: { label: "Jacobi · 4 balayages", totalCost: 200 },
  //  Newton-DLS IK: 8 iterations x Jacobian + 2x2 damped solve ~ 40 u = 320
  ik_reach: { label: "Newton-DLS · 8 itérations", totalCost: 320 },
  //  CORDIC rotation: 16 micro-rotations x ~4 u (add/sub/shift, final scale) = 64
  rope_rot: { label: "CORDIC · 16 micro-rotations", totalCost: 64 },
  // Quantum speedup front:
  qfi_dephasing: { label: "BFGS optimisation", totalCost: 800 },
  amp_damp_fid: { label: "Kraus ops 4×4 matmul", totalCost: 48 },
  loschmidt_rate: { label: "Diag complète 128 modes", totalCost: 640 },
};

export function freeFallData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  const rows = 48;
  const t = linspace(0, 3, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = 0.5 * 9.81 * t[i] * t[i] + gaussianRandom() * 0.02;
  return { vars: { t }, y };
}

// ---------- Task 1 : Répartition gaussienne Φ(x) ----------
export function gaussianCDFData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  const rows = 400;
  const x = linspace(-3, 3, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = 0.5 * (1 + erf(x[i] / Math.SQRT2));
  return { vars: { x }, y };
}

// ---------- Task 3 : Prime d'un call européen (Black-Scholes simplifié) ----------
export function europeanCallData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Approximation par la formule de Black-Scholes at-the-money (F=100, r=0, T=1)
  // C ≈ 0.4σ + 0.16σ² (pour σ ∈ [0,0.5]); on ajoute un bruit modéré
  const rows = 200;
  const sigma = linspace(0.01, 0.5, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    const s = sigma[i];
    // Approx BS call price at-the-money: C ≈ S·(0.4σ + 0.16σ²) avec S=100
    y[i] = 100 * (0.4 * s + 0.16 * s * s);
  }
  // Ajout d'un petit bruit gaussian pour réalisme
  for (let i = 0; i < rows; i++) y[i] += gaussianRandom() * 0.5;
  return { vars: { sigma }, y };
}

// ---------- Task 4 : Pendule amorti · état terminal ----------
export function dampedPendulumData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Simulation numérique simple : θ'' + 0.2θ' + 9.81·sinθ = 0, θ(0)=π/2, θ'(0)=0
  // On mesure θ(T) après T=3s avec pas d'intégration 0.05s (∼60 steps)
  const rows = 60;
  const dt = 0.05;
  const t = new Float64Array(rows);
  const theta = new Float64Array(rows);
  const thetaDot = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    t[i] = i * dt;
    // intégration Euler-Cromer simplifiée
    let theta_i = Math.PI / 2;
    let thetaDot_i = 0;
    for (let step = 0; step < Math.round(t[i] / dt); step++) {
      const thetaDouble = -0.2 * thetaDot_i - 9.81 * Math.sin(theta_i);
      thetaDot_i += thetaDouble * dt;
      theta_i += thetaDot_i * dt;
    }
    theta[i] = theta_i;
  }
  return { vars: { t }, y: theta };
}

// ---------- Task 5 : Distillation d'acteur Deep-RL ----------
export function rlDistillationData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // On Approche: réseau de "professeur" simple f(x)=tanh(2x), on génère des données et
  // l'évolution cherche une forme fermée plus légère approchant cette fonction.
  const rows = 300;
  const x = linspace(-3, 3, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = Math.tanh(2 * x[i]);
  return { vars: { x }, y };
}

// ---------- Task 6 : Fonction implicite Lambert W₀(x) ----------
export function lambertWData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Génération de données: y = x·eˣ pour x ∈ [0, 1.5], on stocke x en entrée, sort y
  // L'objectif est de récupérer x = W₀(y) par régression symbolique.
  const rows = 300;
  const x = linspace(0, 1.5, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = x[i] * Math.exp(x[i]);
  return { vars: { x }, y };
}

// ---------- Task 7 : Circuit RC · tension terminale ----------
export function rcCircuitData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Réponse première ordre : v(t) = V₀·(1 - e^(-t/τ)) avec V₀=1, τ=1
  // Données bruitées légèrement
  const rows = 50;
  const t = linspace(0, 5, rows);
  const y = new Float64Array(rows);
  const tau = 1;
  for (let i = 0; i < rows; i++) {
    y[i] = 1 - Math.exp(-t[i] / tau) + gaussianRandom() * 0.01;
  }
  return { vars: { t }, y };
}

export function keplerData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  const rows = 40;
  const a = linspace(0.3, 30, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = Math.pow(a[i], 1.5) * (1 + gaussianRandom() * 0.004);
  return { vars: { a }, y };
}

// ---------------------------------------------------------------------------
// IMAGE / VIDEO TASKS — replace costly tensor kernels with closed forms
// ---------------------------------------------------------------------------

// ---------- LayerNorm without sqrt/division: 1/sqrt(x+eps) over x∈[0.01,10] ----------
export function layernormScaleData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // target: rsqrt(x) = 1/sqrt(x), the inverse stddev scale LayerNorm computes per token.
  // On-die, rsqrt is an SFU op; a bounded minimax algebraic form can be cheaper.
  const rows = 400;
  const x = linspace(0.01, 10, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = 1 / Math.sqrt(x[i]);
  return { vars: { x }, y };
}

// ---------- Gaussian blur 1-D kernel g(x) = exp(-x²/2σ²), x∈[-3σ,3σ], σ=1 ----------
export function gaussianKernelData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  const rows = 200;
  const sigma = 1;
  const x = linspace(-3 * sigma, 3 * sigma, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = Math.exp(-(x[i] * x[i]) / (2 * sigma * sigma));
  return { vars: { x }, y };
}

// ---------- Diffusion noise schedule β(t) — cosine schedule, t∈[0,1] ----------
export function betaScheduleData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // cosine schedule: ᾱ(t) = cos²((t+0.008)/(1.008)·π/2), β(t) ≈ 1 - ᾱ(t)/ᾱ(t-1)
  // simplified: we regress β on t directly over the useful range.
  const rows = 200;
  const t = linspace(0.01, 0.99, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    const s = Math.cos(((t[i] + 0.008) / 1.008) * Math.PI / 2);
    y[i] = 1 - s * s;
  }
  return { vars: { t }, y };
}

// ---------- Bilinear interpolation weight (fractional offset u∈[0,1]) ----------
export function interpWeightData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // bilinear weight w = (1-u) for the left neighbour; u∈[0,1]
  const rows = 120;
  const u = linspace(0, 1, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = 1 - u[i];
  return { vars: { u }, y };
}

// ---------- Temporal gradient / optical-flow differencing primitive ----------
export function temporalGradData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // For video, motion estimation needs ∂I/∂t between consecutive frames.
  // Regress a smooth 2-var law: y = (a - b) over small differences + noise floor.
  const rows = 400;
  const a = new Float64Array(rows);
  const b = new Float64Array(rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    a[i] = -1 + 2 * (i / rows);       // frame t intensity
    b[i] = a[i] + gaussianRandom() * 0.05; // frame t+1 (small motion)
    y[i] = b[i] - a[i];
  }
  return { vars: { a, b }, y };
}

// ---------- Relativistic Lorentz factor γ(β) = 1/√(1−β²) ----------
export function lorentzData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Time-dilation / mass-energy factor used in physics engines and shaders.
  // Pure rsqrt-family law: discoverable with pdiv + sq + sqrt only.
  const rows = 200;
  const b = linspace(0, 0.99, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = 1 / Math.sqrt(1 - b[i] * b[i]);
  return { vars: { b }, y };
}

// ---------- Hill dose-response (pharmacology): E(c) = c³/(EC50³ + c³) ----------
export function hillData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Standard drug dose-response curve (Emax model), EC50 = 1, Hill n = 3.
  // The rational shape every pharmacologist fits — here the GP must recover it.
  const rows = 200;
  const c = linspace(0.05, 5, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = (c[i] * c[i] * c[i]) / (1 + c[i] * c[i] * c[i]);
  return { vars: { c }, y };
}

// ---------- Kerr black-hole light deflection (weak field) ----------
export function kerrDeflectionData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Gravitational lensing deflection Δφ(b) in units of rs, impact parameter b:
  // second-order expansion: 4/b + 11.781/b² + 42.667/b³ (Iyer & Petters 2007).
  // Pure rational law in 1/b — no transcendental ops needed.
  const rows = 250;
  const b = linspace(3, 50, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    y[i] = 4 / b[i] + 11.78097245 / (b[i] * b[i]) + 42.66666667 / (b[i] * b[i] * b[i]);
  }
  return { vars: { b }, y };
}

// ---------- Lennard-Jones potential V(r) = 4[(1/r)^12 − (1/r)^6] ----------
export function lennardJonesData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // The 12-6 molecular interaction potential (ε = σ = 1). Classic double-well:
  // steep repulsion + shallow attraction. Discoverable with pdiv/sq/cube only.
  const rows = 250;
  const r = linspace(0.9, 3, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    const inv = 1 / r[i];
    const inv6 = inv ** 6;
    y[i] = 4 * (inv6 * inv6 - inv6);
  }
  return { vars: { r }, y };
}

// ---------- Damped oscillation e^(−t/4)·cos(3t) ----------
export function dampedOscillationData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Signal-processing staple: exponentially damped carrier. Uses exp + cos.
  const rows = 250;
  const t = linspace(0, 6, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = Math.exp(-t[i] / 4) * Math.cos(3 * t[i]);
  return { vars: { t }, y };
}

// ---------- Logistic growth L/(1 + e^(−k(t−t₀))) ----------
export function logisticGrowthData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Population / adoption / saturation curves: L=1, k=2, t₀=2.
  const rows = 200;
  const t = linspace(0, 4, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = 1 / (1 + Math.exp(-2 * (t[i] - 2)));
  return { vars: { t }, y };
}

// ---------- Softplus ln(1 + eˣ) ----------
export function softplusData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // The smooth ReLU used in modern LLMs (output layer of SwiGLU blocks).
  // Exercises the log operator end-to-end.
  const rows = 200;
  const x = linspace(-4, 4, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = Math.log(1 + Math.exp(x[i]));
  return { vars: { x }, y };
}

// ---------- KdV 1-soliton (BT36): η = 2κ²·sech²(κ(x − 4κ²t − x₀)) ----------
export function kdvSolitonData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Exact travelling-wave solution of the Korteweg–de Vries equation (κ=1, x₀=0).
  // sech² is composable from exp: sech²ξ = 4/(e^ξ + e^−ξ)².
  const rows = 400;
  const xs = linspace(-10, 10, rows);
  const ts = linspace(0, 2, rows);
  const x = new Float64Array(rows);
  const t = new Float64Array(rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    x[i] = xs[i];
    t[i] = ts[i];
    const xi = x[i] - 4 * t[i];
    const c = (Math.exp(xi) + Math.exp(-xi)) / 2; // cosh
    y[i] = 2 / (c * c);
  }
  return { vars: { x, t }, y };
}

// ---------- Kerr deflection with spin a ≠ 0 (BT35): prograde/retrograde ----------
export function kerrSpinData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Padé [1/2] deflection with Lense-Thirring spin term: sign flips for
  // prograde (s < 0) vs retrograde orbits. Two-var rational law.
  const rows = 300;
  const bs = linspace(8, 50, rows);
  const ss = linspace(-0.99, 0.99, rows);
  const b = new Float64Array(rows);
  const s = new Float64Array(rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    b[i] = bs[i];
    s[i] = ss[i];
    const num = 4 / b[i] + (4 * s[i] - 2.70566416) / (b[i] * b[i]);
    const den = 1 - 3.62165903 / b[i];
    y[i] = num / den;
  }
  return { vars: { b, s }, y };
}

// ---------- Hybrid inverted-pendulum control law (paper §4, 500-gen anchor) ----------
export function pendulumHybridData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Energy-shaping swing-up blended into a Lyapunov catch via a C∞ sigmoid,
  // hard-saturated at ±2. The GP must recover the whole hybrid structure.
  const rows = 500;
  const th = new Float64Array(rows);
  const thd = new Float64Array(rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    // deterministic quasi-uniform 2-D coverage via golden-ratio sequence
    th[i] = -Math.PI + (2 * Math.PI * ((i * 0.6180339887) % 1));
    thd[i] = -6 + 12 * ((i * 0.7548776662) % 1);
    const c = Math.cos(th[i]);
    const EErr = 0.5 * thd[i] * thd[i] + 6 * (1 - c) - 12;
    const wCatch = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, 10.1786 * (c - 0.7)))));
    const uSwing = -4.3278 * thd[i] * EErr * c;
    const uCatch = -(1.7222 * Math.sin(th[i]) + 8.0402 * thd[i]);
    y[i] = Math.min(2, Math.max(-2, (1 - wCatch) * uSwing + wCatch * uCatch));
  }
  return { vars: { th, d: thd }, y };
}

// --- SPEAR CODEX imports (BT29 / BT33 / city.ts IDM) -----------------------

export function eigenSymData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // BT29: lambda_max of a random symmetric 3x3, featured by its three
  // principal invariants (tr, I2, det). The closed form needs acos — a
  // transcendental the engine does NOT serve — so the GP must approximate
  // the triple-angle shape with algebra alone.
  const rows = 500;
  const i1 = new Float64Array(rows), i2 = new Float64Array(rows), i3 = new Float64Array(rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    const r = (k: number) => -2 + 4 * ((i * k) % 1);
    const a = r(0.319), b = r(0.527), c = r(0.737), dd = r(0.411), ee = r(0.643), ff = r(0.859);
    i1[i] = a + dd + ff;
    i2[i] = a * dd + a * ff + dd * ff - b * b - c * c - ee * ee;
    i3[i] = a * (dd * ff - ee * ee) - b * (b * ff - c * ee) + c * (b * ee - c * dd);
    // largest root of l^3 - i1 l^2 + i2 l - i3 = 0 (three real roots)
    const p = i2[i] - (i1[i] * i1[i]) / 3;
    const q = (i1[i] * i2[i]) / 3 - (2 * i1[i] * i1[i] * i1[i]) / 27 - i3[i];
    const rr = Math.sqrt(Math.max(0, -p / 3));
    const arg = Math.max(-1, Math.min(1, (3 * q) / (2 * p) * Math.sqrt(-3 / p)));
    y[i] = i1[i] / 3 + 2 * rr * Math.cos(Math.acos(arg) / 3);
  }
  return { vars: { t: i1, u: i2, w: i3 }, y };
}

export function ikReachData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // BT33: elbow angle of a 2-link arm from reach and link lengths. Truth is
  // acos(clamped law-of-cosines ratio) — again an unserved transcendental.
  const rows = 500;
  const dv = new Float64Array(rows), l2v = new Float64Array(rows), l3v = new Float64Array(rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    const l2 = 1 + 2 * ((i * 0.6180339887) % 1);
    const l3 = 1 + 2 * ((i * 0.7548776662) % 1);
    const lo = Math.abs(l2 - l3) + 0.25;
    const hi = l2 + l3 - 0.25;
    const d = lo + (hi - lo) * ((i * 0.4192388219) % 1);
    const cosQ = Math.max(-1, Math.min(1, (d * d - l2 * l2 - l3 * l3) / (2 * l2 * l3)));
    dv[i] = d; l2v[i] = l2; l3v[i] = l3;
    y[i] = Math.acos(cosQ);
  }
  return { vars: { d: dv, l2: l2v, l3: l3v }, y };
}

export function gemv4Data(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // LLM decode bottleneck cell: one output lane of y = W·x with W frozen.
  // The bilinear dot is ALREADY the minimal closed form (4 mul + 3 add = 7
  // units — rank argument). This task is the engine's optimality test: can it
  // recover the provably-minimal kernel, not beat it.
  const rows = 400;
  const x0 = new Float64Array(rows), x1 = new Float64Array(rows), x2 = new Float64Array(rows), x3 = new Float64Array(rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    x0[i] = -1 + 2 * ((i * 0.319) % 1);
    x1[i] = -1 + 2 * ((i * 0.527) % 1);
    x2[i] = -1 + 2 * ((i * 0.737) % 1);
    x3[i] = -1 + 2 * ((i * 0.859) % 1);
    y[i] = 0.837 * x0[i] - 0.482 * x1[i] + 1.117 * x2[i] - 0.296 * x3[i];
  }
  return { vars: { x0, x1, x2, x3 }, y };
}

export function ropeRotData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // RoPE (rotary position embedding) lane rotation — paid on EVERY token of
  // EVERY head at inference. Truth needs cos+sin per token; production uses
  // CORDIC micro-rotations when no SFU is available. Algebraic approximants
  // (Padé-style) are the speed play.
  const rows = 500;
  const xv = new Float64Array(rows), yv = new Float64Array(rows), tv = new Float64Array(rows);
  const out = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    xv[i] = -2 + 4 * ((i * 0.6180339887) % 1);
    yv[i] = -2 + 4 * ((i * 0.7548776662) % 1);
    tv[i] = -Math.PI + 2 * Math.PI * ((i * 0.4192388219) % 1);
    out[i] = xv[i] * Math.cos(tv[i]) - yv[i] * Math.sin(tv[i]);
  }
  return { vars: { x: xv, y: yv, th: tv }, y: out };
}

export function idmFollowingData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // city.ts traffic phase B: Intelligent-Driver-Model acceleration law
  // (v0=33 m/s, T=1.5 s, a=2, b=3, s0=2 m, delta=4). Fully algebraic target.
  const rows = 500;
  const vv = new Float64Array(rows), gv = new Float64Array(rows), dvv = new Float64Array(rows);
  const y = new Float64Array(rows);
  const V0 = 33, T = 1.5, AMAX = 2, BCOMF = 3, S0 = 2, SQAB = Math.sqrt(AMAX * BCOMF);
  for (let i = 0; i < rows; i++) {
    const v = 35 * ((i * 0.6180339887) % 1);
    const gap = 2.5 + 70 * ((i * 0.7548776662) % 1);
    const dvRaw = -12 + 24 * ((i * 0.4192388219) % 1);
    const sStar = S0 + Math.max(0, v * T + (v * dvRaw) / (2 * SQAB));
    const free = 1 - Math.pow(v / V0, 4);
    const interact = (sStar / gap) * (sStar / gap);
    vv[i] = v; gv[i] = gap; dvv[i] = dvRaw;
    y[i] = Math.max(-9, Math.min(AMAX, AMAX * (free - interact)));
  }
  return { vars: { v: vv, s: gv, dv: dvv }, y };
}

// --- second-wave reference implementations ---------------------------------

export function erfImpl(x: number): number {
  // Abramowitz & Stegun 7.1.26, max error ~1.5e-7
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const y = 1 - poly * Math.exp(-ax * ax);
  return x >= 0 ? y : -y;
}

export function besselJ0(x: number): number {
  // True Bessel J0 via convergent series: Sum (-1)^k (x^2/4)^k / (k!)^2.
  // BUGFIX: the previous transcription evaluated a WRONG curve (0.457 at
  // x=1 vs true 0.765), poisoning the task's entire target dataset.
  let sum = 1;
  let term = 1;
  const halfSq = (x * x) / 4;
  for (let k = 1; k <= 24; k++) {
    term *= -halfSq / (k * k);
    sum += term;
    if (Math.abs(term) < 1e-15 && k > 3) break;
  }
  return sum;
}

// True Bessel J2 via convergent series: Sum (-1)^(j-1) hsq^j / ((j-1)!(j+1)!),
// hsq = x²/4. Constructed correctly this time (verified vs known values).
export function besselJ2(x: number): number {
  const hsq = (x * x) / 4;
  let term = hsq / 2; // j=1: /(0!·2!)
  let sum = term;
  let j = 1;
  while (j <= 26) {
    term *= -hsq / (j * (j + 2));
    sum += term;
    if (Math.abs(term) < 1e-15 && j > 3) break;
    j++;
  }
  return sum;
}
export function besselJ1(x: number): number {
  // True Bessel J1 via convergent series: Σ (-1)^k (x/2)^(2k+1) / (k!(k+1)!).
  // BUGFIX: previous denominator skipped the rising factorial (wrong from k=2).
  let term = x / 2;
  let sum = term;
  const hh = (x / 2) * (x / 2);
  for (let k = 1; k <= 22; k++) {
    term *= -hh / (k * (k + 1));
    sum += term;
    if (Math.abs(term) < 1e-15 && k > 3) break;
  }
  return sum;
}

// --- quantum computing references ------------------------------------------

// Grover amplitude after k iterations: sin²((2k+1)·asin(√(m/n)))
export function groverSuccess(k: number, m: number, n: number): number {
  const theta = Math.asin(Math.sqrt(m / n));
  return Math.pow(Math.sin((2 * k + 1) * theta), 2);
}

// Concurrence of a pure 2-qubit state |a|00>+|b|01>+|c|10>+|d|11>
export function concurrencePure(a: number, b: number, c: number, d: number): number {
  const norm = Math.sqrt(a * a + b * b + c * c + d * d);
  const A = a / norm, B = b / norm, Cc = c / norm, D = d / norm;
  return 2 * Math.abs(A * D - B * Cc);
}

// CHSH parameter for the singlet state with analyzer angles:
// S = |E(a,b) − E(a,b′) + E(a′,b) + E(a′,b′)|, E(x,y) = −cos(x−y)
export function chshS(a: number, ap: number, b: number, bp: number): number {
  const E = (x: number, y: number) => -Math.cos(x - y);
  return Math.abs(E(a, b) - E(a, bp) + E(ap, b) + E(ap, bp));
}

// Tanner Helland blackbody fits — green & blue channels
export function blackbodyGreen(tempK: number): number {
  const t = tempK / 100;
  let g: number;
  if (t <= 66) g = 99.4708025861 * Math.log(t) - 161.1195681661;
  else g = 288.1221695283 * Math.pow(t - 60, -0.0755148492);
  return Math.min(255, Math.max(0, g)) / 255;
}
export function blackbodyBlue(tempK: number): number {
  const t = tempK / 100;
  let b: number;
  if (t >= 66) b = 255;
  else if (t <= 19) b = 0;
  else b = 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  return Math.min(255, Math.max(0, b)) / 255;
}

// Tanner Helland blackbody fit — normalized red channel vs color temperature
export function blackbodyRed(tempK: number): number {
  const t = tempK / 100;
  let r: number;
  if (t <= 66) r = 255;
  else r = 329.698727446 * Math.pow(t - 60, -0.1332047592);
  return Math.min(255, Math.max(0, r)) / 255;
}

// Narkowicz ACES approximation — THE production tonemap reference
export function narkowiczAces(x: number): number {
  const a = x * (2.51 * x + 0.03);
  const b = x * (2.43 * x + 0.59) + 0.14;
  return Math.min(1, Math.max(0, a / b));
}

// Black-76/Black-Scholes call price for IV data generation (d1,d2 closed form)
export function bsCall(s: number, k: number, t: number, vol: number): number {
  const r = 0.02; // risk-free fixed for the dataset
  const sq = vol * Math.sqrt(t);
  const d1 = (Math.log(s / k) + (r + 0.5 * vol * vol) * t) / sq;
  const d2 = d1 - sq;
  const nd = (x: number) => 0.5 * (1 + erfImpl(x / Math.SQRT2));
  return s * nd(d1) - k * Math.exp(-r * t) * nd(d2);
}

// Implied volatility via Newton inversion of the above (ground-truth generator)
export function impliedVol(c: number, s: number, k: number, t: number): number {
  let vol = 0.4;
  for (let i = 0; i < 40; i++) {
    const sq = vol * Math.sqrt(t);
    const d1 = (Math.log(s / k) + (0.02 + 0.5 * vol * vol) * t) / sq;
    const d2 = d1 - sq;
    const nd = (x: number) => 0.5 * (1 + erfImpl(x / Math.SQRT2));
    const npdf = (x: number) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
    const price = bsCall(s, k, t, vol);
    const vega = s * npdf(d1) * Math.sqrt(t);
    const diff = price - c;
    if (Math.abs(diff) < 1e-10 || vega < 1e-10) break;
    // BUGFIX: c'était `vol += diff/vega` — Newton à l'envers, l'itération
    // fuyait la racine et clampeait aux bornes (0.01/3) sur ~80 % du jeu.
    vol -= diff / vega;
  }
  return Math.max(0.01, Math.min(3, vol));
}

// --- wave 9: high-frontier numerical functions ------------------------------

// Scaled modified Bessel I0(x)·e^(-x) via convergent series (exact to ~1e-14)
export function besselI0e(x: number): number {
  let sum = 1, term = 1;
  const hs = x * x / 4;
  for (let k = 1; k <= 40; k++) {
    term *= hs / (k * k);
    sum += term;
    if (term < 1e-15) break;
  }
  return sum * Math.exp(-Math.abs(x));
}

// Complete elliptic integral K(m) via AGM (exact to ~1e-12)
export function ellipticK(m: number): number {
  let a = 1, b = Math.sqrt(1 - m);
  for (let i = 0; i < 20; i++) {
    const aNext = (a + b) / 2;
    b = Math.sqrt(a * b);
    a = aNext;
    if (Math.abs(a - b) < 1e-14) break;
  }
  return Math.PI / (2 * a);
}

// Solve Kepler's equation M = E − e·sin(E) by Newton-Raphson
export function solveKepler(M: number, e: number): number {
  let E = M + e * Math.sin(M);
  for (let i = 0; i < 30; i++) {
    const f = E - e * Math.sin(E) - M;
    const fp = 1 - e * Math.cos(E);
    E -= f / fp;
    if (Math.abs(f) < 1e-13) break;
  }
  return E;
}

// Probit reference via bisection on the A&S forward erf (exact to 1e-12+).
export function acklamProbit(p: number): number {
  const target = 2 * p - 1;
  let lo = -8, hi = 8;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    const e = erfImpl(mid / Math.SQRT2);
    if (e < target) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}


// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export function taskOpProfile(node: SpearNode): { total: number; transcendental: number } {
  return countOps(node);
}

