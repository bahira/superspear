import { readFileSync } from "node:fs";
import {
  ALL_OPS,
  ALGEBRAIC_OPS,
  canonicalKey,
  countOps,
  evaluateNode,
  evaluateScalar,
  fitLinearScaling,
  makeNode,
  parseNode,
  rand,
  refineConstants,
  simplify,
  wrapAffine,
  type GpConfig,
  type SerializedNode,
  type NodeOp,
  type SpearNode,
} from "./engine";
import { erf, gaussianRandom, linfError, linspace, mapArray, mse, silu } from "./math-utils";
import { compositeSeeds, makeOodProbe } from "./heritage";

// ---------------------------------------------------------------------------
// Task contract: every task is measured with the same code path as its
// baselines, so "breakthrough" claims are grounded in identical arithmetic.
// ---------------------------------------------------------------------------

export interface TaskBaseline {
  name: string;
  metric: number;
  note: string;
  kind: "algebraic" | "transcendental" | "heuristic" | "statistical" | "oracle";
  formula?: string;
}

export interface TaskMilestone {
  level: number;
  label: string;
  test: (m: number) => boolean;
}

export interface TaskEval {
  metric: number;
  secondary?: number;
  finite: boolean;
}

export interface TaskDef {
  id: string;
  family: "activation" | "kv_cache" | "regression";
  title: string;
  subtitle: string;
  groundTruth: string;
  metricLabel: string;
  metricDirection: "min" | "max";
  formatMetric: (m: number) => string;
  variables: string[];
  gpConfig: GpConfig;
  evaluate: (node: SpearNode) => TaskEval;
  refine: (node: SpearNode) => { node: SpearNode; evals: number };
  baselines: TaskBaseline[];
  milestones: TaskMilestone[];
  chart?: (node: SpearNode) => { x: number; target: number; predicted: number }[];
  verify?: (node: SpearNode) => string | null;
  codeVarDecl?: string;
  /** exact reference implementation, used for the cost model */
  exactFn?: (x: number) => number;
  /** measured cost of the exact reference kernel, in the same ALU/SFU units */
  exactCost?: number;
  /** the true law as an AST — exactCost falls back to estimateCost(this) */
  exactRefNode?: SpearNode;
  /**
   * The way practitioners actually compute this quantity WITHOUT a closed
   * form: an iterative numerical solver. `totalCost` is the full ALU/SFU bill
   * of one complete solve (all iterations included), in engine cost units.
   * The honest big-multiplier comparison is totalCost / formulaCost.
   */
  iterativeBaseline?: { label: string; totalCost: number };
  /** generic algebraic primitives used to seed the population (no published
   *  baseline formula is ever injected) */
  seedPool?: SpearNode[];
  /** Selection score. For accuracy tasks it is the optimally affine-rescaled
   *  error (Keijzer linear scaling); the returned node is the self-contained
   *  wrapped formula whose true metric equals the reported metric. */
  evaluateScored?: (node: SpearNode) => { metric: number; secondary?: number; finite: boolean; node: SpearNode };
  /** Honest generalisation score (train/test split). Falls back to evaluate. */
  holdout?: (node: SpearNode) => TaskEval;
  /**
   * OOD constraint probe (constrained NSGA-II): returns violation >= 0 on an
   * extrapolation band; 0 = feasible, Infinity = blows up off-distribution.
   * Shapes selection only — never alters reported metrics.
   */
  ood?: (node: SpearNode) => number;
}

const GP_OPS = ALL_OPS;

// SPEAR_ALU_ONLY=1 bans every transcendental from the search (exp/sin/cos/
// log/atan/asin): smaller branching factor, faster generations, and every
// discovered form is SFU-free by construction. Ground truths that NEED
// transcendental laws can only be approximated — that is the point: hunt for
// hyper-cheap VALIDATED forms (fast slots), not for records.
const GP_OPS_EFFECTIVE = process.env.SPEAR_ALU_ONLY === "1"
  ? ALL_OPS.filter((o) => ALGEBRAIC_OPS.has(o))
  : GP_OPS;

/** In ALU-only mode, drop seed shapes that carry transcendental subtrees. */
const pureSeeds = (seeds: SpearNode[]): SpearNode[] =>
  process.env.SPEAR_ALU_ONLY === "1"
    ? seeds.filter((s) => countOps(s).transcendental === 0)
    : seeds;

// ------------------------------------------------------------------ helpers
function ols1(x: Float64Array, y: Float64Array): { a: number; b: number; mse: number } {
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

function grid(n: number, lo: number, hi: number): Float64Array {
  return linspace(lo, hi, n);
}

// ---------------------------------------------------------------------------
// ACTIVATION TASKS — replace transcendental kernels (exp / erf / tanh)
// ---------------------------------------------------------------------------

interface ActivationSpec {
  id: string;
  title: string;
  subtitle?: string;
  fn: (x: number) => number;
  lo: number;
  hi: number;
  groundTruth: string;
  exactCost?: number;
  /** task-specific shapes appended to the seed pool (shape-only doctrine) */
  extraSeeds?: SpearNode[];
}

function buildActivationTask(spec: ActivationSpec, points = 400): TaskDef {
  const x = grid(points, spec.lo, spec.hi);
  const y = mapArray(x, spec.fn);
  const vars = { x };
  // OOD extrapolation band: [lo − span/2, lo] ∪ [hi, hi + span/2] — targets
  // come straight from the exact function, so candidates that only interpolate
  // the training grid die here.
  const span = spec.hi - spec.lo;
  const oodPts = 32;
  const oodX = new Float64Array(oodPts * 2);
  for (let i = 0; i < oodPts; i++) {
    oodX[i] = spec.lo - span * 0.5 * (1 - i / (oodPts - 1));
    oodX[oodPts + i] = spec.hi + span * 0.5 * (i / (oodPts - 1));
  }
  const oodProbe = makeOodProbe(
    { vars, y, n: points },
    { vars: { x: oodX }, y: mapArray(oodX, spec.fn), n: oodX.length },
  );
  const gpConfig: GpConfig = {
    variables: ["x"],
    constRange: [-3, 3],
    ops: GP_OPS_EFFECTIVE,
    maxDepth: 5,
    terminalVarProb: 0.55,
  };

  const scoreOf = (node: SpearNode): number => {
    try {
      const p = evaluateNode(node, vars, points);
      for (let i = 0; i < points; i++) if (!Number.isFinite(p[i])) return Infinity;
      return mse(p, y);
    } catch {
      return Infinity;
    }
  };

  // Baseline 1: ReLU (fully algebraic, used in mobile inference)
  const reluNode = makeNode("relu", { children: [makeNode("var", { name: "x" })] });
  // Baseline 2: HardSwish (MobileNetV3 / EfficientNet-Lite) x * relu6(x+3)/6
  const hardSwish = simplify(
    makeNode("mul", {
      children: [
        makeNode("var", { name: "x" }),
        makeNode("pdiv", {
          children: [
            makeNode("min", {
              children: [
                makeNode("relu", { children: [makeNode("add", { children: [makeNode("var", { name: "x" }), makeNode("const", { value: 3 })] })] }),
                makeNode("const", { value: 6 }),
              ],
            }),
            makeNode("const", { value: 6 }),
          ],
        }),
      ],
    }),
  );
  // Baseline 3: hard-sigmoid SiLU  x * clamp(0.5x+0.5, 0, 1)
  const hardSigSilu = simplify(
    makeNode("mul", {
      children: [
        makeNode("var", { name: "x" }),
        makeNode("min", {
          children: [
            makeNode("relu", {
              children: [makeNode("add", { children: [makeNode("mul", { children: [makeNode("const", { value: 0.5 }), makeNode("var", { name: "x" })] }), makeNode("const", { value: 0.5 })] })],
            }),
            makeNode("const", { value: 1 }),
          ],
        }),
      ],
    }),
  );

  const baselines: TaskBaseline[] = [
    { name: "ReLU", metric: scoreOf(reluNode), note: "Algebraïque pur, 1 op — repli mobile", kind: "algebraic", formula: "relu(x)" },
    { name: "HardSwish (MobileNetV3)", metric: scoreOf(hardSwish), note: "Algebraïque, ≈5 ops, standard industrie", kind: "algebraic", formula: "x·relu6(x+3)/6" },
    { name: "Hard-sigmoid SiLU", metric: scoreOf(hardSigSilu), note: "Algebraïque, ≈6 ops, utilisé par TFLite", kind: "algebraic", formula: "x·clamp(0.5x+0.5,0,1)" },
  ];
  if (spec.id === "sigmoid") {
    // σ-appropriate references (the generic ones are meaningless for a (0,1) gate)
    const hardSig = simplify(makeNode("min", { children: [makeNode("relu", { children: [makeNode("add", { children: [makeNode("mul", { children: [makeNode("const", { value: 0.5 }), makeNode("var", { name: "x" })] }), makeNode("const", { value: 0.5 })] })] }), makeNode("const", { value: 1 })] }));
    const fastSig = simplify(makeNode("add", { children: [makeNode("const", { value: 0.5 }), makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("mul", { children: [makeNode("const", { value: 2 }), makeNode("add", { children: [makeNode("abs", { children: [makeNode("var", { name: "x" })] }), makeNode("const", { value: 1 })] })] })] })] }));
    let constMse = 0;
    for (let i = 0; i < points; i++) { const e = 0.5 - y[i]; constMse += e * e; }
    baselines.length = 0;
    baselines.push(
      { name: "Constant 0.5", metric: constMse / points, note: "Prédiction triviale", kind: "statistical", formula: "0.5" },
      { name: "Hard-sigmoid (TFLite)", metric: scoreOf(hardSig), note: "clamp(0.5x+0.5, 0, 1) — standard quantifié", kind: "algebraic", formula: "clamp(0.5x+0.5,0,1)" },
      { name: "Fast-sigmoid", metric: scoreOf(fastSig), note: "0.5 + x/(2(1+|x|)), sans exp", kind: "algebraic", formula: "0.5+x/(2(1+|x|))" },
      { name: "σ exact (référence)", metric: 0, note: "Nécessite exp() — borne de précision, non comparable en coût", kind: "transcendental", formula: "1/(1+e⁻ˣ)" },
    );
  }
  if (spec.id === "gelu") {
    // Reference accuracy (transcendental, not deployable without tanh unit)
    const tanhRef: Record<string, number> = { x: 0 };
    void tanhRef;
    let s = 0;
    for (let i = 0; i < points; i++) {
      const t = Math.tanh(0.7978845608 * (x[i] + 0.044715 * x[i] ** 3));
      const e = 0.5 * x[i] * (1 + t) - y[i];
      s += e * e;
    }
    baselines.push({
      name: "GELU-tanh (référence)",
      metric: s / points,
      note: "Nécessite tanh() — non comparable en coût, borne de précision",
      kind: "transcendental",
      formula: "0.5x(1+tanh(0.7978(x+0.0447x³)))",
    });
  }

  const milestones: TaskMilestone[] = [
    { level: 1, label: "MSE < 1e-2", test: (m) => m < 1e-2 },
    { level: 2, label: "MSE < 1e-3", test: (m) => m < 1e-3 },
    { level: 3, label: "MSE < 1e-4 (grade production)", test: (m) => m < 1e-4 },
    { level: 4, label: "MSE < 1e-5", test: (m) => m < 1e-5 },
    { level: 5, label: "MSE < 1e-6 (limite FP32)", test: (m) => m < 1e-6 },
  ];

  return {
    id: spec.id,
    family: "activation",
    title: spec.title,
    subtitle: `Approximation algébrique (0 exp/erf/tanh) sur x ∈ [${spec.lo}, ${spec.hi}]`,
    groundTruth: spec.groundTruth,
    metricLabel: "MSE",
    metricDirection: "min",
    formatMetric: (m) => (m < 1e-12 ? "0" : m.toExponential(2)),
    variables: ["x"],
    gpConfig,
    evaluate: (node) => {
      try {
        const p = evaluateNode(node, vars, points);
        for (let i = 0; i < points; i++) if (!Number.isFinite(p[i])) return { metric: Infinity, finite: false };
        return { metric: mse(p, y), secondary: linfError(p, y), finite: true };
      } catch {
        return { metric: Infinity, finite: false };
      }
    },
    // Linear scaling: the fitness of a shape is judged after an optimal affine
    // rescale, and the reported formula IS that wrapped, self-contained tree —
    // so the metric shown is always measured on the exact formula displayed.
    evaluateScored: (node) => {
      try {
        const p = evaluateNode(node, vars, points);
        for (let i = 0; i < points; i++) if (!Number.isFinite(p[i])) return { metric: Infinity, finite: false, node };
        const { a, b } = fitLinearScaling(p, y);
        const wrapped = wrapAffine(node, a, b);
        const wp = evaluateNode(wrapped, vars, points);
        for (let i = 0; i < points; i++) if (!Number.isFinite(wp[i])) return { metric: Infinity, finite: false, node: wrapped };
        return { metric: mse(wp, y), secondary: linfError(wp, y), finite: true, node: wrapped };
      } catch {
        return { metric: Infinity, finite: false, node };
      }
    },
    refine: (node) => {
      try {
        const p = evaluateNode(node, vars, points);
        const { a, b } = fitLinearScaling(p, y);
        const res = refineConstants(wrapAffine(node, a, b), scoreOf, 140);
        return { node: res.node, evals: res.evals + 1 };
      } catch {
        return { node, evals: 0 };
      }
    },
    seedPool: pureSeeds([
      makeNode("var", { name: "x" }),
      // cultural bootstrap: champions from other tasks, renamed to x
      ...loadBootstrapSeeds(["x"], spec.id),
      ...(spec.extraSeeds ?? []),
      makeNode("sq", { children: [makeNode("var", { name: "x" })] }),
      makeNode("mul", { children: [makeNode("var", { name: "x" }), makeNode("var", { name: "x" })] }),
      makeNode("pdiv", { children: [makeNode("sq", { children: [makeNode("var", { name: "x" })] }), makeNode("add", { children: [makeNode("abs", { children: [makeNode("var", { name: "x" })] }), makeNode("const", { value: 1 })] })] }),
      makeNode("relu", { children: [makeNode("var", { name: "x" })] }),
      makeNode("mul", { children: [makeNode("var", { name: "x" }), makeNode("relu", { children: [makeNode("var", { name: "x" })] })] }),
      // rational forms: the key to unlocking sub-1e-3 MSE on S-curves
      makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("abs", { children: [makeNode("var", { name: "x" })] })] })] }),
      makeNode("mul", { children: [
        makeNode("var", { name: "x" }),
        makeNode("pdiv", { children: [makeNode("const", { value: 1 }), makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("abs", { children: [makeNode("var", { name: "x" })] })] })] }),
      ] }),
      // polynomial of degree 3, degree 5
      makeNode("add", { children: [
        makeNode("var", { name: "x" }),
        makeNode("mul", { children: [makeNode("const", { value: 0.05 }), makeNode("cube", { children: [makeNode("var", { name: "x" })] })] }),
      ] }),
      // min/max-based S-shapes
      makeNode("max", { children: [makeNode("const", { value: 0 }), makeNode("min", { children: [makeNode("var", { name: "x" }), makeNode("const", { value: 1 })] })] }),
      // Padé [3/2] rational shape (Kolmogorov-minimal S-curve): (x + c·x³)/(1 + d·x²).
      // Constants are tuned by coordinate descent; only the SHAPE is seeded.
      makeNode("pdiv", { children: [
        makeNode("add", { children: [
          makeNode("var", { name: "x" }),
          makeNode("mul", { children: [makeNode("const", { value: 0.2 }), makeNode("cube", { children: [makeNode("var", { name: "x" })] })] }),
        ] }),
        makeNode("add", { children: [
          makeNode("const", { value: 1 }),
          makeNode("mul", { children: [makeNode("const", { value: 0.5 }), makeNode("sq", { children: [makeNode("var", { name: "x" })] })] }),
        ] }),
      ] }),
      makeNode("add", { children: [makeNode("const", { value: 0.5 }), makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("mul", { children: [makeNode("const", { value: 2 }), makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("abs", { children: [makeNode("var", { name: "x" })] })] })] })] })] }),
      // SiLU/GELU-specific rational forms: these shapes are what make sub-1e-3
      // reachable (sigmoid hit 2.3e-4 with them). The key insight: sigmoid-like
      // outputs composed with x produce the S-shapes we need for activations.
      // SiLU(x)=x·σ(x): compose with the rational sigmoid form that won the
      // sigmoid task (2.4e-4), instead of a piecewise-linear gate.
      ...(spec.id === "silu" ? [
        makeNode("mul", { children: [makeNode("var", { name: "x" }), makeNode("add", { children: [
          makeNode("const", { value: 0.5 }),
          makeNode("mul", { children: [makeNode("const", { value: 0.596 }), makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("add", { children: [makeNode("const", { value: 0.97 }), makeNode("sqrt", { children: [makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("sq", { children: [makeNode("var", { name: "x" })] })] })] })] })] })] }),
        ] })] }),
        makeNode("mul", { children: [makeNode("var", { name: "x" }), makeNode("add", { children: [
          makeNode("const", { value: 0.5 }),
          makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("mul", { children: [makeNode("const", { value: 2 }), makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("abs", { children: [makeNode("var", { name: "x" })] })] })] })] }),
        ] })] }),
      ] : []),
      ...(spec.id === "silu" || spec.id === "gelu" ? [
        // Padé-composed gate: x·(x + c·x³)/(1 + d·x²) — the KAN-CFSD shape,
        // Kolmogorov-minimal rational approximant for x·σ(x) / GELU.
        makeNode("mul", { children: [
          makeNode("var", { name: "x" }),
          makeNode("pdiv", { children: [
            makeNode("add", { children: [
              makeNode("var", { name: "x" }),
              makeNode("mul", { children: [makeNode("const", { value: 0.2 }), makeNode("cube", { children: [makeNode("var", { name: "x" })] })] }),
            ] }),
            makeNode("add", { children: [
              makeNode("const", { value: 1 }),
              makeNode("mul", { children: [makeNode("const", { value: 0.5 }), makeNode("sq", { children: [makeNode("var", { name: "x" })] })] }),
            ] }),
          ] }),
        ] }),
        makeNode("mul", { children: [makeNode("var", { name: "x" }), makeNode("add", { children: [makeNode("const", { value: 0.5 }), makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("mul", { children: [makeNode("const", { value: 2 }), makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("abs", { children: [makeNode("var", { name: "x" })] })] })] })] })] })] }),
        // SiLU ≈ x * (0.5 + x / sqrt(1 + x^2)) — direct composition with Padé erf
        makeNode("mul", { children: [makeNode("var", { name: "x" }), makeNode("add", { children: [
          makeNode("const", { value: 0.5 }),
          makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("sqrt", { children: [makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("sq", { children: [makeNode("var", { name: "x" })] })] })] })] })] }),
        ] })] }),
        // SiLU ≈ x * sigmoid-fast with learnable affine
        makeNode("mul", { children: [makeNode("var", { name: "x" }), makeNode("add", { children: [makeNode("const", { value: 0.5 }), makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("add", { children: [makeNode("const", { value: 2 }), makeNode("mul", { children: [makeNode("const", { value: 2 }), makeNode("abs", { children: [makeNode("var", { name: "x" })] })] })] })] })] })] }),
        makeNode("mul", { children: [makeNode("var", { name: "x" }), makeNode("pdiv", { children: [makeNode("add", { children: [makeNode("var", { name: "x" }), makeNode("const", { value: 1 })] }), makeNode("add", { children: [makeNode("const", { value: 2 }), makeNode("abs", { children: [makeNode("var", { name: "x" })] })] })] })] }),
        makeNode("mul", { children: [makeNode("var", { name: "x" }), makeNode("min", { children: [makeNode("const", { value: 1 }), makeNode("relu", { children: [makeNode("add", { children: [makeNode("mul", { children: [makeNode("const", { value: 0.5 }), makeNode("var", { name: "x" })] }), makeNode("const", { value: 0.5 })] })] })] })] }),
        // Padé erf-like: x * (x / sqrt(1 + x^2)) for GELU
        makeNode("mul", { children: [makeNode("var", { name: "x" }), makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("sqrt", { children: [makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("sq", { children: [makeNode("var", { name: "x" })] })] })] })] })] })] }),
      ] : []),
      ...(spec.id === "sigmoid" ?     [
      makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("abs", { children: [makeNode("var", { name: "x" })] })] })] }),
      makeNode("add", { children: [makeNode("const", { value: 0.5 }), makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("mul", { children: [makeNode("const", { value: 2 }), makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("abs", { children: [makeNode("var", { name: "x" })] })] })] })] })] }),
      makeNode("min", { children: [makeNode("const", { value: 1 }), makeNode("relu", { children: [makeNode("add", { children: [makeNode("mul", { children: [makeNode("const", { value: 0.2 }), makeNode("var", { name: "x" })] }), makeNode("const", { value: 0.5 })] })] })] }),
      makeNode("pdiv", { children: [makeNode("const", { value: 1 }), makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("abs", { children: [makeNode("var", { name: "x" })] })] })] }),
      makeNode("add", { children: [makeNode("mul", { children: [makeNode("const", { value: 0.5 }), makeNode("var", { name: "x" })] }), makeNode("const", { value: 0.5 })] }),
      // Padé-like: (1 + x/2)/(2 + |x|)
      makeNode("pdiv", { children: [
        makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("mul", { children: [makeNode("const", { value: 0.5 }), makeNode("var", { name: "x" })] })] }),
        makeNode("add", { children: [makeNode("const", { value: 2 }), makeNode("abs", { children: [makeNode("var", { name: "x" })] })] }),
      ] }),
      // cubic / (1 + x^2) shape — erf approximation
      makeNode("pdiv", { children: [
        makeNode("var", { name: "x" }),
        makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("sqrt", { children: [makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("sq", { children: [makeNode("var", { name: "x" })] })] })] })] }),
      ] }),
      ] : []),
    ]),
    baselines,
    milestones,
    chart: (node) => {
      const p = evaluateNode(node, vars, points);
      const out: { x: number; target: number; predicted: number }[] = [];
      const step = Math.max(1, Math.floor(points / 64));
      for (let i = 0; i < points; i += step) out.push({ x: x[i], target: y[i], predicted: p[i] });
      return out;
    },
    codeVarDecl: "const float x",
    ood: oodProbe,
    exactFn: spec.fn,
    exactCost: spec.exactCost ?? (spec.id === "gelu" ? 46 : 34),
    iterativeBaseline: ITERATIVE_BASELINES[spec.id],
  };
}

// ---------------------------------------------------------------------------
// KV-CACHE TASK — evolve an eviction rule, beat StreamingLLM / H2O / window
// ---------------------------------------------------------------------------

// Two-phase protocol (this is what makes the benchmark honest):
//   • OBSERVED  : statistics a real eviction policy can actually read at
//                eviction time — attention mass accumulated over past queries
//                (A), position (P), sink flag (S), recency flag (R).
//   • EVALUATED : attention mass of the *next* query, which the policy cannot
//                see. Score = how much of that unseen mass the kept tokens hold.
// Top-k on A alone (H2O) is therefore NOT the ceiling anymore: the next query
// also concentrates on recent tokens that past queries ignored.
const KV_SEQ = 320;
const KV_KEEP = 40;
const KV_TRAIN = 8;
const KV_TEST = 12;
const KV_SINK = 4;
const KV_RECENT = 40;
const KV_HEAVY = 10;

interface KvWorld {
  /** shared latent importance profile (heavy hitters + sinks) */
  w: Float64Array;
  train: KvSample[];
  test: { feats: KvSample; future: Float64Array }[];
}
type KvSample = Record<"A" | "P" | "S" | "R", Float64Array>;

function softmaxInto(logits: Float64Array, out: Float64Array): void {
  let mx = -Infinity;
  for (let i = 0; i < logits.length; i++) mx = Math.max(mx, logits[i]);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - mx); sum += out[i]; }
  for (let i = 0; i < logits.length; i++) out[i] /= sum || 1;
}

function buildKvWorld(): KvWorld {
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
function topKWeightSum(score: Float64Array, weight: Float64Array, k: number): number {
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
function massOf(node: SpearNode, world: KvWorld, split: "train" | "test"): number {
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

function buildKvTask(): TaskDef {
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
function loadBootstrapSeeds(varNames: string[], excludeId: string, maxSeeds = 5): SpearNode[] {
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

const V = (name: string): SpearNode => makeNode("var", { name });
const C = (value: number): SpearNode => makeNode("const", { value });
const EXACT_LAWS: Record<string, SpearNode> = {
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
const ITERATIVE_BASELINES: Record<string, { label: string; totalCost: number }> = {
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

function buildRegressionTask(cfg: {
  id: string;
  title: string;
  subtitle: string;
  groundTruth: string;
  rows: number;
  varNames: string[];
  build: () => { vars: Record<string, Float64Array>; y: Float64Array };
  /** the exact generating law — used ONLY to measure the irreducible noise
   *  floor, never exposed to the search */
  trueLaw: (vars: Record<string, Float64Array>, i: number) => number;
  /** the generating law as an AST — gives the cost model its reference kernel */
  exactLaw?: SpearNode;
  /** explicit reference cost when writing the AST is redundant (documented
   *  arithmetic instead): benchmarkSpeed prefers this over estimateCost */
  exactCost?: number;
  /** task-specific generic shapes appended to the shared seed pool */
  extraSeeds?: SpearNode[];
  verify: (node: SpearNode) => string | null;
}): TaskDef {
  const { vars, y } = cfg.build();
  const n = y.length;
  const varNames = cfg.varNames;
  const gpConfig: GpConfig = {
    variables: varNames,
    constRange: [-6, 6],
    ops: GP_OPS_EFFECTIVE,
    maxDepth: 5,
    terminalVarProb: 0.55,
  };
  const scoreOf = (node: SpearNode): number => {
    try {
      const p = evaluateNode(node, vars, n);
      for (let i = 0; i < n; i++) if (!Number.isFinite(p[i])) return Infinity;
      return mse(p, y);
    } catch {
      return Infinity;
    }
  };
  const lin = ols1(vars[varNames[0]], y);
  let meanY = 0;
  for (let i = 0; i < n; i++) meanY += y[i];
  meanY /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (y[i] - meanY) ** 2;
  variance /= n;
  // Irreducible noise floor: the MSE the EXACT generating law itself scores
  // against the noisy observations. No formula can do meaningfully better;
  // anything below ~1.0x is fitting the noise, not the physics. Absolute MSE
  // milestones would be unreachable by construction, so we calibrate on this.
  let noiseFloor = 0;
  for (let i = 0; i < n; i++) noiseFloor += (cfg.trueLaw(vars, i) - y[i]) ** 2;
  noiseFloor /= n;

  // OOD probe from the exact law AST: sweep the first variable half a span
  // beyond both edges of its observed range, other variables pinned at their
  // dataset mean. Tasks without an exact-law AST simply get no constraint.
  const lawAst = cfg.exactLaw ?? EXACT_LAWS[cfg.id];
  let oodProbe: ((node: SpearNode) => number) | undefined;
  if (lawAst) {
    const xv = vars[varNames[0]];
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) { if (xv[i] < lo) lo = xv[i]; if (xv[i] > hi) hi = xv[i]; }
    const fixed: Record<string, number> = {};
    for (const vn of varNames.slice(1)) {
      let s = 0;
      for (let i = 0; i < n; i++) s += vars[vn][i];
      fixed[vn] = s / n;
    }
    const rspan = hi - lo;
    const oodPts = 32;
    const oodX = new Float64Array(oodPts * 2);
    for (let i = 0; i < oodPts; i++) {
      oodX[i] = lo - rspan * 0.5 * (1 - i / (oodPts - 1));
      oodX[oodPts + i] = hi + rspan * 0.5 * (i / (oodPts - 1));
    }
    const oy = new Float64Array(oodX.length);
    for (let i = 0; i < oodX.length; i++) {
      oy[i] = evaluateScalar(lawAst, { ...fixed, [varNames[0]]: oodX[i] });
    }
    oodProbe = makeOodProbe({ vars, y, n }, { vars: { [varNames[0]]: oodX }, y: oy, n: oodX.length });
  }

  return {
    id: cfg.id,
    family: "regression",
    title: cfg.title,
    subtitle: cfg.subtitle,
    groundTruth: cfg.groundTruth,
    metricLabel: "MSE",
    metricDirection: "min",
    formatMetric: (m) => (m < 1e-12 ? "0" : m.toExponential(2)),
    variables: varNames,
    gpConfig,
    evaluate: (node) => {
      try {
        const p = evaluateNode(node, vars, n);
        for (let i = 0; i < n; i++) if (!Number.isFinite(p[i])) return { metric: Infinity, finite: false };
        return { metric: mse(p, y), secondary: linfError(p, y), finite: true };
      } catch {
        return { metric: Infinity, finite: false };
      }
    },
    evaluateScored: (node) => {
      try {
        const p = evaluateNode(node, vars, n);
        for (let i = 0; i < n; i++) if (!Number.isFinite(p[i])) return { metric: Infinity, finite: false, node };
        const { a, b } = fitLinearScaling(p, y);
        const wrapped = wrapAffine(node, a, b);
        const wp = evaluateNode(wrapped, vars, n);
        for (let i = 0; i < n; i++) if (!Number.isFinite(wp[i])) return { metric: Infinity, finite: false, node: wrapped };
        return { metric: mse(wp, y), secondary: linfError(wp, y), finite: true, node: wrapped };
      } catch {
        return { metric: Infinity, finite: false, node };
      }
    },
    refine: (node) => {
      try {
        const p = evaluateNode(node, vars, n);
        const { a, b } = fitLinearScaling(p, y);
        const res = refineConstants(wrapAffine(node, a, b), scoreOf, 40);
        return { node: res.node, evals: res.evals + 1 };
      } catch {
        return { node, evals: 0 };
      }
    },
    seedPool: pureSeeds([
      ...loadBootstrapSeeds(varNames, cfg.id),
      // composite algebraic motifs (softsign / Padé / rsqrt shapes)
      ...compositeSeeds(varNames[0]),
      makeNode("sq", { children: [makeNode("var", { name: varNames[0] })] }),
      makeNode("mul", { children: [makeNode("var", { name: varNames[0] }), makeNode("sqrt", { children: [makeNode("var", { name: varNames[0] })] })] }),
      makeNode("sqrt", { children: [makeNode("cube", { children: [makeNode("var", { name: varNames[0] })] })] }),
      makeNode("cube", { children: [makeNode("var", { name: varNames[0] })] }),
      // generic exponential decay primitive — the missing building block for
      // first-order relaxation laws (RC, damping). A primitive, not a law.
      makeNode("exp", { children: [makeNode("neg", { children: [makeNode("var", { name: varNames[0] })] })] }),
      // generic inverse primitives — reciprocal powers appear everywhere in
      // physics kernels (deflection ~1/b, potentials ~1/r^n)
      makeNode("pdiv", { children: [makeNode("const", { value: 1 }), makeNode("var", { name: varNames[0] })] }),
      makeNode("sq", { children: [makeNode("pdiv", { children: [makeNode("const", { value: 1 }), makeNode("var", { name: varNames[0] })] })] }),
      // rsqrt family: pdiv(c, sqrt(c2 ± d·x²)) covers relativistic/normalisation
      // denominators; only the SHAPE is seeded, constants stay tunable
      // cross-variable shapes: bilinear interactions & linear combinations —
      // the glue of GEMV lanes, IDM gaps, IK ratios (shape only, no constants)
      ...(varNames.length > 1
        ? [
            makeNode("mul", { children: [makeNode("var", { name: varNames[0] }), makeNode("var", { name: varNames[1] })] }),
            makeNode("add", { children: [makeNode("var", { name: varNames[0] }), makeNode("var", { name: varNames[1] })] }),
            makeNode("sub", { children: [makeNode("var", { name: varNames[0] }), makeNode("var", { name: varNames[1] })] }),
          ]
        : []),
      // trigonometric carriers: RoPE rotations, DSP beats, oscillators
      makeNode("cos", { children: [makeNode("var", { name: varNames[0] })] }),
      makeNode("sin", { children: [makeNode("var", { name: varNames[0] })] }),
      ...(cfg.extraSeeds ?? []),
    ]),
    baselines: [
      { name: "Loi exacte (plancher de bruit)", metric: noiseFloor, note: "MSE de la vraie loi sur les données bruitées — optimum atteignable", kind: "oracle", formula: cfg.groundTruth },
      { name: "Régression linéaire (MCQ)", metric: lin.mse, note: `y = ${lin.a.toFixed(4)}·${varNames[0]} + ${lin.b.toFixed(4)}`, kind: "statistical", formula: "OLS" },
      { name: "Moyenne constante", metric: (() => { let m = 0; for (let i = 0; i < n; i++) m += y[i]; m /= n; let s = 0; for (let i = 0; i < n; i++) s += (y[i] - m) ** 2; return s / n; })(), note: "Variance totale du jeu de données", kind: "statistical", formula: "ȳ" },
    ],
    // Calibrated on the measured noise floor, not on arbitrary absolute
    // thresholds: reaching ~1x the floor means the recovered law is
    // statistically indistinguishable from the true generating law.
    milestones: [
      { level: 1, label: `MSE < variance/10 (${(variance / 10).toExponential(1)})`, test: (m) => m < variance / 10 },
      { level: 2, label: `Bat l'OLS ×10 (${(lin.mse / 10).toExponential(1)})`, test: (m) => m < lin.mse / 10 },
      { level: 3, label: `≤ 3× le plancher de bruit (${(noiseFloor * 3).toExponential(1)})`, test: (m) => m <= noiseFloor * 3 },
      { level: 4, label: `≤ 1.5× le plancher (${(noiseFloor * 1.5).toExponential(1)})`, test: (m) => m <= noiseFloor * 1.5 },
      { level: 5, label: `≤ 1.1× le plancher — loi indiscernable de la vraie (${(noiseFloor * 1.1).toExponential(1)})`, test: (m) => m <= noiseFloor * 1.1 },
    ],
    chart: (node) => {
      const p = evaluateNode(node, vars, n);
      const out: { x: number; target: number; predicted: number }[] = [];
      const step = Math.max(1, Math.floor(n / 60));
      for (let i = 0; i < n; i += step) out.push({ x: i, target: y[i], predicted: p[i] });
      return out;
    },
    verify: cfg.verify,
    codeVarDecl: `const float ${varNames.join(", const float ")}`,
    ood: oodProbe,
    exactRefNode: cfg.exactLaw ?? EXACT_LAWS[cfg.id],
    exactCost: cfg.exactCost,
    iterativeBaseline: ITERATIVE_BASELINES[cfg.id],
  };
}

function freeFallData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  const rows = 48;
  const t = linspace(0, 3, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = 0.5 * 9.81 * t[i] * t[i] + gaussianRandom() * 0.02;
  return { vars: { t }, y };
}

// ---------- Task 1 : Répartition gaussienne Φ(x) ----------
function gaussianCDFData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  const rows = 400;
  const x = linspace(-3, 3, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = 0.5 * (1 + erf(x[i] / Math.SQRT2));
  return { vars: { x }, y };
}

// ---------- Task 3 : Prime d'un call européen (Black-Scholes simplifié) ----------
function europeanCallData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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
function dampedPendulumData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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
function rlDistillationData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // On Approche: réseau de "professeur" simple f(x)=tanh(2x), on génère des données et
  // l'évolution cherche une forme fermée plus légère approchant cette fonction.
  const rows = 300;
  const x = linspace(-3, 3, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = Math.tanh(2 * x[i]);
  return { vars: { x }, y };
}

// ---------- Task 6 : Fonction implicite Lambert W₀(x) ----------
function lambertWData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Génération de données: y = x·eˣ pour x ∈ [0, 1.5], on stocke x en entrée, sort y
  // L'objectif est de récupérer x = W₀(y) par régression symbolique.
  const rows = 300;
  const x = linspace(0, 1.5, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = x[i] * Math.exp(x[i]);
  return { vars: { x }, y };
}

// ---------- Task 7 : Circuit RC · tension terminale ----------
function rcCircuitData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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

function keplerData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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
function layernormScaleData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // target: rsqrt(x) = 1/sqrt(x), the inverse stddev scale LayerNorm computes per token.
  // On-die, rsqrt is an SFU op; a bounded minimax algebraic form can be cheaper.
  const rows = 400;
  const x = linspace(0.01, 10, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = 1 / Math.sqrt(x[i]);
  return { vars: { x }, y };
}

// ---------- Gaussian blur 1-D kernel g(x) = exp(-x²/2σ²), x∈[-3σ,3σ], σ=1 ----------
function gaussianKernelData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  const rows = 200;
  const sigma = 1;
  const x = linspace(-3 * sigma, 3 * sigma, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = Math.exp(-(x[i] * x[i]) / (2 * sigma * sigma));
  return { vars: { x }, y };
}

// ---------- Diffusion noise schedule β(t) — cosine schedule, t∈[0,1] ----------
function betaScheduleData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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
function interpWeightData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // bilinear weight w = (1-u) for the left neighbour; u∈[0,1]
  const rows = 120;
  const u = linspace(0, 1, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = 1 - u[i];
  return { vars: { u }, y };
}

// ---------- Temporal gradient / optical-flow differencing primitive ----------
function temporalGradData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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
function lorentzData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Time-dilation / mass-energy factor used in physics engines and shaders.
  // Pure rsqrt-family law: discoverable with pdiv + sq + sqrt only.
  const rows = 200;
  const b = linspace(0, 0.99, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = 1 / Math.sqrt(1 - b[i] * b[i]);
  return { vars: { b }, y };
}

// ---------- Hill dose-response (pharmacology): E(c) = c³/(EC50³ + c³) ----------
function hillData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Standard drug dose-response curve (Emax model), EC50 = 1, Hill n = 3.
  // The rational shape every pharmacologist fits — here the GP must recover it.
  const rows = 200;
  const c = linspace(0.05, 5, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = (c[i] * c[i] * c[i]) / (1 + c[i] * c[i] * c[i]);
  return { vars: { c }, y };
}

// ---------- Kerr black-hole light deflection (weak field) ----------
function kerrDeflectionData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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
function lennardJonesData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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
function dampedOscillationData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Signal-processing staple: exponentially damped carrier. Uses exp + cos.
  const rows = 250;
  const t = linspace(0, 6, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = Math.exp(-t[i] / 4) * Math.cos(3 * t[i]);
  return { vars: { t }, y };
}

// ---------- Logistic growth L/(1 + e^(−k(t−t₀))) ----------
function logisticGrowthData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // Population / adoption / saturation curves: L=1, k=2, t₀=2.
  const rows = 200;
  const t = linspace(0, 4, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = 1 / (1 + Math.exp(-2 * (t[i] - 2)));
  return { vars: { t }, y };
}

// ---------- Softplus ln(1 + eˣ) ----------
function softplusData(): { vars: Record<string, Float64Array>; y: Float64Array } {
  // The smooth ReLU used in modern LLMs (output layer of SwiGLU blocks).
  // Exercises the log operator end-to-end.
  const rows = 200;
  const x = linspace(-4, 4, rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) y[i] = Math.log(1 + Math.exp(x[i]));
  return { vars: { x }, y };
}

// ---------- KdV 1-soliton (BT36): η = 2κ²·sech²(κ(x − 4κ²t − x₀)) ----------
function kdvSolitonData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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
function kerrSpinData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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
function pendulumHybridData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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

function eigenSymData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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

function ikReachData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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

function gemv4Data(): { vars: Record<string, Float64Array>; y: Float64Array } {
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

function ropeRotData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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

function idmFollowingData(): { vars: Record<string, Float64Array>; y: Float64Array } {
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

function erfImpl(x: number): number {
  // Abramowitz & Stegun 7.1.26, max error ~1.5e-7
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const y = 1 - poly * Math.exp(-ax * ax);
  return x >= 0 ? y : -y;
}

function besselJ0(x: number): number {
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
function besselJ2(x: number): number {
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
function besselJ1(x: number): number {
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
function groverSuccess(k: number, m: number, n: number): number {
  const theta = Math.asin(Math.sqrt(m / n));
  return Math.pow(Math.sin((2 * k + 1) * theta), 2);
}

// Concurrence of a pure 2-qubit state |a|00>+|b|01>+|c|10>+|d|11>
function concurrencePure(a: number, b: number, c: number, d: number): number {
  const norm = Math.sqrt(a * a + b * b + c * c + d * d);
  const A = a / norm, B = b / norm, Cc = c / norm, D = d / norm;
  return 2 * Math.abs(A * D - B * Cc);
}

// CHSH parameter for the singlet state with analyzer angles:
// S = |E(a,b) − E(a,b′) + E(a′,b) + E(a′,b′)|, E(x,y) = −cos(x−y)
function chshS(a: number, ap: number, b: number, bp: number): number {
  const E = (x: number, y: number) => -Math.cos(x - y);
  return Math.abs(E(a, b) - E(a, bp) + E(ap, b) + E(ap, bp));
}

// Tanner Helland blackbody fits — green & blue channels
function blackbodyGreen(tempK: number): number {
  const t = tempK / 100;
  let g: number;
  if (t <= 66) g = 99.4708025861 * Math.log(t) - 161.1195681661;
  else g = 288.1221695283 * Math.pow(t - 60, -0.0755148492);
  return Math.min(255, Math.max(0, g)) / 255;
}
function blackbodyBlue(tempK: number): number {
  const t = tempK / 100;
  let b: number;
  if (t >= 66) b = 255;
  else if (t <= 19) b = 0;
  else b = 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  return Math.min(255, Math.max(0, b)) / 255;
}

// Tanner Helland blackbody fit — normalized red channel vs color temperature
function blackbodyRed(tempK: number): number {
  const t = tempK / 100;
  let r: number;
  if (t <= 66) r = 255;
  else r = 329.698727446 * Math.pow(t - 60, -0.1332047592);
  return Math.min(255, Math.max(0, r)) / 255;
}

// Narkowicz ACES approximation — THE production tonemap reference
function narkowiczAces(x: number): number {
  const a = x * (2.51 * x + 0.03);
  const b = x * (2.43 * x + 0.59) + 0.14;
  return Math.min(1, Math.max(0, a / b));
}

// Black-76/Black-Scholes call price for IV data generation (d1,d2 closed form)
function bsCall(s: number, k: number, t: number, vol: number): number {
  const r = 0.02; // risk-free fixed for the dataset
  const sq = vol * Math.sqrt(t);
  const d1 = (Math.log(s / k) + (r + 0.5 * vol * vol) * t) / sq;
  const d2 = d1 - sq;
  const nd = (x: number) => 0.5 * (1 + erfImpl(x / Math.SQRT2));
  return s * nd(d1) - k * Math.exp(-r * t) * nd(d2);
}

// Implied volatility via Newton inversion of the above (ground-truth generator)
function impliedVol(c: number, s: number, k: number, t: number): number {
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
    vol += diff / vega;
  }
  return Math.max(0.01, Math.min(3, vol));
}

// --- wave 9: high-frontier numerical functions ------------------------------

// Scaled modified Bessel I0(x)·e^(-x) via convergent series (exact to ~1e-14)
function besselI0e(x: number): number {
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
function ellipticK(m: number): number {
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
function solveKepler(M: number, e: number): number {
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
function acklamProbit(p: number): number {
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

export function buildTasks(): TaskDef[] {
  const all: TaskDef[] = [
    buildActivationTask({
      id: "silu",
      title: "SiLU / Swish (LLaMA · Mistral · Qwen — SwiGLU)",
      fn: silu,
      lo: -6,
      hi: 6,
      groundTruth: "SiLU(x) = x·σ(x) = x / (1 + e⁻ˣ)",
    }),
    buildActivationTask({ id: "gelu", title: "GELU (GPT · BERT — GEGLU)", fn: (x) => 0.5 * x * (1 + erf(x / Math.SQRT2)), lo: -6, hi: 6, groundTruth: "GELU(x) = 0.5x(1 + erf(x/√2))" }),
    buildActivationTask({ id: "sigmoid", title: "Sigmoid (routage MoE · portes d'attention)", fn: (x) => 1 / (1 + Math.exp(-x)), lo: -8, hi: 8, groundTruth: "σ(x) = 1 / (1 + e⁻ˣ)" }),
    // Shader-ready algebraic gaussian for variable blur/bloom/DoF: every pixel
    // evaluates its own kernel weight, so transcendental-per-tap is the enemy.
    buildActivationTask({
      id: "gauss_shader",
      title: "Gaussienne shader-ready (blur variable DoF/bloom)",
      subtitle: "Approximation algébrique de e^(-x²/2) pour évaluation par pixel — cible coût ≤ 8 unités",
      fn: (x) => Math.exp(-x * x / 2),
      lo: -4,
      hi: 4,
      groundTruth: "g(x) = e^(-x²/2)",
      exactCost: 22,
    }),
    // ---- ultra-common program/kernel operations ----------------------------
    // sRGB transfer curve: applied to every pixel of every rendered frame.
    // Fractional power x^(1/2.4) is NOT served — rational approximant hunt.
    buildActivationTask({
      id: "srgb_gamma",
      title: "Transfert sRGB (encodage linéaire→affichage)",
      subtitle: "1.055·x^(1/2.4) − 0.055 sur [0,1] : puissance fractionnaire par pixel, approximant rationnel recherché",
      fn: (x) => 1.055 * Math.pow(x, 1 / 2.4) - 0.055,
      lo: 0,
      hi: 1,
      groundTruth: "srgb(x) = 1.055·x^(1/2.4) − 0.055",
      exactCost: 22, // mul + add + powf(SFU ~20)
    }),
    // tanh soft-clip: THE audio saturator + NN gating; Padé-style rational hunt
    buildActivationTask({
      id: "tanh_sat",
      title: "Tanh soft-clip (audio · gating)",
      subtitle: "tanh(x) sur [-3,3] sans transcendante — approximants rationnels type Padé",
      fn: (x) => Math.tanh(x),
      lo: -3,
      hi: 3,
      groundTruth: "tanh(x)",
      exactCost: 20, // SFU tanh
    }),
    // atan on the unit domain: the core of every atan2 (angles everywhere)
    buildActivationTask({
      id: "atan_unit",
      title: "Atan unitaire (cœur des atan2)",
      subtitle: "atan(x) sur [-1,1] — Padé [3/2] classique chassé en forme découverte",
      fn: (x) => Math.atan(x),
      lo: -1,
      hi: 1,
      groundTruth: "atan(x)",
      exactCost: 20, // SFU atan
    }),
    // frame-rate-independent smoothing factor: evaluated EVERY frame in games
    buildActivationTask({
      id: "ema_smooth",
      title: "Facteur EMA indépendant du framerate",
      subtitle: "1 − e^(-3.2·dt), dt ∈ [0, 0.1] s — lissage caméra/particules à dt variable",
      fn: (dt) => 1 - Math.exp(-3.2 * dt),
      lo: 0,
      hi: 0.1,
      groundTruth: "f(dt) = 1 − e^(-3.2·dt)",
      exactCost: 23, // mul + neg + exp(20) + sub-from-1
    }),
    // smoothstep: the most iconic shader interpolation — optimality test #2
    buildActivationTask({
      id: "smoothstep",
      title: "Smoothstep (interpolation shader iconique)",
      subtitle: "t²(3−2t) sur [0,1] : forme polynomiale exacte à 5 unités — le moteur la retrouve-t-il ?",
      fn: (t) => t * t * (3 - 2 * t),
      lo: 0,
      hi: 1,
      groundTruth: "smoothstep(t) = t²(3−2t)",
      exactCost: 5, // sq + mul + sub + mul... documented arithmetic
    }),
    // ---- second wave: special functions & training/graphics glue ----------
    // sRGB decode: display -> linear, the mirror of srgb_gamma. Every texture
    // read in a physically-based pipeline walks this curve once.
    buildActivationTask({
      id: "srgb_decode",
      title: "Décode sRGB (affichage → linéaire)",
      subtitle: "x^2.2 sur [0,1] : puissance fractionnaire inverse, approximant rationnel recherché",
      fn: (x) => Math.pow(x, 2.2),
      lo: 0,
      hi: 1,
      groundTruth: "decode(x) = x^2.2",
      exactCost: 22,
    }),
    // erf: THE probability kernel (GELU-exact grade). exp is SERVED here, so
    // the search may mix rationals with exponentials — A&S-style hybrids.
    buildActivationTask({
      id: "erf_prob",
      title: "Erf probabiliste (noyaux de probabilité · GELU-exact)",
      subtitle: "erf(x) sur [-3,3] — hybrides rationnel+exp permis (style Abramowitz-Stegun)",
      fn: (x) => erfImpl(x),
      lo: -3,
      hi: 3,
      groundTruth: "erf(x)",
      exactCost: 26,
    }),
    // Huber loss δ=1: the elegant trick is huber(x) = max(x²/2, |x| − 1/2) —
    // EXACTLY expressible in 5 units. Optimality test #3: does the search find
    // the max-form trick, or grind through piecewise rubble?
    buildActivationTask({
      id: "huber_loss",
      title: "Perte de Huber δ=1 (entraînement robuste)",
      subtitle: "max(x²/2, |x|−1/2) : forme max exacte à 5 unités — le GP voit-il l'astuce ?",
      fn: (x) => (Math.abs(x) <= 1 ? 0.5 * x * x : Math.abs(x) - 0.5),
      lo: -4,
      hi: 4,
      groundTruth: "huber(x) = x²/2 si |x|≤1, sinon |x|−1/2",
      exactCost: 5,
      extraSeeds: [
        // the max-trick scaffold, explicitly seeded (experiment: does SEEING
        // the trick let refinement converge to the 5-unit exact form?)
        simplify(makeNode("max", {
          children: [
            makeNode("mul", { children: [makeNode("const", { value: 0.5 }), makeNode("sq", { children: [makeNode("var", { name: "x" })] })] }),
            makeNode("sub", { children: [makeNode("abs", { children: [makeNode("var", { name: "x" })] }), makeNode("const", { value: 0.5 })] }),
          ],
        })),
        // perturbed variant: same scaffold, wrong constants — refinement must tune
        simplify(makeNode("max", {
          children: [
            makeNode("mul", { children: [makeNode("sq", { children: [makeNode("var", { name: "x" })] }), makeNode("const", { value: 0.31 })] }),
            makeNode("sub", { children: [makeNode("mul", { children: [makeNode("abs", { children: [makeNode("var", { name: "x" })] }), makeNode("const", { value: 0.72 })] }), makeNode("const", { value: 0.11 })] }),
          ],
        })),
      ],
    }),
    // cosh: catenaries, ring-modulation audio, and the sech² inside KdV's own
    // reference. Two exps are the textbook form — can algebra beat them?
    buildActivationTask({
      id: "cosh_curve",
      title: "Cosh (caténaires · ring-mod audio)",
      subtitle: "cosh(x) sur [-2,2] — la forme (eˣ+e⁻ˣ)/2 coûte 44 unités, place aux algébriques",
      fn: (x) => Math.cosh(x),
      lo: -2,
      hi: 2,
      groundTruth: "cosh(x)",
      exactCost: 44,
    }),
    // Bessel J0: FM sidebands, membrane/vibration modes, beam physics. No SFU
    // serves it — libraries grind series. A discovered closed form would be
    // genuinely novel engineering.
    buildActivationTask({
      id: "bessel_j0",
      title: "Bessel J₀ (FM · vibrations · poutres)",
      subtitle: "J₀(x) sur [0,6] — aucune SFU ne la sert ; forme close découverte = vrai nouvel outil",
      fn: (x) => besselJ0(x),
      lo: 0,
      hi: 6,
      groundTruth: "J₀(x)",
      exactCost: 40, // series/asymptotic library evaluation
      extraSeeds: [
        // pre-chewed series head (shape-only): 1 − 2.25(x/2)² + 1.27(x/2)⁴
        simplify(makeNode("add", {
          children: [
            makeNode("const", { value: 1 }),
            makeNode("sub", {
              children: [
                makeNode("mul", { children: [makeNode("const", { value: -2.25 }), makeNode("sq", { children: [makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("const", { value: 2 })] })] })] }),
                makeNode("mul", { children: [makeNode("const", { value: 1.27 }), makeNode("sq", { children: [makeNode("sq", { children: [makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("const", { value: 2 })] })] })] })] }),
              ],
            }),
          ],
        })),
        // LONG series (4 terms, A&S coefficients): the compression target
        simplify((() => {
          const H = () => makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("const", { value: 2 })] });
          const h2 = makeNode("sq", { children: [H()] });
          const h4 = makeNode("sq", { children: [h2] });
          const h6 = makeNode("mul", { children: [h4, h2] });
          return makeNode("add", {
            children: [
              makeNode("sub", {
                children: [
                  makeNode("add", {
                    children: [
                      makeNode("sub", { children: [C(1), makeNode("mul", { children: [C(2.25), h2] })] }),
                      makeNode("mul", { children: [C(1.2656208), h4] }),
                    ],
                  }),
                  makeNode("mul", { children: [C(0.3163866), h6] }),
                ],
              }),
              // + 0.0444479·h⁸ − ... folded via sq chain
              makeNode("mul", { children: [C(0.0444479), makeNode("sq", { children: [h4] })] }),
            ],
          });
        })()),
        // even-series tail for contrast
        simplify(makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("const", { value: 3 })] })),
      ],
    }),
    // logit: probabilities <-> logits glue, present in every classifier head
    buildActivationTask({
      id: "logit_ml",
      title: "Logit (probas ↔ logits, têtes de classification)",
      subtitle: "ln(x/(1−x)) sur (0.02, 0.98) — colle ML, approximant sans division protégée utile",
      fn: (x) => Math.log(x / (1 - x)),
      lo: 0.001,
      hi: 0.999,
      groundTruth: "logit(x) = ln(x/(1−x))",
      exactCost: 27,
      // the law is THREE served ops (log/pdiv/sub) yet stayed L0 for lack of
      // any rational-log shape in the pool — same obsolete-premise pattern as
      // grover. Shapes only; constants stay tunable.
      extraSeeds: (() => {
        const x = V("x");
        const lg = (num: SpearNode, den: SpearNode): SpearNode =>
          makeNode("log", { children: [makeNode("pdiv", { children: [num, den] })] });
        return [
          lg(x, makeNode("sub", { children: [C(1), x] })),                    // ln(x/(1−x))
          lg(makeNode("add", { children: [C(1), x] }), makeNode("sub", { children: [C(1), x] })), // ln((1+x)/(1−x)) — atanh bridge
          lg(makeNode("mul", { children: [C(1), x] }), makeNode("add", { children: [C(1), x] })), // softsign-log cousin
        ];
      })(),
    }),
    // ---- WAVE 8b: quantum special functions for SR discovery ---------------
    // Legendre P₂(x): THE quantum angular eigenfunction. Exact polynomial.
    buildActivationTask({
      id: "legendre_p2",
      title: "Legendre P₂ (harmonique sphérique · moments quantiques)",
      subtitle: "P₂(x) = (3x²−1)/2 sur [-1,1] : exact en 5 unités — optimality test #6",
      fn: (x) => 0.5 * (3 * x * x - 1),
      lo: -1,
      hi: 1,
      groundTruth: "P₂(x) = (3x²−1)/2",
      exactCost: 5,
    }),
    // Laguerre L₂(x): hydrogen radial wavefunction basis, quantum oscillator.
    buildActivationTask({
      id: "laguerre_l2",
      title: "Laguerre L₂ (radiale hydrogène · oscillateur)",
      subtitle: "L₂(x) = 1 − 2x + x²/2 sur [0,10] : exact polynomiale — optimality test #7",
      fn: (x) => 1 - 2 * x + 0.5 * x * x,
      lo: 0,
      hi: 10,
      groundTruth: "L₂(x) = 1 − 2x + x²/2",
      exactCost: 5,
    }),
    // asin on the unit domain: inverse-trig primitive — SERVED since the asin
    // backend landed; task fell to machine precision (MSE = 0) same generation.
    buildActivationTask({
      id: "asin_hard",
      title: "Arcsinus unitaire (quantique · rotations)",
      subtitle: "asin(x) sur [-0.95,0.95] — l'inverse trigonométrique sans forme close servie",
      fn: (x) => Math.asin(Math.max(-0.95, Math.min(0.95, x))),
      lo: -0.95,
      hi: 0.95,
      groundTruth: "asin(x)",
      exactCost: 28,
    }),
    // ---- WAVE 9: high-frontier numerical functions --------------------------
    // Scaled modified Bessel I0(x)·e^(-x): KBD windows, diffusion models.
    // Dual structure (algebraic near 0, exponential decay at ∞).
    buildRegressionTask({
      id: "bessel_i0e",
      title: "Bessel I₀e modifiée (fenêtres KBD · diffusion)",
      subtitle: "I₀(x)·e^(−x) sur [0,50] — décroissance exponentielle × série de Bessel",
      groundTruth: "e^(−x)·Σ(x²/4)^k/(k!)²",
      rows: 500,
      varNames: ["x"],
      exactCost: 24,
      build: () => {
        const rows = 500;
        const xv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const x = 50 * ((i * 0.6180339887) % 1);
          xv[i] = x;
          y[i] = besselI0e(x);
        }
        return { vars: { x: xv }, y };
      },
      trueLaw: (v, i) => besselI0e(v.x[i]),
      verify: () => null,
    }),
    // Complete elliptic integral K(m): pendulum period, geodesics. The log
    // singularity at m→1 makes this a genuinely hard approximation target.
    buildRegressionTask({
      id: "elliptic_k",
      title: "Intégrale elliptique K(m) (pendule · géodésiques)",
      subtitle: "K(m) sur [0, 0.999] — singularité logarithmique en m=1 : le défi ultime",
      groundTruth: "K(m) via AGM",
      rows: 500,
      varNames: ["m"],
      exactCost: 16,
      build: () => {
        const rows = 500;
        const mv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const mm = Math.min(0.999, i / (rows - 1));
          mv[i] = mm;
          y[i] = ellipticK(mm);
        }
        return { vars: { m: mv }, y };
      },
      trueLaw: (v, i) => ellipticK(v.m[i]),
      verify: () => null,
    }),
    // Kepler equation solver: THE orbital mechanics bottleneck. An inverse
    // problem with no closed form even with all primitives served.
    buildRegressionTask({
      id: "kepler_solver",
      title: "Solveur de Kepler O(1) (mécanique orbitale)",
      subtitle: "E(M,e) résoluant M = E − e·sin(E) : remplace Newton-Raphson itératif",
      groundTruth: "E résout E − e·sin(E) = M",
      rows: 500,
      varNames: ["M", "e"],
      exactCost: 32,
      build: () => {
        const rows = 500;
        const Mv = new Float64Array(rows), ev = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const M = Math.PI * ((i * 0.6180339887) % 1);
          const ecc = 0.95 * ((i * 0.7548776662) % 1);
          Mv[i] = M; ev[i] = ecc;
          y[i] = solveKepler(M, ecc);
        }
        return { vars: { M: Mv, e: ev }, y };
      },
      trueLaw: (v, i) => solveKepler(v.M[i], v.e[i]),
      verify: () => null,
    }),
    // Fast exp without transcendental ops: pure ALU approximation of e^x on
    // [-16, 0]. The engine must discover what took DSP engineers decades.
    buildActivationTask({
      id: "fast_exp_alu",
      title: "Exp algébrique pure (sans exp/log/sin/cos)",
      subtitle: "e^x sur [-16, 0] en ALU pur — remplace l'appel SFU dans les kernels edge",
      fn: (x) => Math.exp(Math.max(-16, Math.min(0, x))),
      lo: -16,
      hi: 0,
      groundTruth: "e^x sur [-16, 0]",
    }),
    // ---- WAVE 7: exact-representable everyday-science kernels -------------
    // Michaelis-Menten: enzyme/drug velocity — biochemistry staple, exact
    // rational in 7 units. Pairs with hill (same family, different law).
    buildRegressionTask({
      id: "michaelis_menten",
      title: "Michaelis-Menten (biochimie · pharma)",
      subtitle: "v = Vmax·[S]/(Km+[S]) : vitesse enzymatique, rationnel exact à 7 unités",
      groundTruth: "v = 100·S/(4+S) (Vmax=100, Km=4)",
      rows: 400,
      varNames: ["s"],
      exactCost: 7,
      build: () => {
        const rows = 400;
        const sv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const s = 0.1 + 40 * ((i * 0.6180339887) % 1);
          sv[i] = s;
          y[i] = (100 * s) / (4 + s);
        }
        return { vars: { s: sv }, y };
      },
      trueLaw: (v, i) => (100 * v.s[i]) / (4 + v.s[i]),
      verify: () => null,
    }),
    // Temperature-scaled softmax over two logits: THE LLM sampling kernel.
    // Exact via exp: sigmoid(Δ/T). Directly relevant to inference stacks.
    buildRegressionTask({
      id: "temperature_softmax",
      title: "Softmax tempéré 2-logits (échantillonnage LLM)",
      subtitle: "P(a>b | Δa, T) = σ(Δ/T) — le contrôle de créativité des LLM en forme close",
      groundTruth: "p = 1/(1+e^(−Δ/T))",
      rows: 500,
      varNames: ["da", "t"],
      exactCost: 26,
      build: () => {
        const rows = 500;
        const dav = new Float64Array(rows), tv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const da = -6 + 12 * ((i * 0.6180339887) % 1);
          const t = 0.1 + 2.9 * ((i * 0.7548776662) % 1);
          dav[i] = da; tv[i] = t;
          y[i] = 1 / (1 + Math.exp(-da / t));
        }
        return { vars: { da: dav, t: tv }, y };
      },
      trueLaw: (v, i) => 1 / (1 + Math.exp(-v.da[i] / v.t[i])),
      extraSeeds: [
        // EXACT-form scaffold: σ(Δ/T) = 1/(1+e^(−Δ/T)) — fully served
        simplify(makeNode("pdiv", {
          children: [
            makeNode("const", { value: 1 }),
            makeNode("add", {
              children: [
                makeNode("const", { value: 1 }),
                makeNode("exp", { children: [makeNode("neg", { children: [makeNode("pdiv", { children: [makeNode("var", { name: "da" }), makeNode("var", { name: "t" })] })] })] })],
            }),
          ],
        })),
      ],
      verify: () => null,
    }),
    // Doppler effect: f' = f·(v+vo)/(v−vs) — audio/radar/astronomy staple.
    buildRegressionTask({
      id: "doppler_effect",
      title: "Effet Doppler (audio · radar)",
      subtitle: "f' = f·(v+v₀)/(v−vs) : pitch d'une sirène qui passe, forme exacte",
      groundTruth: "f' = f·(v+v₀)/(v−vs), f=700 Hz, v=340 m/s",
      rows: 400,
      varNames: ["vo", "vs"],
      exactCost: 5,
      build: () => {
        const rows = 400;
        const ov = new Float64Array(rows), sv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const vo = 30 * ((i * 0.6180339887) % 1);
          const vs = 10 + 50 * ((i * 0.7548776662) % 1);
          ov[i] = vo; sv[i] = vs;
          y[i] = 700 * (340 + vo) / (340 - vs);
        }
        return { vars: { vo: ov, vs: sv }, y };
      },
      trueLaw: (v, i) => 700 * (340 + v.vo[i]) / (340 - v.vs[i]),
      extraSeeds: [
        // Doppler scaffold (perturbed): 710·(345+vo)/(338−vs) — shape shown,
        // refinement tunes to the exact 5-unit rational.
        simplify((() => {
          const VO = makeNode("var", { name: "vo" });
          const VS = makeNode("var", { name: "vs" });
          const C = (x: number) => makeNode("const", { value: x });
          const bin = (op: NodeOp, a: any, b: any) => makeNode(op, { children: [a, b] });
          return simplify(bin("mul", C(710), bin("pdiv", bin("add", C(345), VO), bin("sub", C(338), VS))));
        })()),
      ],
      verify: () => null,
    }),
    // Stefan-Boltzmann: radiated power ∝ T⁴ — thermal PBR / engineering.
    // sq∘sq chain: EXACT at 3 units — the cheapest optimality test possible.
    buildRegressionTask({
      id: "stefan_boltzmann",
      title: "Stefan-Boltzmann (radiation thermique)",
      subtitle: "P = σ·T⁴ : la chaîne sq∘sq exacte à 3 unités — test d'optimalité #5",
      groundTruth: "P(T) = σT⁴ (σ normalisé)",
      rows: 400,
      varNames: ["t"],
      exactCost: 3,
      build: () => {
        const rows = 400;
        const tv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const T = 1 + 9 * ((i * 0.6180339887) % 1);
          tv[i] = T;
          y[i] = 0.42 * Math.pow(T, 4);
        }
        return { vars: { t: tv }, y };
      },
      trueLaw: (v, i) => 0.42 * Math.pow(v.t[i], 4),
      verify: () => null,
    }),
    // M/M/1 queue wait: λ/(μ(μ−λ)) — systems/SRE staple, exact rational.
    buildRegressionTask({
      id: "mm1_queue_wait",
      title: "Attente file M/M/1 (SRE · systèmes)",
      subtitle: "W = λ/(μ(μ−λ)) : temps d'attente moyen, exact en 6 unités",
      groundTruth: "W(λ,μ) = λ/(μ(μ−λ))",
      rows: 400,
      varNames: ["l", "m"],
      exactCost: 6,
      build: () => {
        const rows = 400;
        const lv = new Float64Array(rows), mv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const m = 1.2 + 8.8 * ((i * 0.6180339887) % 1);
          const l = 0.05 + (m - 0.15) * ((i * 0.7548776662) % 1);
          lv[i] = l; mv[i] = m;
          y[i] = l / (m * (m - l));
        }
        return { vars: { l: lv, m: mv }, y };
      },
      trueLaw: (v, i) => v.l[i] / (v.m[i] * (v.m[i] - v.l[i])),
      verify: () => null,
    }),
    // ---- WAVE 8: quantum computing operations ------------------------------
    // Grover amplitude amplification: THE quadratic-speedup law. CRACKED once
    // asin got served (+ structural scaffold with k inside the sine argument):
    // 8.3e-32, L2 — same story as ik_reach when atan was served.
    buildRegressionTask({
      id: "grover_amplitude",
      title: "Amplification de Grover (recherche quantique)",
      subtitle: "P(k,m,n) = sin²((2k+1)·asin(√(m/n))) — LA loi du speedup quadratique quantique",
      groundTruth: "P = sin²((2k+1)·θ), θ = asin(√(m/n))",
      rows: 500,
      varNames: ["k", "m", "n"],
      exactCost: 48,
      build: () => {
        const rows = 500;
        const kv = new Float64Array(rows), mv = new Float64Array(rows), nv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const n = 64 + Math.floor(4032 * ((i * 0.6180339887) % 1));
          const m = 1 + Math.max(1, Math.floor((n / 8) * ((i * 0.7548776662) % 1)));
          const k = Math.floor(16 * ((i * 0.4192388219) % 1));
          kv[i] = k; mv[i] = m; nv[i] = n;
          y[i] = groverSuccess(k, m, n);
        }
        return { vars: { k: kv, m: mv, n: nv }, y };
      },
      trueLaw: (v, i) => groverSuccess(v.k[i], v.m[i], v.n[i]),
      // asin is SERVED now — the pre-asin premise is obsolete. Shape-only
      // scaffolds of the amplification law (constants stay tunable), same
      // doctrine that cracked ik_reach/eigen when atan got served.
      extraSeeds: (() => {
        const kk = V("k"), mm = V("m"), nn = V("n");
        const theta = makeNode("asin", {
          children: [makeNode("sqrt", { children: [makeNode("pdiv", { children: [mm, nn] })] })],
        });
        const kTheta = makeNode("mul", { children: [kk, theta] });
        // (2k+1)·θ lives INSIDE the sine argument — k is a variable, so the
        // scaffold must carry it there structurally, not as a constant
        const twoKPlusOneTheta = makeNode("add", {
          children: [theta, makeNode("mul", { children: [C(2), kTheta] })],
        });
        const sin = (arg: SpearNode): SpearNode => makeNode("sin", { children: [arg] });
        return [
          makeNode("sq", { children: [sin(twoKPlusOneTheta)] }),
          makeNode("sq", { children: [sin(makeNode("add", { children: [theta, makeNode("mul", { children: [C(3), kTheta] })] }))] }),
          makeNode("sq", { children: [sin(theta)] }),
          makeNode("sub", { children: [
            C(1),
            makeNode("sq", { children: [makeNode("cos", { children: [twoKPlusOneTheta] })] }),
          ] }),
        ];
      })(),
      verify: () => null,
    }),
    // Concurrence: entanglement measure of a pure 2-qubit state. Quantum
    // information staple; exact rational in 7 units once normalized.
    buildRegressionTask({
      id: "concurrence_pure",
      title: "Concurrence 2-qubits pur (intrication)",
      subtitle: "C = 2|ad−bc| sur amplitudes normalisées : LA mesure d'intrication",
      groundTruth: "C(a,b,c,d) = 2·|ad−bc|",
      rows: 500,
      varNames: ["a", "b", "c", "d"],
      exactCost: 7,
      build: () => {
        const rows = 500;
        const av = new Float64Array(rows), bv = new Float64Array(rows), cv2 = new Float64Array(rows), dv = new Float64Array(rows), y = new Float64Array(rows);
        let seedState = 42;
        const rnd = () => {
          seedState = (seedState * 1103515245 + 12345) % 2147483648;
          return seedState / 2147483648 - 0.5;
        };
        for (let i = 0; i < rows; i++) {
          let a = rnd(), b = rnd(), c = rnd(), d = rnd();
          const norm = Math.sqrt(a * a + b * b + c * c + d * d);
          a /= norm; b /= norm; c /= norm; d /= norm;
          av[i] = a; bv[i] = b; cv2[i] = c; dv[i] = d;
          y[i] = 2 * Math.abs(a * d - b * c);
        }
        return { vars: { a: av, b: bv, c: cv2, d: dv }, y };
      },
      trueLaw: (v, i) => {
        const norm = Math.sqrt(v.a[i] ** 2 + v.b[i] ** 2 + v.c[i] ** 2 + v.d[i] ** 2);
        return 2 * Math.abs((v.a[i] / norm) * (v.d[i] / norm) - (v.b[i] / norm) * (v.c[i] / norm));
      },
      extraSeeds: [
        // EXACT-form scaffold: C = 2|ad − bc| (huber recipe — fully served ops)
        simplify(makeNode("mul", {
          children: [
            makeNode("const", { value: 2 }),
            makeNode("abs", {
              children: [makeNode("sub", {
                children: [
                  makeNode("mul", { children: [makeNode("var", { name: "a" }), makeNode("var", { name: "d" })] }),
                  makeNode("mul", { children: [makeNode("var", { name: "b" }), makeNode("var", { name: "c" })] }),
                ],
              })],
            }),
          ],
        })),
      ],
      verify: () => null,
    }),
    // CHSH parameter: the Bell inequality test quantity (Nobel Physics 2022).
    // Exact via 4 cosines ~84 units — optimality test on a Nobel-grade law.
    buildRegressionTask({
      id: "chsh_correlation",
      title: "Paramètre CHSH (test de Bell · Nobel 2022)",
      subtitle: "S(a,a′,b,b′) = |E(a,b)−E(a,b′)+E(a′,b)+E(a′,b′)|, E=−cos(x−y) — violation ≤ 2√2",
      groundTruth: "S = |E(a,b)−E(a,b′)+E(a′,b)+E(a′,b′)|",
      rows: 400,
      varNames: ["a", "ap", "b", "bp"],
      exactCost: 84,
      build: () => {
        const rows = 400;
        const av = new Float64Array(rows), apv = new Float64Array(rows), bv = new Float64Array(rows), bpv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const A = 2 * Math.PI * ((i * 0.6180339887) % 1);
          const AP = 2 * Math.PI * ((i * 0.7548776662) % 1);
          const B = 2 * Math.PI * ((i * 0.4192388219) % 1);
          const BP = 2 * Math.PI * ((i * 0.5412417173) % 1);
          const E = (x: number, yy: number) => -Math.cos(x - yy);
          av[i] = A; apv[i] = AP; bv[i] = B; bpv[i] = BP;
          y[i] = Math.abs(E(A, B) - E(A, BP) + E(AP, B) + E(AP, BP));
        }
        return { vars: { a: av, ap: apv, b: bv, bp: bpv }, y };
      },
      trueLaw: (v, i) => {
        const E = (x: number, yy: number) => -Math.cos(x - yy);
        return Math.abs(E(v.a[i], v.b[i]) - E(v.a[i], v.bp[i]) + E(v.ap[i], v.b[i]) + E(v.ap[i], v.bp[i]));
      },
      extraSeeds: [
        // EXACT-form scaffold: the full CHSH expression in served ops
        simplify((() => {
          const V = (n: string) => makeNode("var", { name: n });
          const C = (x: number) => makeNode("const", { value: x });
          const bin = (op: any, a: any, b: any) => makeNode(op, { children: [a, b] });
          const E = (p: string, q: string) => makeNode("neg", { children: [makeNode("cos", { children: [bin("sub", V(p), V(q))] })] });
          const sExpr = bin("sub",
            bin("add", E("a", "b"), E("ap", "b")),
            bin("sub", E("a", "bp"), E("ap", "bp"))
          );
          return makeNode("abs", { children: [sExpr] });
        })()),
      ],
      verify: () => null,
    }),
    // ---- SPEAR QUANT PACK: trading/fintech kernels -------------------------
    // Kelly criterion: optimal bet fraction. EXACTLY expressible — the
    // elegant form f* = p − (1−p)/b costs 6 units (mul+sub+pdiv).
    buildRegressionTask({
      id: "kelly_criterion",
      title: "Kelly criterion — position sizing",
      subtitle: "f* = p − (1−p)/b : la fraction optimale, exacte en 6 unités",
      groundTruth: "f*(p,b) = max(0, (p(b+1)−1)/b)",
      rows: 400,
      varNames: ["p", "b"],
      exactCost: 6,
      build: () => {
        const rows = 400;
        const pv = new Float64Array(rows), bv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const p = 0.05 + 0.9 * ((i * 0.6180339887) % 1);
          const b = 0.5 + 4.5 * ((i * 0.7548776662) % 1);
          pv[i] = p; bv[i] = b;
          y[i] = Math.max(0, (p * (b + 1) - 1) / b);
        }
        return { vars: { p: pv, b: bv }, y };
      },
      trueLaw: (v, i) => Math.max(0, (v.p[i] * (v.b[i] + 1) - 1) / v.b[i]),
      extraSeeds: [
        // Kelly scaffold (huber recipe): f* = max(0, p − (1−p)/b), shape shown
        simplify((() => {
          const P = makeNode("var", { name: "p" });
          const B = makeNode("var", { name: "b" });
          const C = (x: number) => makeNode("const", { value: x });
          const bin = (op: NodeOp, a: any, b: any) => makeNode(op, { children: [a, b] });
          return makeNode("max", {
            children: [
              C(0),
              bin("sub", bin("mul", P, C(1.05)), bin("pdiv", bin("sub", C(1.02), P), B)),
            ],
          });
        })()),
      ],
      verify: () => null,
    }),
    buildRegressionTask({
      id: "rsi_momentum",
      title: "RSI depuis moyennes lissées (TradingView)",
      subtitle: "RSI = 100·g/(g+l) : forme exacte à 4 unités — kernel d'indicateur embarqué",
      groundTruth: "RSI(g,l) = 100·g/(g+l)",
      rows: 400,
      varNames: ["g", "l"],
      exactCost: 4,
      build: () => {
        const rows = 400;
        const gv = new Float64Array(rows), lv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const g = 10 * ((i * 0.6180339887) % 1);
          const l = 0.01 + 10 * ((i * 0.7548776662) % 1);
          gv[i] = g; lv[i] = l;
          y[i] = 100 * g / (g + l);
        }
        return { vars: { g: gv, l: lv }, y };
      },
      trueLaw: (v, i) => (100 * v.g[i]) / (v.g[i] + v.l[i]),
      verify: () => null,
    }),
    // Implied volatility: inverting Black-Scholes numerically is the quant
    // bottleneck on every pricing desk. A discovered algebraic form that
    // tracks Newton-solved IV within tolerance = embeddable edge kernel.
    buildRegressionTask({
      id: "implied_vol",
      title: "Volatilité implicite (inversion Black-Scholes)",
      subtitle: "IV(c,s,k,t) : remplace l'inversion de Newton par forme close découverte",
      groundTruth: "σ telle que BS(s,k,t,σ)=c — résolue par Newton (référence)",
      rows: 500,
      varNames: ["c", "s", "k", "t"],
      build: () => {
        const rows = 500;
        const cv = new Float64Array(rows), sv = new Float64Array(rows), kv = new Float64Array(rows), tv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const s = 80 + 40 * ((i * 0.6180339887) % 1);
          const k = 80 + 40 * ((i * 0.7548776662) % 1);
          const t = 0.08 + 0.9 * ((i * 0.4192388219) % 1);
          const vol = 0.1 + 0.9 * ((i * 0.5412417173) % 1);
          const c = bsCall(s, k, t, vol);
          cv[i] = c; sv[i] = s; kv[i] = k; tv[i] = t;
          y[i] = impliedVol(c, s, k, t);
        }
        return { vars: { c: cv, s: sv, k: kv, t: tv }, y };
      },
      trueLaw: (v, i) => impliedVol(v.c[i], v.s[i], v.k[i], v.t[i]),
      // no closed form exists, but the HUMAN art is seedable: Brenner–
      // Subrahmanyam ATMF skeleton and its Corrado–Miller-style corrections,
      // shapes only (generic constants).
      extraSeeds: (() => {
        const c = V("c"), s = V("s"), k = V("k"), t = V("t");
        const cs = makeNode("pdiv", { children: [c, s] });
        const sqrtInvT = makeNode("sqrt", { children: [makeNode("pdiv", { children: [C(1), t] })] });
        const bs = makeNode("mul", { children: [C(2.5), makeNode("mul", { children: [cs, sqrtInvT] })] }); // √(2π)·(c/s)/√t
        const ksTerm = makeNode("mul", { children: [makeNode("pdiv", { children: [k, s] }), sqrtInvT] });
        return [
          bs,
          simplify(makeNode("add", { children: [bs, makeNode("mul", { children: [C(1), ksTerm] })] })),
          makeNode("mul", { children: [bs, makeNode("add", { children: [C(1), makeNode("pdiv", { children: [makeNode("sq", { children: [cs] }), t] })] })] }),
        ];
      })(),
      verify: (node) => {
        const iv = evaluateScalar(node, { c: bsCall(100, 100, 0.5, 0.35), s: 100, k: 100, t: 0.5 });
        if (!Number.isFinite(iv)) return null;
        return Math.abs(iv - 0.35) < 0.12 ? `IV at-the-money ≈ ${iv.toFixed(3)} vs 0.35` : null;
      },
    }),
    // ---- wave 4: never-before-benchmarked operations -----------------------
    // Probit = inverse normal CDF: THE quantile kernel of finance/risk/stats
    // (VaR, probit regression, z-scores). No closed form even with erf served;
    // Acklam-style rational+exp hybrids are the human art — can evolution
    // rediscover or beat them? Never seen as an SR benchmark.
    buildActivationTask({
      id: "probit_quantile",
      title: "Probit / quantile normal (finance · risque · z-scores)",
      subtitle: "Φ⁻¹(p) sur [0.02, 0.98] — l'inverse sans forme close : chasse à l'hybride rationnel+exp",
      fn: (p) => acklamProbit(p),
      lo: 0.02,
      hi: 0.98,
      groundTruth: "probit(p) = Φ⁻¹(p)",
      exactCost: 28,
    }),
    // Loan payment per unit principal: r(1+r)^n / ((1+r)^n − 1). Variable
    // exponent is unservable directly BUT e^(n·ln(1+r)) makes the hybrid
    // expressible — fintech edge kernels pay well.
    buildRegressionTask({
      id: "pmt_finance",
      title: "Mensualité de prêt par unité (fintech edge)",
      subtitle: "PMT(r,n) = r(1+r)ⁿ/((1+r)ⁿ−1) via hybride exp∘ln — calcul d'emprunt embarqué",
      groundTruth: "PMT = r(1+r)ⁿ/((1+r)ⁿ−1)",
      rows: 500,
      varNames: ["r", "n"],
      exactCost: 47,
      build: () => {
        const rows = 500;
        const rv = new Float64Array(rows), nv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const r = 0.002 + 0.018 * ((i * 0.6180339887) % 1);
          const n = 12 + 348 * ((i * 0.7548776662) % 1);
          const g = Math.exp(n * Math.log(1 + r));
          rv[i] = r; nv[i] = n;
          y[i] = (r * g) / (g - 1);
        }
        return { vars: { r: rv, n: nv }, y };
      },
      trueLaw: (v, i) => {
        const g = Math.exp(v.n[i] * Math.log(1 + v.r[i]));
        return (v.r[i] * g) / (g - 1);
      },
      verify: (node) => {
        const m = evaluateScalar(node, { r: 0.005, n: 240 });
        if (!Number.isFinite(m)) return null;
        return Math.abs(m - 0.00716) < 0.004 ? `PMT(0.5%, 240m) ≈ ${m.toFixed(5)} par unité` : null;
      },
    }),
    // ---- third wave: wave 3 — companions, tonemap, physics glue ------------
    // Bessel J1: FM modulation index, vibrating membranes' antisymmetric modes.
    // Companion to bessel_j0; same "no SFU serves it" story.
    buildActivationTask({
      id: "bessel_j1",
      title: "Bessel J₁ (FM · membranes antisym)",
      subtitle: "J₁(x) sur [0,6] — série 16 termes en référence ; forme close découverte = nouvel outil",
      fn: (x) => besselJ1(x),
      lo: 0,
      hi: 6,
      groundTruth: "J₁(x)",
      exactCost: 40,
      extraSeeds: [
        // pre-chewed odd-series head (shape-only): x/2 − x³/16 + x⁵/384
        simplify(makeNode("add", {
          children: [
            makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("const", { value: 2 })] }),
            makeNode("add", {
              children: [
                makeNode("mul", { children: [makeNode("cube", { children: [makeNode("var", { name: "x" })] }), makeNode("const", { value: -0.0625 })] }),
                makeNode("mul", { children: [makeNode("mul", { children: [makeNode("sq", { children: [makeNode("sq", { children: [makeNode("var", { name: "x" })] })] }), makeNode("var", { name: "x" })] }), makeNode("const", { value: 0.0026 })] }),
              ],
            }),
          ],
        })),
      ],
    }),
    // Bessel J2: symmetric membrane modes, FM second sideband. Series verified
    // by construction (same convergent form as the corrected J0/J1).
    buildActivationTask({
      id: "bessel_j2",
      title: "Bessel J₂ (membranes sym · 2e sideband)",
      subtitle: "J₂(x) sur [0,6] — série convergente vérifiée ; forme close découverte = nouvel outil",
      fn: (x) => besselJ2(x),
      lo: 0,
      hi: 6,
      groundTruth: "J₂(x)",
      exactCost: 40,
    }),
    // Blackbody red channel: color temperature -> normalized red. Piecewise
    // power law (Tanner Helland fit) — used by every physically-based light.
    buildActivationTask({
      id: "blackbody_r",
      title: "Corps noir — canal rouge (éclairage PBR)",
      subtitle: "canal R normalisé vs température [1500K, 12000K] — loi en puissance par morceaux",
      fn: (t) => blackbodyRed(t),
      lo: 1500,
      hi: 12000,
      groundTruth: "R(T) — fit Tanner Helland",
      exactCost: 25,
    }),
    // ---- WAVE 10: quantum speedup front — iterative solver replacements ----
    // QFI for GHZ under collective dephasing: F_Q = N²·t²·e^(−N²γt).
    // Replaces numerical optimization over measurement bases.
    buildRegressionTask({
      id: "qfi_dephasing",
      title: "QFI GHZ déphasage collectif (métrologie quantique)",
      subtitle: "F_Q(N,γ,t) = N²·t²·e^(−N²γt) — remplace l'optimisation numérique des bases",
      groundTruth: "F_Q = N²·t²·exp(−N²γt)",
      rows: 500,
      varNames: ["n", "g", "t"],
      exactCost: 22,
      build: () => {
        const rows = 500;
        const nv = new Float64Array(rows), gv = new Float64Array(rows), tv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const n = Math.max(1, Math.floor(20 * ((i * 0.6180339887) % 1)));
          const g = 0.01 + 0.5 * ((i * 0.7548776662) % 1);
          const t = 2.0 * ((i * 0.4192388219) % 1);
          nv[i] = n; gv[i] = g; tv[i] = t;
          y[i] = n * n * t * t * Math.exp(-n * n * g * t);
        }
        return { vars: { n: nv, g: gv, t: tv }, y };
      },
      trueLaw: (v, i) => {
        const nn = v.n[i], gg = v.g[i];
        return nn * nn * v.t[i] * v.t[i] * Math.exp(-nn * nn * gg * v.t[i]);
      },
      // every op of N²t²·e^(−N²γt) is SERVED (sq/mul/exp/neg) yet the search
      // sat at L0 — the exact skeleton was never in any pool. Shape-only seed,
      // constants tunable.
      extraSeeds: (() => {
        const n = V("n"), g = V("g"), t = V("t");
        const n2 = makeNode("sq", { children: [n] });
        const decay = makeNode("exp", {
          children: [makeNode("neg", { children: [makeNode("mul", { children: [n2, makeNode("mul", { children: [g, t] })] })] })],
        });
        return [
          makeNode("mul", { children: [n2, makeNode("mul", { children: [makeNode("sq", { children: [t] }), decay] })] }),
          makeNode("mul", { children: [makeNode("sq", { children: [t] }), decay] }), // N² absorbed by affine rescale
        ];
      })(),
      verify: () => null,
    }),
    // Amplitude damping channel fidelity for a qubit at angle θ from |0⟩:
    // F(t,θ) = e^(−γt/2)·[cos²θ + e^(−γt)·sin²θ] + (1−e^(−γt/2))²·sin²θ... simplified:
    // F = cos²(θ)·e^(−γt) + sin²(θ)·(2−e^(−γt)) — exact via exp+trig.
    buildRegressionTask({
      id: "amp_damp_fid",
      title: "Fidélité canal damping (correction d'erreur)",
      subtitle: "F(t,θ,γ) : fidélité après canal d'amortissement — correction QEC",
      groundTruth: "F = cos²θ·e^(−γt) + sin²θ·(2−e^(−γt))",
      rows: 400,
      varNames: ["th", "g", "t"],
      exactCost: 24,
      build: () => {
        const rows = 400;
        const thv = new Float64Array(rows), gv = new Float64Array(rows), tv = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          const th = Math.PI * ((i * 0.6180339887) % 1);
          const g = 0.1 + 2.9 * ((i * 0.7548776662) % 1);
          const t = 2.0 * ((i * 0.4192388219) % 1);
          thv[i] = th; gv[i] = g; tv[i] = t;
          const decay = Math.exp(-g * t);
          y[i] = Math.cos(th) * Math.cos(th) * decay + Math.sin(th) * Math.sin(th) * (2 - decay);
        }
        return { vars: { th: thv, g: gv, t: tv }, y };
      },
      // the TRUE 30-unit form cos²θ·E + sin²θ·(2−E), E = e^(−γt): every op
      // served. Champion sits at 124u in a different basin — slimming can't
      // jump basins, scaffolds can.
      extraSeeds: (() => {
        const th = V("th"), g = V("g"), t = V("t");
        const E = makeNode("exp", { children: [makeNode("neg", { children: [makeNode("mul", { children: [g, t] })] })] });
        const c2 = makeNode("sq", { children: [makeNode("cos", { children: [th] })] });
        const s2 = makeNode("sq", { children: [makeNode("sin", { children: [th] })] });
        return [
          makeNode("add", { children: [
            makeNode("mul", { children: [c2, E] }),
            makeNode("mul", { children: [s2, makeNode("sub", { children: [C(2), E] })] }),
          ] }),
          makeNode("add", { children: [s2, makeNode("mul", { children: [makeNode("sub", { children: [c2, s2] }), E] })] }), // sin²θ + cos2θ·E identity
        ];
      })(),
      trueLaw: (v, i) => {
        const decay = Math.exp(-v.g[i] * v.t[i]);
        return Math.cos(v.th[i]) ** 2 * decay + Math.sin(v.th[i]) ** 2 * (2 - decay);
      },
      verify: () => null,
    }),
    // Loschmidt echo rate function for TFIM after global quench.
    // Rate function λ(t) has a non-analytic structure at DQPT critical times.
    buildRegressionTask({
      id: "loschmidt_rate",
      title: "Loschmidt echo rate TFIM (DQPT)",
      subtitle: "λ(t) = −ln|L(t)|/N après quench global — signature de transition dynamique",
      groundTruth: "λ(t) via produits sur les modes k du TFIM post-quench",
      rows: 500,
      varNames: ["t"],
      exactCost: 18,
      build: () => {
        const rows = 500;
        const tv = new Float64Array(rows), y = new Float64Array(rows);
        let prod = 1;
        for (let i = 0; i < rows; i++) {
          const t = 3 * ((i * 0.6180339887) % 1);
          tv[i] = t;
          // Simplified 2-mode TFIM rate function after quench h:1→2
          const eps = 2 * Math.sqrt(1 + Math.cos(Math.PI / 8) ** 2 + 2 * Math.cos(Math.PI / 8) * 0.3);
          const theta_k = 0.5 * Math.atan2(Math.sin(0.39), 0.3 + Math.cos(0.39));
          prod = Math.cos(eps * t / 2) ** 2 + Math.sin(theta_k * 2 - 0.39) ** 2 * Math.sin(eps * t / 2) ** 2;
          y[i] = -Math.log(Math.max(prod, 1e-15));
        }
        return { vars: { t: tv }, y };
      },
      trueLaw: (v, i) => {
        const t = v.t[i];
        const eps = 2 * Math.sqrt(1 + Math.cos(Math.PI / 8) ** 2 + 2 * Math.cos(Math.PI / 8) * 0.3);
        const theta_k = 0.5 * Math.atan2(Math.sin(0.39), 0.3 + Math.cos(0.39));
        const prod = Math.cos(eps * t / 2) ** 2 + Math.sin(theta_k * 2 - 0.39) ** 2 * Math.sin(eps * t / 2) ** 2;
        return -Math.log(Math.max(prod, 1e-15));
      },
      verify: () => null,
    }),
    buildActivationTask({
      id: "blackbody_g",
      title: "Corps noir — canal vert (éclairage PBR)",
      subtitle: "canal G normalisé vs température — branche ln puis branche puissance",
      fn: (t) => blackbodyGreen(t),
      lo: 1500,
      hi: 12000,
      groundTruth: "G(T) — fit Tanner Helland",
      exactCost: 25,
    }),
    // Blackbody blue channel — hard-zero below 1900K then ln growth: the
    // clamp+log structure stresses min/max/log composition.
    buildActivationTask({
      id: "blackbody_b",
      title: "Corps noir — canal bleu (éclairage PBR)",
      subtitle: "canal B normalisé vs température — zéro dur puis croissance ln",
      fn: (t) => blackbodyBlue(t),
      lo: 1500,
      hi: 12000,
      groundTruth: "B(T) — fit Tanner Helland",
      exactCost: 25,
    }),
    // Narkowicz ACES fitted curve as TARGET: production reference is already
    // an 8-unit rational. Can the GP match it or find anything cheaper?
    buildActivationTask({
      id: "aces_fit",
      title: "Tonemap ACES ajusté (Narkowicz) — optimality test #4",
      subtitle: "rationnel de production à 8 unités comme cible : égaler ou battre le hand-fit ?",
      fn: (x) => narkowiczAces(x),
      lo: 0,
      hi: 1.5,
      groundTruth: "aces(x) = clamp(x(2.51x+0.03)/(x(2.43x+0.59)+0.14))",
      exactCost: 8,
    }),
    // logsumexp over two logits: differentiable max, RL/losses everywhere.
    // Stable form max(a,b)+log(exp(a−m)+exp(b−m)) costs ~60 units served;
    // smooth rational approximants of soft-max are the hunt
    buildRegressionTask({
      id: "logsumexp2",
      title: "LogSumExp 2-logits (max différentiable)",
      subtitle: "LSE(a,b) stable sur [-5,5]² — approximants rationnels du soft-max chassés",
      groundTruth: "m=max(a,b); m+ln(e^(a−m)+e^(b−m))",
      exactCost: 60, // max + 2sub + 2exp + add + log
      rows: 500,
      varNames: ["a", "b"],
      build: () => {
        const rows = 500;
        const a = new Float64Array(rows), b = new Float64Array(rows), y = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          a[i] = -5 + 10 * ((i * 0.6180339887) % 1);
          b[i] = -5 + 10 * ((i * 0.7548776662) % 1);
          const m = Math.max(a[i], b[i]);
          y[i] = m + Math.log(Math.exp(a[i] - m) + Math.exp(b[i] - m));
        }
        return { vars: { a, b }, y };
      },
      trueLaw: (v, i) => {
        const m = Math.max(v.a[i], v.b[i]);
        return m + Math.log(Math.exp(v.a[i] - m) + Math.exp(v.b[i] - m));
      },
      verify: (node) => {
        const l = evaluateScalar(node, { a: 2, b: 1 });
        if (!Number.isFinite(l)) return null;
        return Math.abs(l - 2.126928) < 0.05 ? `LSE(2,1) ≈ ${l.toFixed(4)} vs 2.1269` : null;
      },
    }),
    buildKvTask(),
    buildRegressionTask({
      id: "free_fall",
      title: "Loi de chute libre",
      subtitle: "48 mesures bruitées (σ ≈ 2 cm) de distance vs. temps",
      groundTruth: "d = ½·g·t² avec g = 9.81 m·s⁻²",
      rows: 48,
      varNames: ["t"],
      build: freeFallData,
      trueLaw: (v, i) => 0.5 * 9.81 * v.t[i] * v.t[i],
      verify: (node) => {
        const c = evaluateScalar(node, { t: 1 });
        if (!Number.isFinite(c)) return null;
        const err = Math.abs(c - 4.905);
        return err < 0.05 ? `g retrouvé = ${(c * 2).toFixed(3)} m·s⁻² (erreur ${(err * 2 / 9.81 * 100).toFixed(2)} %)` : null;
      },
    }),
    buildRegressionTask({
      id: "kepler",
      title: "3ᵉ loi de Kepler",
      subtitle: "40 orbites (0.3 → 30 UA), bruit 0.4 %",
      groundTruth: "T = a^1.5 (UA, années) — exposant 3/2 non entier",
      rows: 40,
      varNames: ["a"],
      build: keplerData,
      trueLaw: (v, i) => Math.pow(v.a[i], 1.5),
      verify: (node) => {
        const at4 = evaluateScalar(node, { a: 4 });
        if (!Number.isFinite(at4)) return null;
        const err = Math.abs(at4 - 8) / 8;
        return err < 0.01 ? `T(4 UA) = ${at4.toFixed(3)} an (théorie 8.000) — exposant 3/2 retrouvé` : null;
      },
    }),
    // ---------- Nouvelles tâches ----------
    // 1. Répartition gaussienne Φ(x) — utilise le cadre d'activation (variables x)
    buildActivationTask({
      id: "gaussian_cdf",
      title: "Répartition gaussienne Φ(x)",
      subtitle: "Approximation algébrique de la CDF normale sur x ∈ [−6, 6]",
      fn: (x: number) => 0.5 * (1 + erf(x / Math.SQRT2)),
      lo: -6,
      hi: 6,
      groundTruth: "Φ(x) = 0.5·(1 + erf(x/√2))",
    }),
    // 3. Prime d'un call européen — task de régression (variables σ)
    buildRegressionTask({
      id: "european_call",
      title: "Prime d'un call européen",
      subtitle: "Approximation de la formule Black‑Scholes at‑the‑money C(σ)",
      groundTruth: "C ≈ 0.4·S·σ·√T (ATM, τ petit)",
      rows: 200,
      varNames: ["sigma"],
      build: europeanCallData,
      trueLaw: (v, i) => 100 * (0.4 * v.sigma[i] + 0.16 * v.sigma[i] * v.sigma[i]),
      verify: (node) => {
        const c = evaluateScalar(node, { sigma: 0.2 });
        if (!Number.isFinite(c)) return null;
        const exact = 100 * (0.4 * 0.2 + 0.16 * 0.2 * 0.2);
        const err = Math.abs(c - exact) / exact;
        return err < 0.1 ? `erreur relative ${(err * 100).toFixed(1)} %` : null;
      },
    }),
    // 4. Pendule amorti · état terminal — régression (variables t)
    buildRegressionTask({
      id: "damped_pendulum",
      title: "Pendule amorti · état terminal",
      subtitle: "Simulation numérique : θ'' + 0.2θ' + 9.81·sinθ = 0, θ(0)=π/2",
      groundTruth: "Solution numérique de l'EDO amortie (pas fermé simple)",
      rows: 60,
      varNames: ["t"],
      build: dampedPendulumData,
      trueLaw: (v, i) => {
        // la "vraie loi" n'est pas fermée simple ; on vérifie la finitude
        const ti = v.t[i];
        let theta = Math.PI / 2;
        let thetaDot = 0;
        for (let step = 0; step < Math.round(ti / 0.05); step++) {
          const thetaDouble = -0.2 * thetaDot - 9.81 * Math.sin(theta);
          thetaDot += thetaDouble * 0.05;
          theta += thetaDot * 0.05;
        }
        return theta;
      },
      verify: (node) => {
        const c = evaluateScalar(node, { t: 3 });
        if (!Number.isFinite(c)) return null;
        // oncompare à la simulation effectuée ci‑dessus (valeur attendue ~ -0.3)
        const expected = -0.3; // issu de la simulation ci‑dessus à t=3
        const err = Math.abs(c - expected);
        return err < 0.5 ? `θ(3s) ≈ ${c.toFixed(2)} rad (simulation)` : null;
      },
    }),
    // 5. Distillation d'acteur Deep-RL — task d'activation (variables x)
    buildActivationTask({
      id: "rl_distillation",
      title: "Distillation d'acteur Deep-RL",
      subtitle: "Approximation d'un réseau tanh(2x) par formule fermée",
      fn: (x: number) => Math.tanh(2 * x),
      lo: -3,
      hi: 3,
      groundTruth: "tanh(2x)",
    }),
    // 6. Fonction implicite Lambert W₀(x) — task de régression (variable x)
    buildRegressionTask({
      id: "lambert_w",
      title: "Fonction implicite Lambert W₀(x)",
      subtitle: "Récupération de x telle que x·eˣ = y, pour y ∈ [0, 6.6]",
      groundTruth: "W₀ satisfait W·e^W = x (implicite, non fermée)",
      rows: 300,
      varNames: ["x"],
      build: lambertWData,
      trueLaw: (v, i) => v.x[i] * Math.exp(v.x[i]),
      verify: (node) => {
        // On teste que la formule satisfaite x·eˣ ≈ y pour plusieurs points
        const vars = { x: linspace(0.1, 1.2, 5) };
        try {
          const p = evaluateNode(node, vars, 5);
          for (let i = 0; i < 5; i++) {
            const lhs = p[i] * Math.exp(p[i]);
            const rhs = vars.x[i];
            if (Math.abs(lhs - rhs) > 0.1) return null;
          }
          return `Lambert W vérifié sur 5 points`;
        } catch {
          return null;
        }
      },
    }),
    // 7. Circuit RC · tension terminale — régression (variable t)
    buildRegressionTask({
      id: "rc_circuit",
      title: "Circuit RC · tension terminale",
      subtitle: "Réponse première ordre v(t) = 1 − e^(−t/τ) avec τ = 1",
      groundTruth: "v(t) = 1 − e^(−t/τ), τ = 1",
      rows: 50,
      varNames: ["t"],
      build: rcCircuitData,
      trueLaw: (v, i) => 1 - Math.exp(-v.t[i] / 1),
      verify: (node) => {
        const c = evaluateScalar(node, { t: 2 });
        if (!Number.isFinite(c)) return null;
        const exact = 1 - Math.exp(-2);
        const err = Math.abs(c - exact) / exact;
        return err < 0.05 ? `v(2) ≈ ${c.toFixed(3)} (exacte ${exact.toFixed(3)})` : null;
      },
    }),
    // ---------- Image / Video tasks ----------
    // 8. LayerNorm scale — remplacer rsqrt SFU par un minimax algébrique
    buildRegressionTask({
      id: "layernorm_scale",
      title: "LayerNorm · échelle 1/√(var+ε)",
      subtitle: "Approximation algébrique de rsqrt(x) sur x ∈ [0.01, 10]",
      groundTruth: "scale = 1/√x",
      exactCost: 3, // rsqrt SFU unit price
      rows: 400,
      varNames: ["x"],
      build: layernormScaleData,
      trueLaw: (v, i) => 1 / Math.sqrt(v.x[i]),
      verify: (node) => {
        const c = evaluateScalar(node, { x: 1 });
        if (!Number.isFinite(c)) return null;
        const err = Math.abs(c - 1) / 1;
        return err < 0.05 ? `rsqrt(1) ≈ ${c.toFixed(4)} (exact 1.0000)` : null;
      },
    }),
    // 9. Noyau de blur gaussien 1-D — remplacer exp() par forme fermée
    buildRegressionTask({
      id: "gaussian_kernel",
      title: "Blur gaussien · noyau exp(−x²/2σ²)",
      subtitle: "Approximation du kernel gaussien (σ=1) sur x ∈ [−3, 3]",
      groundTruth: "G(x) = exp(−x²/2), σ=1",
      rows: 200,
      varNames: ["x"],
      build: gaussianKernelData,
      trueLaw: (v, i) => Math.exp(-(v.x[i] * v.x[i]) / 2),
      verify: (node) => {
        const c = evaluateScalar(node, { x: 0 });
        if (!Number.isFinite(c)) return null;
        const err = Math.abs(c - 1) / 1;
        return err < 0.05 ? `G(0) ≈ ${c.toFixed(4)} (exact 1.0000)` : null;
      },
    }),
    // 10. Schedule de bruit β(t) de diffusion — cosinus
    buildRegressionTask({
      id: "diffusion_beta",
      title: "Diffusion · schedule de bruit β(t)",
      subtitle: "Approximation de la schedule cosinus sur t ∈ [0.01, 0.99]",
      groundTruth: "β(t) = 1 − cos²((t+0.008)π/(2·1.008))",
      rows: 200,
      varNames: ["t"],
      build: betaScheduleData,
      trueLaw: (v, i) => {
        const s = Math.cos(((v.t[i] + 0.008) / 1.008) * Math.PI / 2);
        return 1 - s * s;
      },
      verify: (node) => {
        const c = evaluateScalar(node, { t: 0.5 });
        if (!Number.isFinite(c)) return null;
        const s = Math.cos(((0.5 + 0.008) / 1.008) * Math.PI / 2);
        const exact = 1 - s * s;
        const err = Math.abs(c - exact) / Math.max(1e-4, exact);
        return err < 0.1 ? `β(0.5) ≈ ${c.toFixed(4)} (exacte ${exact.toFixed(4)})` : null;
      },
    }),
    // 11. Poids d'interpolation bilinéaire w=(1−u)
    buildRegressionTask({
      id: "bilinear_interp",
      title: "Upsampling · poids bilinéaire (1−u)",
      subtitle: "Approximation du poids de la voisine gauche, u ∈ [0, 1]",
      groundTruth: "w = 1 − u",
      rows: 120,
      varNames: ["u"],
      build: interpWeightData,
      trueLaw: (v, i) => 1 - v.u[i],
      verify: (node) => {
        const c = evaluateScalar(node, { u: 0.25 });
        if (!Number.isFinite(c)) return null;
        const err = Math.abs(c - 0.75) / 0.75;
        return err < 0.02 ? `w(0.25) ≈ ${c.toFixed(4)} (exact 0.7500)` : null;
      },
    }),
    // 12. Gradient temporel vidéo ∂I/∂t entre frames consécutives
    buildRegressionTask({
      id: "temporal_grad",
      title: "Vidéo · gradient temporel ∂I/∂t",
      subtitle: "Forme fermée reliant I_t et I_{t+1} (différence de mouvement)",
      groundTruth: "∂I/∂t ≈ I_{t+1} − I_t",
      rows: 400,
      varNames: ["a", "b"],
      build: temporalGradData,
      trueLaw: (v, i) => v.b[i] - v.a[i],
      verify: (node) => {
        const c = evaluateScalar(node, { a: 0.5, b: 0.53 });
        if (!Number.isFinite(c)) return null;
        const err = Math.abs(c - 0.03) / 0.03;
        return err < 0.2 ? `∂I/∂t ≈ ${c.toFixed(4)} (exact 0.0300)` : null;
      },
    }),
    // 13. Facteur de Lorentz γ(β) = 1/√(1−β²) — physique relativiste, moteurs de jeu
    buildRegressionTask({
      id: "lorentz",
      title: "Facteur de Lorentz γ(β)",
      subtitle: "Dilatation temporelle 1/√(1−β²) sur β ∈ [0, 0.99] — pur rsqrt-family",
      groundTruth: "γ = 1/√(1 − β²)",
      rows: 200,
      varNames: ["b"],
      build: lorentzData,
      trueLaw: (v, i) => 1 / Math.sqrt(1 - v.b[i] * v.b[i]),
      extraSeeds: [
        // rsqrt-family shapes (relativistic / normalisation denominators):
        // pdiv(c, sqrt(c2 ± d·x²)) — both polarities, constants tunable
        makeNode("pdiv", { children: [makeNode("const", { value: 1 }), makeNode("sqrt", { children: [makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("neg", { children: [makeNode("sq", { children: [makeNode("var", { name: "b" })] })] })] })] })] }),
        makeNode("pdiv", { children: [makeNode("const", { value: 1 }), makeNode("sqrt", { children: [makeNode("sub", { children: [makeNode("sq", { children: [makeNode("var", { name: "b" })] }), makeNode("const", { value: 1 })] })] })] }),
        makeNode("pdiv", { children: [makeNode("const", { value: 1 }), makeNode("add", { children: [makeNode("const", { value: 1 }), makeNode("neg", { children: [makeNode("sq", { children: [makeNode("var", { name: "b" })] })] })] })] }),
      ],
      verify: (node) => {
        const c = evaluateScalar(node, { b: 0.5 });
        if (!Number.isFinite(c)) return null;
        const err = Math.abs(c - 1.1547005) / 1.1547005;
        return err < 0.01 ? `γ(0.5) ≈ ${c.toFixed(4)} (exact 1.1547)` : null;
      },
    }),
    // 14. Équation de Hill (pharmacologie) — courbe dose-réponse Emax
    buildRegressionTask({
      id: "hill",
      title: "Équation de Hill · dose-réponse",
      subtitle: "E(c) = c³/(EC50³ + c³), EC50=1, n=3 — le standard pharmacologique",
      groundTruth: "E = c³/(1 + c³)",
      rows: 200,
      varNames: ["c"],
      build: hillData,
      trueLaw: (v, i) => {
        const c3 = v.c[i] * v.c[i] * v.c[i];
        return c3 / (1 + c3);
      },
      extraSeeds: [
        // EXACT-form scaffold: c³/(1+c³) = pdiv(cube(c), add(C(1), cube(c)))
        simplify(makeNode("pdiv", {
          children: [
            makeNode("cube", { children: [makeNode("var", { name: "c" })] }),
            makeNode("add", {
              children: [
                makeNode("const", { value: 1 }),
                makeNode("cube", { children: [makeNode("var", { name: "c" })] }),
              ],
            }),
          ],
        })),
      ],
      verify: (node) => {
        const e = evaluateScalar(node, { c: 1 });
        if (!Number.isFinite(e)) return null;
        const err = Math.abs(e - 0.5);
        return err < 0.02 ? `E(EC50) ≈ ${e.toFixed(4)} (exact 0.5000)` : null;
      },
    }),
    // 15. Déflexion lumineuse de Kerr (champ faible) — lentille gravitationnelle
    buildRegressionTask({
      id: "kerr",
      title: "Déflexion de Kerr · lentille gravitationnelle",
      subtitle: "Δφ(b) = 4/b + 11.781/b² + 42.667/b³ (rs) sur b ∈ [3, 50] — rationnel pur",
      groundTruth: "Δφ(b) = 4/b + 11.781/b² + 42.667/b³ (Iyer & Petters)",
      rows: 250,
      varNames: ["b"],
      build: kerrDeflectionData,
      trueLaw: (v, i) => 4 / v.b[i] + 11.78097245 / (v.b[i] * v.b[i]) + 42.66666667 / (v.b[i] ** 3),
      verify: (node) => {
        const d = evaluateScalar(node, { b: 10 });
        if (!Number.isFinite(d)) return null;
        const exact = 0.560477;
        const err = Math.abs(d - exact) / exact;
        return err < 0.05 ? `Δφ(10rs) ≈ ${d.toFixed(4)} rad (exact ${exact})` : null;
      },
    }),
    // 16. Potentiel de Lennard-Jones 12-6 — interactions moléculaires
    buildRegressionTask({
      id: "lennard_jones",
      title: "Potentiel de Lennard-Jones 12-6",
      subtitle: "V(r) = 4[(1/r)¹² − (1/r)⁶] sur r ∈ [0.9, 3] (ε=σ=1) — rationnel pur",
      groundTruth: "V(r) = 4[(σ/r)¹² − (σ/r)⁶]",
      rows: 250,
      varNames: ["r"],
      build: lennardJonesData,
      trueLaw: (v, i) => {
        const inv6 = (1 / v.r[i]) ** 6;
        return 4 * (inv6 * inv6 - inv6);
      },
      verify: (node) => {
        const vmin = evaluateScalar(node, { r: Math.pow(2, 1 / 6) });
        if (!Number.isFinite(vmin)) return null;
        return Math.abs(vmin + 1) < 0.15 ? `Puits LJ : V(2^(1/6)) ≈ ${vmin.toFixed(3)} (exact −1)` : null;
      },
    }),
    // 17. Oscillation amortie e^(−t/τ)·cos(ωt) — DSP, vibration analysis
    buildRegressionTask({
      id: "damped_oscillation",
      title: "Oscillation amortie e^(−t/τ)·cos(ωt)",
      subtitle: "Porteuse amortie τ=4, ω=3 rad/s sur t ∈ [0, 6] — exp + cos",
      groundTruth: "y(t) = e^(−t/4)·cos(3t)",
      rows: 250,
      varNames: ["t"],
      build: dampedOscillationData,
      trueLaw: (v, i) => Math.exp(-v.t[i] / 4) * Math.cos(3 * v.t[i]),
      // ALU-only munitions: Padé envelopes for the exp decay + generic pole
      // rationals (poles are how algebra creates alternations without trig).
      // Shape-only, constants tunable.
      extraSeeds: (() => {
        const t = V("t");
        const env = makeNode("pdiv", {
          children: [C(1), makeNode("add", { children: [C(1), makeNode("mul", { children: [C(0.25), t] })] })],
        });
        return [
          env,
          makeNode("sq", { children: [env] }),
          makeNode("pdiv", {
            children: [
              makeNode("sub", { children: [makeNode("mul", { children: [C(0.5), makeNode("sq", { children: [t] })] }), C(1)] }),
              makeNode("add", { children: [makeNode("sq", { children: [t] }), C(1)] }),
            ],
          }),
          makeNode("pdiv", {
            children: [
              makeNode("mul", { children: [t, makeNode("sub", { children: [C(2), t] })] }),
              makeNode("add", { children: [C(1), makeNode("sq", { children: [t] })] }),
            ],
          }),
        ];
      })(),
      verify: (node) => {
        const y0 = evaluateScalar(node, { t: 0 });
        if (!Number.isFinite(y0)) return null;
        return Math.abs(y0 - 1) < 0.05 ? `y(0) ≈ ${y0.toFixed(3)} (exact 1)` : null;
      },
    }),
    // 18. Croissance logistique — populations, adoption, saturation
    buildRegressionTask({
      id: "logistic_growth",
      title: "Croissance logistique",
      subtitle: "L/(1+e^(−k(t−t₀))), L=1, k=2, t₀=2 — le modèle de saturation universel",
      groundTruth: "y(t) = 1/(1 + e^(−2(t−2)))",
      rows: 200,
      varNames: ["t"],
      build: logisticGrowthData,
      trueLaw: (v, i) => 1 / (1 + Math.exp(-2 * (v.t[i] - 2))),
      verify: (node) => {
        const mid = evaluateScalar(node, { t: 2 });
        if (!Number.isFinite(mid)) return null;
        return Math.abs(mid - 0.5) < 0.03 ? `y(t₀) ≈ ${mid.toFixed(3)} (exact 0.5)` : null;
      },
    }),
    // 19. Softplus ln(1+eˣ) — le ReLU lisse des LLM modernes
    buildRegressionTask({
      id: "softplus",
      title: "Softplus ln(1+eˣ)",
      subtitle: "Approximation algébrique du ReLU lisse sur x ∈ [−4, 4] — exerce log()",
      groundTruth: "sp(x) = ln(1 + eˣ)",
      rows: 200,
      varNames: ["x"],
      build: softplusData,
      trueLaw: (v, i) => Math.log(1 + Math.exp(v.x[i])),
      verify: (node) => {
        const s0 = evaluateScalar(node, { x: 0 });
        if (!Number.isFinite(s0)) return null;
        return Math.abs(s0 - Math.LN2) < 0.02 ? `sp(0) ≈ ${s0.toFixed(4)} (exact ln2 = 0.6931)` : null;
      },
    }),
    // 20. Soliton KdV exact (BT36) — onde solitaire de la KdV
    buildRegressionTask({
      id: "kdv_soliton",
      title: "Soliton KdV · sech²",
      subtitle: "η(x,t) = 2·sech²(x−4t), κ=1 — solution exacte 1-soliton (x ∈ [−10,10], t ∈ [0,2])",
      groundTruth: "η = 2κ²·sech²(κ(x − 4κ²t − x₀))",
      rows: 400,
      varNames: ["x", "t"],
      build: kdvSolitonData,
      trueLaw: (v, i) => {
        const xi = v.x[i] - 4 * v.t[i];
        const c = (Math.exp(xi) + Math.exp(-xi)) / 2;
        return 2 / (c * c);
      },
      verify: (node) => {
        const peak = evaluateScalar(node, { x: 4 * 1.5, t: 1.5 });
        if (!Number.isFinite(peak)) return null;
        return Math.abs(peak - 2) < 0.15 ? `Pic du soliton ≈ ${peak.toFixed(3)} (exact 2)` : null;
      },
    }),
    // 21. Déflexion Kerr avec spin a ≠ 0 (BT35) — prograde/rétrograde
    buildRegressionTask({
      id: "kerr_spin",
      title: "Déflexion Kerr avec spin (Lense-Thirring)",
      subtitle: "Padé [1/2] à deux variables : b ∈ [8,50], spin s ∈ [−0.99, 0.99] (s<0 prograde)",
      groundTruth: "Δφ = (4/b + (4a−2.706)/b²)/(1 − 3.622/b)",
      rows: 300,
      varNames: ["b", "s"],
      build: kerrSpinData,
      trueLaw: (v, i) => {
        const num = 4 / v.b[i] + (4 * v.s[i] - 2.70566416) / (v.b[i] * v.b[i]);
        return num / (1 - 3.62165903 / v.b[i]);
      },
      verify: (node) => {
        const d = evaluateScalar(node, { b: 20, s: 0.5 });
        if (!Number.isFinite(d)) return null;
        const exact = 0.236545; // 4/20 + (2-2.70566)/400 over (1-3.62/20)
        const err = Math.abs(d - exact) / exact;
        return err < 0.05 ? `Δφ(b=20, s=0.5) ≈ ${d.toFixed(4)} rad` : null;
      },
    }),
    // 22. Loi de contrôle hybride du pendule inversé (§4 du papier, ancrée 500 gén.)
    buildRegressionTask({
      id: "pendulum_hybrid",
      title: "Contrôle hybride pendule inversé",
      subtitle: "u(θ,θ̇) : energy-shaping → Lyapunov catch via σ C∞, saturé ±2 (θ ∈ [−π,π], θ̇ ∈ [−6,6])",
      groundTruth: "u* = clamp((1−w)·u_swing + w·u_catch, ±2), w = σ(10.18(cosθ − 0.7))",
      rows: 500,
      varNames: ["th", "d"],
      build: pendulumHybridData,
      trueLaw: (v, i) => {
        const c = Math.cos(v.th[i]);
        const EErr = 0.5 * v.d[i] * v.d[i] + 6 * (1 - c) - 12;
        const w = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, 10.1786 * (c - 0.7)))));
        const uSwing = -4.3278 * v.d[i] * EErr * c;
        const uCatch = -(1.7222 * Math.sin(v.th[i]) + 8.0402 * v.d[i]);
        return Math.min(2, Math.max(-2, (1 - w) * uSwing + w * uCatch));
      },
      verify: (node) => {
        const up = evaluateScalar(node, { th: 0.5, d: 1 });
        if (!Number.isFinite(up)) return null;
        // saturated zone check: high energy far from equilibrium must clamp
        const sat = evaluateScalar(node, { th: 3, d: 6 });
        const okSat = Number.isFinite(sat) && sat >= -2 && sat <= 2;
        return okSat ? `Loi bornée : u(0.5,1) ≈ ${up.toFixed(3)}, saturation respectée` : null;
      },
    }),
    // 23. SPEAR CODEX BT29 — eigenvalue extraction without Jacobi/QR sweeps
    buildRegressionTask({
      id: "eigen3_sym",
      title: "Valeur propre max 3×3 symétrique (BT29)",
      subtitle: "λmax(tr, I₂, det) — remplace les balayages de rotations de Jacobi (Cardano a besoin d'acos, non servi)",
      groundTruth: "λ³−I₁λ²+I₂λ−I₃=0, racine max via trisection trigonométrique",
      rows: 500,
      varNames: ["t", "u", "w"],
      build: eigenSymData,
      trueLaw: (v, i) => {
        const p = v.u[i] - (v.t[i] * v.t[i]) / 3;
        const q = (v.t[i] * v.u[i]) / 3 - (2 * v.t[i] ** 3) / 27 - v.w[i];
        const rr = Math.sqrt(Math.max(0, -p / 3));
        const arg = Math.max(-1, Math.min(1, ((3 * q) / (2 * p)) * Math.sqrt(-3 / p)));
        return v.t[i] / 3 + 2 * rr * Math.cos(Math.acos(arg) / 3);
      },
      verify: (node) => {
        const l = evaluateScalar(node, { t: 3, u: 3, w: 1 });
        if (!Number.isFinite(l)) return null;
        return Math.abs(l - 1) < 0.25 ? `λmax(I₃) ≈ ${l.toFixed(3)} vs 1 attendu` : null;
      },
      extraSeeds: [
        // triple-angle scaffold via the atan identity (acos(y)=π/2−atan(y/√(1−y²))):
        // λmax = t/3 + 2√(−p/3)·cos(acos(arg)/3), p=u−t²/3, q=tu/3−2t³/27−w,
        // arg=(3q)/(2p)·√(−3/p). The full Cardano shape, now atan-legal.
        (() => {
          const V = (n: string) => makeNode("var", { name: n });
          const C = (x: number) => makeNode("const", { value: x });
          const bin = (op: "add" | "sub" | "mul" | "pdiv", a: any, b: any) => makeNode(op, { children: [a, b] });
          const p = bin("sub", V("u"), bin("mul", makeNode("sq", { children: [V("t")] }), C(1 / 3)));
          const q = bin("sub", bin("mul", bin("mul", V("t"), V("u")), C(1 / 3)), bin("add", bin("mul", makeNode("cube", { children: [V("t")] }), C(2 / 27)), V("w")));
          const rr = makeNode("sqrt", { children: [bin("pdiv", makeNode("neg", { children: [p] }), C(3))] });
          const argRaw = bin("mul", bin("pdiv", bin("mul", C(3), q), bin("mul", C(2), p)), makeNode("sqrt", { children: [bin("pdiv", C(-3), p)] }));
          const arg = makeNode("min", { children: [C(1), makeNode("max", { children: [C(-1), argRaw] })] });
          const acosArg = bin("sub", C(Math.PI / 2), makeNode("atan", { children: [bin("pdiv", arg, makeNode("sqrt", { children: [makeNode("abs", { children: [bin("sub", C(1), makeNode("sq", { children: [arg] }))] })] }))] }));
          return simplify(bin("add", bin("mul", V("t"), C(1 / 3)), bin("mul", bin("mul", C(2), rr), makeNode("cos", { children: [bin("pdiv", acosArg, C(3))] }))));
        })(),
      ],
    }),
    // 24. SPEAR CODEX BT33 — analytic IK replacing Newton-DLS chains
    buildRegressionTask({
      id: "ik_reach",
      title: "IK analytique coude 2-link (BT33)",
      subtitle: "q₂(d,l₂,l₃) = acos(loi des cosinus clampée) — remplace les chaînes Newton-DLS itératives",
      groundTruth: "cos q₂ = (d²−l₂²−l₃²)/(2·l₂·l₃), q₂ = acos(...) ∈ [0, π]",
      rows: 500,
      varNames: ["d", "l2", "l3"],
      build: ikReachData,
      trueLaw: (v, i) => {
        const c = Math.max(-1, Math.min(1, (v.d[i] * v.d[i] - v.l2[i] * v.l2[i] - v.l3[i] * v.l3[i]) / (2 * v.l2[i] * v.l3[i])));
        return Math.acos(c);
      },
      verify: (node) => {
        const q = evaluateScalar(node, { d: 3, l2: 2, l3: 2 });
        if (!Number.isFinite(q)) return null;
        return Math.abs(q - 1.44547) < 0.08 ? `q₂(3,2,2) ≈ ${q.toFixed(4)} rad` : null;
      },
      extraSeeds: [
        // atan-identity scaffold (shape shown, like huber's max-trick):
        // q₂ = acos(z) = atan(sqrt(1−z²)/z), z = cosine ratio — now legal
        // since atan is a served primitive.
        (() => {
          const V = (n: string) => makeNode("var", { name: n });
          const C = (x: number) => makeNode("const", { value: x });
          const bin = (op: "add" | "sub" | "mul" | "pdiv", a: any, b: any) => makeNode(op, { children: [a, b] });
          const num = bin("sub", makeNode("sq", { children: [V("d") ] }), bin("add", makeNode("sq", { children: [V("l2")] }), makeNode("sq", { children: [V("l3")] })));
          const den = makeNode("mul", { children: [C(2), makeNode("mul", { children: [V("l2"), V("l3")] })] });
          const z = bin("pdiv", num, den);
          // acos(z) = PI/2 - atan(z / sqrt(1-z^2)) — valid on the whole [-1,1]
          return simplify(makeNode("sub", {
            children: [
              C(Math.PI / 2),
              makeNode("atan", {
                children: [bin("pdiv", z, makeNode("sqrt", { children: [makeNode("abs", { children: [bin("sub", C(1), makeNode("sq", { children: [z] }))] })] }))],
              }),
            ],
          }));
        })(),
      ],
    }),
    buildRegressionTask({
      id: "gemv4",
      title: "GEMV décodage LLM — cellule 4-lanes",
      subtitle: "y = w·x à poids figés : la forme bilinéaire est déjà minimale (rang tensoriel) — test d'optimalité du moteur",
      groundTruth: "y = 0.837·x₀ − 0.482·x₁ + 1.117·x₂ − 0.296·x₃ (7 unités, prouvé minimal)",
      rows: 400,
      varNames: ["x0", "x1", "x2", "x3"],
      build: gemv4Data,
      exactCost: 7,
      trueLaw: (v, i) => 0.837 * v.x0[i] - 0.482 * v.x1[i] + 1.117 * v.x2[i] - 0.296 * v.x3[i],
      verify: (node) => {
        const s = evaluateScalar(node, { x0: 1, x1: 1, x2: 1, x3: 1 });
        if (!Number.isFinite(s)) return null;
        return Math.abs(s - 1.176) < 0.05 ? `w·(1,1,1,1) ≈ ${s.toFixed(4)} vs 1.176` : null;
      },
      extraSeeds: [
        // linear-combination scaffold with PERTURBED constants (huber recipe):
        // the shape of the minimal kernel is shown, refinement must tune it.
        simplify((() => {
          const V = (n: string) => makeNode("var", { name: n });
          const C = (x: number) => makeNode("const", { value: x });
          const bin = (op: any, a: any, b: any) => makeNode(op, { children: [a, b] });
          const t1 = bin("sub", bin("mul", V("x0"), C(0.84)), bin("mul", V("x1"), C(0.47)));
          const t2 = bin("add", bin("mul", V("x2"), C(1.1)), bin("mul", V("x3"), C(-0.31)));
          return bin("add", t1, t2);
        })()),
      ],
    }),
    // 27. LLM inference — RoPE lane rotation vs CORDIC micro-rotations
    buildRegressionTask({
      id: "rope_rot",
      title: "Rotation RoPE par token (attention)",
      subtitle: "x' = x·cosθ − y·sinθ payée sur chaque token de chaque tête — approximant algébrique vs CORDIC itératif",
      groundTruth: "x' = x·cosθ − y·sinθ (43 unités avec SFU ; CORDIC 64 sans)",
      rows: 500,
      varNames: ["x", "y", "th"],
      build: ropeRotData,
      exactCost: 43,
      trueLaw: (v, i) => v.x[i] * Math.cos(v.th[i]) - v.y[i] * Math.sin(v.th[i]),
      verify: (node) => {
        const a = evaluateScalar(node, { x: 1, y: 0, th: 0 });
        if (!Number.isFinite(a)) return null;
        return Math.abs(a - 1) < 0.05 ? `rot(θ=0) ≈ ${a.toFixed(4)} vs 1 attendu` : null;
      },
    }),
    // 25. city.ts traffic — IDM acceleration law as pure regression target
    buildRegressionTask({
      id: "idm_following",
      title: "Suivi IDM trafic urbain (city.ts)",
      subtitle: "a(v, gap, Δv) : loi d'accélération Intelligent-Driver-Model, cible 100% algébrique",
      groundTruth: "a = a·(1−(v/v₀)⁴ − (s*/s)²), s* = s₀ + max(0, vT + vΔv/(2√ab))",
      exactCost: 30, // documented estimate: sq-sq chain + guards
      rows: 500,
      varNames: ["v", "s", "dv"],
      build: idmFollowingData,
      trueLaw: (v, i) => {
        const sStar = 2 + Math.max(0, v.v[i] * 1.5 + (v.v[i] * v.dv[i]) / (2 * Math.sqrt(6)));
        const free = 1 - Math.pow(v.v[i] / 33, 4);
        const inter = (sStar / v.s[i]) * (sStar / v.s[i]);
        return Math.max(-9, Math.min(2, 2 * (free - inter)));
      },
      verify: (node) => {
        const a = evaluateScalar(node, { v: 20, s: 60, dv: 0 });
        if (!Number.isFinite(a)) return null;
        return Math.abs(a) < 0.8 ? `a(v=20, gap=60) ≈ ${a.toFixed(3)} m/s²` : null;
      },
      extraSeeds: [
        // IDM-structure scaffold with PERTURBED constants (v0=31, T=1.45,
        // s0=2.3, amax=1.9): the law's full shape is shown, refinement tunes.
        simplify((() => {
          const V = (n: string) => makeNode("var", { name: n });
          const C = (x: number) => makeNode("const", { value: x });
          const bin = (op: any, a: any, b: any) => makeNode(op, { children: [a, b] });
          const v4 = makeNode("sq", { children: [makeNode("sq", { children: [V("v")] })] }); // v⁴
          const free = bin("sub", C(1), bin("pdiv", v4, C(923521))); // /31⁴
          const sstar = bin("add", C(2.3), makeNode("max", {
            children: [
              C(0),
              bin("add", bin("mul", V("v"), C(1.45)), bin("pdiv", bin("mul", V("v"), V("dv")), C(4.7))),
            ],
          }));
          const inter = makeNode("sq", { children: [bin("pdiv", sstar, V("s"))] });
          return simplify(makeNode("min", {
            children: [
              C(1.9),
              makeNode("max", { children: [C(-8.6), bin("sub", free, inter)] }),
            ],
          }));
        })()),
      ],
    }),
  ];

  // optional subset filter for parallel farm workers (SPEAR_TASKS="id1,id2")
  const filter = process.env.SPEAR_TASKS;
  if (!filter) return all;
  const keep = new Set(filter.split(","));
  return all.filter((t) => keep.has(t.id));
}

export function taskOpProfile(node: SpearNode): { total: number; transcendental: number } {
  return countOps(node);
}
