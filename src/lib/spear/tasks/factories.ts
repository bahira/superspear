import {
  makeNode,
  evaluateNode,
  evaluateScalar,
  fitLinearScaling,
  wrapAffine,
  refineConstants,
  simplify,
  parseNode,
  canonicalKey,
  countOps,
  type SpearNode,
  type GpConfig,
} from "../engine";
import { mapArray, mse, linfError, r2Score } from "../math-utils";
import { makeOodProbe, compositeSeeds } from "../heritage";
import type { TaskBaseline, TaskMilestone, TaskDef, TaskEval } from "./types";
import {
  GP_OPS_EFFECTIVE,
  pureSeeds,
  grid,
  ols1,
  loadBootstrapSeeds,
  V,
  C,
  EXACT_LAWS,
  ITERATIVE_BASELINES,
} from "./shared";

export interface ActivationSpec {
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

export function buildActivationTask(spec: ActivationSpec, points = 400): TaskDef {
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
    r2: (node) => {
      try { return r2Score(evaluateNode(node, vars, points), y); } catch { return -Infinity; }
    },
    exactFn: spec.fn,
    exactCost: spec.exactCost ?? (spec.id === "gelu" ? 46 : 34),
    iterativeBaseline: ITERATIVE_BASELINES[spec.id],
  };
}

export function buildRegressionTask(cfg: {
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
    r2: (node) => {
      try { return r2Score(evaluateNode(node, vars, n), y); } catch { return -Infinity; }
    },
    exactRefNode: cfg.exactLaw ?? EXACT_LAWS[cfg.id],
    exactCost: cfg.exactCost,
    iterativeBaseline: ITERATIVE_BASELINES[cfg.id],
  };
}
