import type { GpConfig, SpearNode } from "../engine";
// SPEAR task contracts - shared by factories, registries and the loop.
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
  /** R² on train data — powers the deployable fast-slot grade (D2: r² ≥ 0.98) */
  r2?: (node: SpearNode) => number;
}
