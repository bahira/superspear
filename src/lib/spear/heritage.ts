// SPEAR — heritage & robustness utilities.
// Two responsibilities:
//   1. OOD probes (constrained NSGA-II): measure how a formula behaves OUTSIDE
//      its training support, returned as a constraint violation for the engine's
//      feasibility-first dominance.
//   2. Composite seed shapes: recurring algebraic motifs (softsign, Padé,
//      shifted-rsqrt) that the GP otherwise rebuilds by hand every run.

import {
  evaluateNode,
  fitLinearScaling,
  makeNode,
  wrapAffine,
  type SpearNode,
} from "./engine";

// ---------------------------------------------------------------------------
// OOD probes
// ---------------------------------------------------------------------------

export interface OodData {
  vars: Record<string, Float64Array>;
  y: Float64Array;
  n: number;
}

/**
 * Relative-RMSE threshold on the out-of-distribution band. A candidate whose
 * extrapolation error exceeds 50% of the target's own scale is infeasible.
 * Chosen loose enough that honest smooth approximations pass, tight enough
 * that pole/clamp blow-ups die.
 */
export const OOD_KAPPA = 0.5;

/**
 * Build an OOD violation probe: affine-rescales the candidate on TRAIN data
 * (same Keijzer linear scaling the scored metric uses), then measures relative
 * RMSE on an extrapolation band. Returns violation >= 0 (0 = feasible),
 * Infinity when anything is non-finite — the engine treats that as infeasible.
 */
export function makeOodProbe(train: OodData, ood: OodData): (node: SpearNode) => number {
  return (node: SpearNode): number => {
    try {
      const p = evaluateNode(node, train.vars, train.n);
      const { a, b } = fitLinearScaling(p, train.y);
      const wrapped = wrapAffine(node, a, b);
      const q = evaluateNode(wrapped, ood.vars, ood.n);
      let se = 0;
      let st = 0;
      for (let i = 0; i < ood.n; i++) {
        if (!Number.isFinite(q[i])) return Infinity;
        const e = q[i] - ood.y[i];
        se += e * e;
        st += ood.y[i] * ood.y[i];
      }
      const scale = Math.sqrt(st / ood.n);
      const rel = Math.sqrt(se / ood.n) / (scale > 1e-12 ? scale : 1);
      return Math.max(0, rel - OOD_KAPPA);
    } catch {
      return Infinity;
    }
  };
}

// ---------------------------------------------------------------------------
// Composite seed shapes (shape-only doctrine: constants stay tunable)
// ---------------------------------------------------------------------------

/** Generic algebraic composites the HoF shows the GP rediscovers constantly. */
export function compositeSeeds(v: string): SpearNode[] {
  const x = makeNode("var", { name: v });
  const C = (value: number): SpearNode => makeNode("const", { value });
  const abs = makeNode("abs", { children: [x] });
  const sq = makeNode("sq", { children: [x] });
  return [
    // softsign family — saturation without transcendental units
    makeNode("pdiv", { children: [x, makeNode("add", { children: [C(1), abs] })] }),
    makeNode("pdiv", { children: [x, makeNode("add", { children: [C(2), abs] })] }),
    // Padé [3/2] rational S-curve: (x + c·x³)/(1 + d·x²)
    makeNode("pdiv", {
      children: [
        makeNode("add", { children: [x, makeNode("mul", { children: [C(0.2), makeNode("cube", { children: [x] })] })] }),
        makeNode("add", { children: [C(1), makeNode("mul", { children: [C(0.5), sq] })] }),
      ],
    }),
    // shifted rsqrt — the SiLU/Lorentz denominator motif x/(a + √(b + x²))
    makeNode("pdiv", {
      children: [
        x,
        makeNode("add", {
          children: [C(0.5), makeNode("sqrt", { children: [makeNode("add", { children: [C(1), sq] })] })],
        }),
      ],
    }),
    // plain inverse gaussian envelope 1/√(1 + x²)
    makeNode("pdiv", { children: [C(1), makeNode("sqrt", { children: [makeNode("add", { children: [C(1), sq] })] })] }),
  ];
}
