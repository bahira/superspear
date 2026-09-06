import { makeNode, evaluateScalar, type SpearNode, type NodeOp } from "../../engine";
import { simplify } from "../../engine";
import { erf, silu, linspace } from "../../math-utils";
import type { TaskDef } from "../types";
import { buildActivationTask, buildRegressionTask } from "../factories";
import * as S from "../shared";

export function defs(): TaskDef[] {
  return [

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
          const bin = (op: NodeOp, a: any, b: any) => makeNode(op, { children: [a, b] });
          return makeNode("max", {
            children: [
              S.C(0),
              bin("sub", bin("mul", P, S.C(1.05)), bin("pdiv", bin("sub", S.C(1.02), P), B)),
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
          const c = S.bsCall(s, k, t, vol);
          cv[i] = c; sv[i] = s; kv[i] = k; tv[i] = t;
          y[i] = S.impliedVol(c, s, k, t);
        }
        return { vars: { c: cv, s: sv, k: kv, t: tv }, y };
      },
      trueLaw: (v, i) => S.impliedVol(v.c[i], v.s[i], v.k[i], v.t[i]),
      // no closed form exists, but the HUMAN art is seedable: Brenner–
      // Subrahmanyam ATMF skeleton and its Corrado–Miller-style corrections,
      // shapes only (generic constants).
      extraSeeds: (() => {
        const c = S.V("c"), s = S.V("s"), k = S.V("k"), t = S.V("t");
        const cs = makeNode("pdiv", { children: [c, s] });
        const sqrtInvT = makeNode("sqrt", { children: [makeNode("pdiv", { children: [S.C(1), t] })] });
        const bs = makeNode("mul", { children: [S.C(2.5), makeNode("mul", { children: [cs, sqrtInvT] })] }); // √(2π)·(c/s)/√t
        const ksTerm = makeNode("mul", { children: [makeNode("pdiv", { children: [k, s] }), sqrtInvT] });
        // Corrado–Miller 1996 (grade production historique): correction du terme
        // forward sur l'ATMF — erreurs < 1 point de vol près de la money-ness.
        // Shapes only: les constantes restent tunables par mutatePolish.
        const halfSk = makeNode("pdiv", { children: [makeNode("sub", { children: [s, k] }), S.C(2)] });
        const skSum = makeNode("add", { children: [s, k] });
        const cm = makeNode("mul", { children: [
          sqrtInvT,
          makeNode("pdiv", { children: [
            makeNode("add", { children: [
              makeNode("sub", { children: [c, halfSk] }),
              makeNode("mul", { children: [S.C(0.6366), makeNode("pdiv", { children: [makeNode("sq", { children: [halfSk] }), skSum] })] }),
            ] }),
            skSum,
          ] }),
        ] });
        // Manaster–Koehler: départ canonique √(2|ln(K/S)|/T) — porte la
        // structure log-moneyness qui domine hors de la money-ness.
        const L = makeNode("log", { children: [makeNode("pdiv", { children: [k, s] })] });
        const mk = makeNode("sqrt", { children: [makeNode("pdiv", { children: [
          makeNode("mul", { children: [S.C(2), makeNode("abs", { children: [L] })] }),
          t,
        ] })] });
        // Brenner corrigé en log-moneyness: σ₀·(1 + a·L·√T + b·L²·√T) — les
        // erreurs du B-S brut sont quadratiques en ln(K/S), le GP règle a,b.
        const corr = makeNode("mul", { children: [
          bs,
          makeNode("add", { children: [
            S.C(1),
            makeNode("add", { children: [
              makeNode("mul", { children: [S.C(0.3), makeNode("pdiv", { children: [L, sqrtInvT] })] }),
              makeNode("pdiv", { children: [makeNode("sq", { children: [L] }), sqrtInvT] }),
            ] }),
          ] }),
        ] });
        // DIAGNOSED GAP (bucketed residuals of the incumbent champion):
        //   OTM/lowvol mse 3.10e-2, ITM/lowvol 2.47e-2  vs  ATM/lowvol 1.74e-3
        // — an 18x spread. The incumbent is a Corrado-Miller form, which is an
        // AT-THE-MONEY approximation: its correction term is linear in (s-k),
        // so it degrades exactly where |log(s/k)| grows and vol is small.
        //
        // Every shape already in this pool ADDS or MULTIPLIES a moneyness term
        // onto the ATM skeleton. None lets moneyness act as a DENOMINATOR, yet
        // that is the structure of the BS inversion away from the money
        // (sigma ~ price / vega, and vega collapses as |log(s/k)| grows). We
        // seed that missing algebraic brick — shape only, constants tunable.
        const L2 = makeNode("sq", { children: [L] });
        // sigma0 / (1 + a*L^2)  — vega-collapse damping
        const damp = makeNode("pdiv", {
          children: [
            bs,
            makeNode("add", {
              children: [S.C(1), makeNode("mul", { children: [S.C(2), L2] })],
            }),
          ],
        });
        // sqrt(sigma0^2 + a*L^2/t) — quadrature blend, the Manaster-Koehler
        // floor fused with the ATM skeleton instead of competing with it
        const blend = makeNode("sqrt", {
          children: [
            makeNode("abs", {
              children: [
                makeNode("add", {
                  children: [
                    makeNode("sq", { children: [bs] }),
                    makeNode("mul", { children: [S.C(2), makeNode("pdiv", { children: [L2, t] })] }),
                  ],
                }),
              ],
            }),
          ],
        });
        // SECOND DIAGNOSED GAP. Residual bucketing showed the incumbent (a
        // refined Corrado-Miller) is weakest at LOW vol away from the money.
        // Probing modulations of the CM core showed the effective lever is not
        // moneyness at all but the NORMALISED PRICE u = c/s: CM*(1 + a*u) alone
        // cuts MSE from 1.32e-2 to 1.01e-2, and combining it with a mild
        // moneyness denominator reaches 7.4e-3. Neither shape was expressible
        // from the existing pool, which only ever added moneyness terms.
        // Shapes only — a, b stay tunable.
        const uNorm = makeNode("pdiv", { children: [c, s] });          // c/s
        const rMoney = makeNode("sub", { children: [makeNode("pdiv", { children: [s, k] }), S.C(1)] }); // s/k - 1
        const cmScaled = makeNode("mul", {
          children: [cm, makeNode("add", { children: [S.C(1), makeNode("mul", { children: [S.C(2), uNorm] })] })],
        });
        const cmScaledDamped = makeNode("pdiv", {
          children: [
            cmScaled,
            makeNode("add", {
              children: [S.C(1), makeNode("mul", { children: [S.C(0.3), makeNode("sq", { children: [rMoney] })] })],
            }),
          ],
        });
        // CM * (1 + a*u/sqrt(t)) — the same lever carried by the time scaling
        const cmScaledT = makeNode("mul", {
          children: [
            cm,
            makeNode("add", {
              children: [S.C(1), makeNode("mul", { children: [S.C(2), makeNode("mul", { children: [uNorm, sqrtInvT] })] })],
            }),
          ],
        });
        return [
          bs,
          cm,
          mk,
          corr,
          damp,
          blend,
          cmScaled,
          cmScaledDamped,
          cmScaledT,
          simplify(makeNode("add", { children: [bs, makeNode("mul", { children: [S.C(1), ksTerm] })] })),
          makeNode("mul", { children: [bs, makeNode("add", { children: [S.C(1), makeNode("pdiv", { children: [makeNode("sq", { children: [cs] }), t] })] })] }),
        ];
      })(),
      verify: (node) => {
        const iv = evaluateScalar(node, { c: S.bsCall(100, 100, 0.5, 0.35), s: 100, k: 100, t: 0.5 });
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
      fn: (p) => S.acklamProbit(p),
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

    // 3. Prime d'un call européen — task de régression (variables σ)
buildRegressionTask({
      id: "european_call",
      title: "Prime d'un call européen",
      subtitle: "Approximation de la formule Black‑Scholes at‑the‑money S.C(σ)",
      groundTruth: "C ≈ 0.4·S·σ·√T (ATM, τ petit)",
      rows: 200,
      varNames: ["sigma"],
      build: S.europeanCallData,
      trueLaw: (v, i) => 100 * (0.4 * v.sigma[i] + 0.16 * v.sigma[i] * v.sigma[i]),
      verify: (node) => {
        const c = evaluateScalar(node, { sigma: 0.2 });
        if (!Number.isFinite(c)) return null;
        const exact = 100 * (0.4 * 0.2 + 0.16 * 0.2 * 0.2);
        const err = Math.abs(c - exact) / exact;
        return err < 0.1 ? `erreur relative ${(err * 100).toFixed(1)} %` : null;
      },
    }),

    // Black-Scholes d₁(σ) — S=110, K=100, r=5%, T=0.25:
    // d₁ = (ln(S/K)+(r+σ²/2)T)/(σ√T) = c/σ + σ/4 — hyperbole+linéaire,
    // le cœur de tout pricer, zéro SFU nécessaire.
buildActivationTask({
      id: "bs_d1_sigma",
      title: "Black-Scholes d₁(σ) (S/K=1.1 · r=5% · T=3m)",
      subtitle: "d₁ = (ln(S/K)+(r+σ²/2)T)/(σ√T) sur σ ∈ [0.08,1.2] — forme rationnelle pure",
      fn: (x) => (Math.log(1.1) + (0.05 + (x * x) / 2) * 0.25) / (x * 0.5),
      lo: 0.08,
      hi: 1.2,
      groundTruth: "d₁(σ) = (ln(S/K)+(r+σ²/2)T)/(σ√T)",
      exactCost: 9,
      extraSeeds: (() => {
        const x = S.V("x");
        return [
          // pont exact: c/x + x/4
          makeNode("add", { children: [
            makeNode("pdiv", { children: [S.C(0.21562036), x] }),
            makeNode("mul", { children: [S.C(0.25), x] }),
          ] }),
        ];
      })(),
    }),

    // Black-Scholes d₂(σ) = d₁ − σ√T = c/σ − σ/4 — même famille, signe opposé
buildActivationTask({
      id: "bs_d2_sigma",
      title: "Black-Scholes d₂(σ) (S/K=1.1 · r=5% · T=3m)",
      subtitle: "d₂ = d₁ − σ√T sur σ ∈ [0.08,1.2] — hyperbole−linéaire, moneyness→probabilité",
      fn: (x) => (Math.log(1.1) + (0.05 + (x * x) / 2) * 0.25) / (x * 0.5) - 0.5 * x,
      lo: 0.08,
      hi: 1.2,
      groundTruth: "d₂(σ) = d₁(σ) − σ√T",
      exactCost: 11,
      extraSeeds: (() => {
        const x = S.V("x");
        return [
          // pont exact: c/x − x/4
          makeNode("add", { children: [
            makeNode("pdiv", { children: [S.C(0.21562036), x] }),
            makeNode("neg", { children: [makeNode("mul", { children: [S.C(0.25), x] })] }),
          ] }),
        ];
      })(),
    }),
  ];
}
