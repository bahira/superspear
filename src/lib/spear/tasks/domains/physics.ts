import { makeNode, evaluateNode, evaluateScalar, type SpearNode, type NodeOp } from "../../engine";
import { simplify } from "../../engine";
import { erf, silu, linspace } from "../../math-utils";
import type { TaskDef } from "../types";
import { buildActivationTask, buildRegressionTask } from "../factories";
import * as S from "../shared";

export function defs(): TaskDef[] {
  return [

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

    // erf: THE probability kernel (GELU-exact grade). exp is SERVED here, so
    // the search may mix rationals with exponentials — A&S-style hybrids.
buildActivationTask({
      id: "erf_prob",
      title: "Erf probabiliste (noyaux de probabilité · GELU-exact)",
      subtitle: "erf(x) sur [-3,3] — hybrides rationnel+exp permis (style Abramowitz-Stegun)",
      fn: (x) => S.erfImpl(x),
      lo: -3,
      hi: 3,
      groundTruth: "erf(x)",
      exactCost: 26,
      // l'op erf est servi depuis la v1.8 — la loi exacte est dans le pool
      extraSeeds: [makeNode("erf", { children: [S.V("x")] })],
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
      fn: (x) => S.besselJ0(x),
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
                      makeNode("sub", { children: [S.C(1), makeNode("mul", { children: [S.C(2.25), h2] })] }),
                      makeNode("mul", { children: [S.C(1.2656208), h4] }),
                    ],
                  }),
                  makeNode("mul", { children: [S.C(0.3163866), h6] }),
                ],
              }),
              // + 0.0444479·h⁸ − ... folded via sq chain
              makeNode("mul", { children: [S.C(0.0444479), makeNode("sq", { children: [h4] })] }),
            ],
          });
        })()),
        // even-series tail for contrast
        simplify(makeNode("pdiv", { children: [makeNode("var", { name: "x" }), makeNode("const", { value: 3 })] })),
      ],
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
          y[i] = S.besselI0e(x);
        }
        return { vars: { x: xv }, y };
      },
      trueLaw: (v, i) => S.besselI0e(v.x[i]),
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
          y[i] = S.ellipticK(mm);
        }
        return { vars: { m: mv }, y };
      },
      trueLaw: (v, i) => S.ellipticK(v.m[i]),
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
          y[i] = S.solveKepler(M, ecc);
        }
        return { vars: { M: Mv, e: ev }, y };
      },
      trueLaw: (v, i) => S.solveKepler(v.M[i], v.e[i]),
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
          const bin = (op: NodeOp, a: any, b: any) => makeNode(op, { children: [a, b] });
          return simplify(bin("mul", S.C(710), bin("pdiv", bin("add", S.C(345), VO), bin("sub", S.C(338), VS))));
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

    // ---- third wave: wave 3 — companions, tonemap, physics glue ------------
    // Bessel J1: FM modulation index, vibrating membranes' antisymmetric modes.
    // Companion to bessel_j0; same "no SFU serves it" story.
buildActivationTask({
      id: "bessel_j1",
      title: "Bessel J₁ (FM · membranes antisym)",
      subtitle: "J₁(x) sur [0,6] — série 16 termes en référence ; forme close découverte = nouvel outil",
      fn: (x) => S.besselJ1(x),
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
      fn: (x) => S.besselJ2(x),
      lo: 0,
      hi: 6,
      groundTruth: "J₂(x)",
      exactCost: 40,
    }),
buildRegressionTask({
      id: "free_fall",
      title: "Loi de chute libre",
      subtitle: "48 mesures bruitées (σ ≈ 2 cm) de distance vs. temps",
      groundTruth: "d = ½·g·t² avec g = 9.81 m·s⁻²",
      rows: 48,
      varNames: ["t"],
      build: S.freeFallData,
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
      build: S.keplerData,
      trueLaw: (v, i) => Math.pow(v.a[i], 1.5),
      verify: (node) => {
        const at4 = evaluateScalar(node, { a: 4 });
        if (!Number.isFinite(at4)) return null;
        const err = Math.abs(at4 - 8) / 8;
        return err < 0.01 ? `T(4 UA) = ${at4.toFixed(3)} an (théorie 8.000) — exposant 3/2 retrouvé` : null;
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
      build: S.dampedPendulumData,
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
      build: S.lambertWData,
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
      build: S.rcCircuitData,
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
      build: S.layernormScaleData,
      trueLaw: (v, i) => 1 / Math.sqrt(v.x[i]),
      verify: (node) => {
        const c = evaluateScalar(node, { x: 1 });
        if (!Number.isFinite(c)) return null;
        const err = Math.abs(c - 1) / 1;
        return err < 0.05 ? `rsqrt(1) ≈ ${c.toFixed(4)} (exact 1.0000)` : null;
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
      build: S.lorentzData,
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
      build: S.hillData,
      trueLaw: (v, i) => {
        const c3 = v.c[i] * v.c[i] * v.c[i];
        return c3 / (1 + c3);
      },
      extraSeeds: [
        // EXACT-form scaffold: c³/(1+c³) = pdiv(cube(c), add(S.C(1), cube(c)))
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
      build: S.kerrDeflectionData,
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
      build: S.lennardJonesData,
      trueLaw: (v, i) => {
        const inv6 = (1 / v.r[i]) ** 6;
        return 4 * (inv6 * inv6 - inv6);
      },
      verify: (node) => {
        const vmin = evaluateScalar(node, { r: Math.pow(2, 1 / 6) });
        if (!Number.isFinite(vmin)) return null;
        return Math.abs(vmin + 1) < 0.15 ? `Puits LJ : S.V(2^(1/6)) ≈ ${vmin.toFixed(3)} (exact −1)` : null;
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
      build: S.dampedOscillationData,
      trueLaw: (v, i) => Math.exp(-v.t[i] / 4) * Math.cos(3 * v.t[i]),
      // ALU-only munitions: Padé envelopes for the exp decay + generic pole
      // rationals (poles are how algebra creates alternations without trig).
      // Shape-only, constants tunable.
      extraSeeds: (() => {
        const t = S.V("t");
        const env = makeNode("pdiv", {
          children: [S.C(1), makeNode("add", { children: [S.C(1), makeNode("mul", { children: [S.C(0.25), t] })] })],
        });
        return [
          env,
          makeNode("sq", { children: [env] }),
          makeNode("pdiv", {
            children: [
              makeNode("sub", { children: [makeNode("mul", { children: [S.C(0.5), makeNode("sq", { children: [t] })] }), S.C(1)] }),
              makeNode("add", { children: [makeNode("sq", { children: [t] }), S.C(1)] }),
            ],
          }),
          makeNode("pdiv", {
            children: [
              makeNode("mul", { children: [t, makeNode("sub", { children: [S.C(2), t] })] }),
              makeNode("add", { children: [S.C(1), makeNode("sq", { children: [t] })] }),
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
      build: S.logisticGrowthData,
      trueLaw: (v, i) => 1 / (1 + Math.exp(-2 * (v.t[i] - 2))),
      verify: (node) => {
        const mid = evaluateScalar(node, { t: 2 });
        if (!Number.isFinite(mid)) return null;
        return Math.abs(mid - 0.5) < 0.03 ? `y(t₀) ≈ ${mid.toFixed(3)} (exact 0.5)` : null;
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
      build: S.kdvSolitonData,
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
      build: S.kerrSpinData,
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
      subtitle: "u(θ,θ̇) : energy-shaping → Lyapunov catch via σ S.C∞, saturé ±2 (θ ∈ [−π,π], θ̇ ∈ [−6,6])",
      groundTruth: "u* = clamp((1−w)·u_swing + w·u_catch, ±2), w = σ(10.18(cosθ − 0.7))",
      rows: 500,
      varNames: ["th", "d"],
      build: S.pendulumHybridData,
      trueLaw: (v, i) => {
        const c = Math.cos(v.th[i]);
        const EErr = 0.5 * v.d[i] * v.d[i] + 6 * (1 - c) - 12;
        const w = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, 10.1786 * (c - 0.7)))));
        const uSwing = -4.3278 * v.d[i] * EErr * c;
        const uCatch = -(1.7222 * Math.sin(v.th[i]) + 8.0402 * v.d[i]);
        return Math.min(2, Math.max(-2, (1 - w) * uSwing + w * uCatch));
      },
      // loi du contrôleur entièrement exprimable dans la grammaire servie
      // (cos/sin/exp/min/max) — seed shape-only, constants tunables
      extraSeeds: (() => {
        const th = S.V("th"), d = S.V("d");
        const c = makeNode("cos", { children: [th] });
        const s = makeNode("sin", { children: [th] });
        // erreur d'énergie: ½θ̇² + 6(1−cosθ) − 12
        const eErr = makeNode("add", { children: [
          makeNode("mul", { children: [S.C(0.5), makeNode("sq", { children: [d] })] }),
          makeNode("sub", { children: [
            makeNode("mul", { children: [S.C(6), makeNode("sub", { children: [S.C(1), c] })] }),
            S.C(12),
          ] }),
        ] });
        const uSwing = makeNode("mul", { children: [
          S.C(-4.33),
          makeNode("mul", { children: [d, makeNode("mul", { children: [eErr, c] })] }),
        ] });
        // blend Lyapunov: w = σ(−k·(cosθ−0.7))
        const w = makeNode("pdiv", { children: [
          S.C(1),
          makeNode("add", { children: [
            S.C(1),
            makeNode("exp", { children: [makeNode("neg", { children: [makeNode("mul", { children: [
              S.C(10.18),
              makeNode("sub", { children: [c, S.C(0.7)] }),
            ] })] })] }),
          ] }),
        ] });
        const uCatch = makeNode("neg", { children: [makeNode("add", { children: [
          makeNode("mul", { children: [S.C(1.72), s] }),
          makeNode("mul", { children: [S.C(8.04), d] }),
        ] })] });
        const blend = makeNode("add", { children: [
          makeNode("mul", { children: [makeNode("sub", { children: [S.C(1), w] }), uSwing] }),
          makeNode("mul", { children: [w, uCatch] }),
        ] });
        return [makeNode("max", { children: [
          S.C(-2),
          makeNode("min", { children: [S.C(2), blend] }),
        ] })];
      })(),
      verify: (node) => {
        const up = evaluateScalar(node, { th: 0.5, d: 1 });
        if (!Number.isFinite(up)) return null;
        // saturated zone check: high energy far from equilibrium must clamp
        const sat = evaluateScalar(node, { th: 3, d: 6 });
        const okSat = Number.isFinite(sat) && sat >= -2 && sat <= 2;
        return okSat ? `Loi bornée : u(0.5,1) ≈ ${up.toFixed(3)}, saturation respectée` : null;
      },
    }),
  ];
}
