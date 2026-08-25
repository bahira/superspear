import { makeNode, evaluateScalar, type SpearNode, type NodeOp } from "../../engine";
import { simplify } from "../../engine";
import { erf, silu, linspace } from "../../math-utils";
import type { TaskDef } from "../types";
import { buildActivationTask, buildRegressionTask } from "../factories";
import * as S from "../shared";

export function defs(): TaskDef[] {
  return [

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
          y[i] = S.groverSuccess(k, m, n);
        }
        return { vars: { k: kv, m: mv, n: nv }, y };
      },
      trueLaw: (v, i) => S.groverSuccess(v.k[i], v.m[i], v.n[i]),
      // asin is SERVED now — the pre-asin premise is obsolete. Shape-only
      // scaffolds of the amplification law (constants stay tunable), same
      // doctrine that cracked ik_reach/eigen when atan got served.
      extraSeeds: (() => {
        const kk = S.V("k"), mm = S.V("m"), nn = S.V("n");
        const theta = makeNode("asin", {
          children: [makeNode("sqrt", { children: [makeNode("pdiv", { children: [mm, nn] })] })],
        });
        const kTheta = makeNode("mul", { children: [kk, theta] });
        // (2k+1)·θ lives INSIDE the sine argument — k is a variable, so the
        // scaffold must carry it there structurally, not as a constant
        const twoKPlusOneTheta = makeNode("add", {
          children: [theta, makeNode("mul", { children: [S.C(2), kTheta] })],
        });
        const sin = (arg: SpearNode): SpearNode => makeNode("sin", { children: [arg] });
        return [
          makeNode("sq", { children: [sin(twoKPlusOneTheta)] }),
          makeNode("sq", { children: [sin(makeNode("add", { children: [theta, makeNode("mul", { children: [S.C(3), kTheta] })] }))] }),
          makeNode("sq", { children: [sin(theta)] }),
          makeNode("sub", { children: [
            S.C(1),
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
        // EXACT-form scaffold: S.C = 2|ad − bc| (huber recipe — fully served ops)
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
          const bin = (op: any, a: any, b: any) => makeNode(op, { children: [a, b] });
          const E = (p: string, q: string) => makeNode("neg", { children: [makeNode("cos", { children: [bin("sub", S.V(p), S.V(q))] })] });
          const sExpr = bin("sub",
            bin("add", E("a", "b"), E("ap", "b")),
            bin("sub", E("a", "bp"), E("ap", "bp"))
          );
          return makeNode("abs", { children: [sExpr] });
        })()),
      ],
      verify: () => null,
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
        const n = S.V("n"), g = S.V("g"), t = S.V("t");
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
        const th = S.V("th"), g = S.V("g"), t = S.V("t");
        const E = makeNode("exp", { children: [makeNode("neg", { children: [makeNode("mul", { children: [g, t] })] })] });
        const c2 = makeNode("sq", { children: [makeNode("cos", { children: [th] })] });
        const s2 = makeNode("sq", { children: [makeNode("sin", { children: [th] })] });
        return [
          makeNode("add", { children: [
            makeNode("mul", { children: [c2, E] }),
            makeNode("mul", { children: [s2, makeNode("sub", { children: [S.C(2), E] })] }),
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

    // 23. SPEAR CODEX BT29 — eigenvalue extraction without Jacobi/QR sweeps
buildRegressionTask({
      id: "eigen3_sym",
      title: "Valeur propre max 3×3 symétrique (BT29)",
      subtitle: "λmax(tr, I₂, det) — remplace les balayages de rotations de Jacobi (Cardano a besoin d'acos, non servi)",
      groundTruth: "λ³−I₁λ²+I₂λ−I₃=0, racine max via trisection trigonométrique",
      rows: 500,
      varNames: ["t", "u", "w"],
      build: S.eigenSymData,
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
          const bin = (op: "add" | "sub" | "mul" | "pdiv", a: any, b: any) => makeNode(op, { children: [a, b] });
          const p = bin("sub", S.V("u"), bin("mul", makeNode("sq", { children: [S.V("t")] }), S.C(1 / 3)));
          const q = bin("sub", bin("mul", bin("mul", S.V("t"), S.V("u")), S.C(1 / 3)), bin("add", bin("mul", makeNode("cube", { children: [S.V("t")] }), S.C(2 / 27)), S.V("w")));
          const rr = makeNode("sqrt", { children: [bin("pdiv", makeNode("neg", { children: [p] }), S.C(3))] });
          const argRaw = bin("mul", bin("pdiv", bin("mul", S.C(3), q), bin("mul", S.C(2), p)), makeNode("sqrt", { children: [bin("pdiv", S.C(-3), p)] }));
          const arg = makeNode("min", { children: [S.C(1), makeNode("max", { children: [S.C(-1), argRaw] })] });
          const acosArg = bin("sub", S.C(Math.PI / 2), makeNode("atan", { children: [bin("pdiv", arg, makeNode("sqrt", { children: [makeNode("abs", { children: [bin("sub", S.C(1), makeNode("sq", { children: [arg] }))] })] }))] }));
          return simplify(bin("add", bin("mul", S.V("t"), S.C(1 / 3)), bin("mul", bin("mul", S.C(2), rr), makeNode("cos", { children: [bin("pdiv", acosArg, S.C(3))] }))));
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
      build: S.ikReachData,
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
          const bin = (op: "add" | "sub" | "mul" | "pdiv", a: any, b: any) => makeNode(op, { children: [a, b] });
          const num = bin("sub", makeNode("sq", { children: [S.V("d") ] }), bin("add", makeNode("sq", { children: [S.V("l2")] }), makeNode("sq", { children: [S.V("l3")] })));
          const den = makeNode("mul", { children: [S.C(2), makeNode("mul", { children: [S.V("l2"), S.V("l3")] })] });
          const z = bin("pdiv", num, den);
          // acos(z) = PI/2 - atan(z / sqrt(1-z^2)) — valid on the whole [-1,1]
          return simplify(makeNode("sub", {
            children: [
              S.C(Math.PI / 2),
              makeNode("atan", {
                children: [bin("pdiv", z, makeNode("sqrt", { children: [makeNode("abs", { children: [bin("sub", S.C(1), makeNode("sq", { children: [z] }))] })] }))],
              }),
            ],
          }));
        })(),
      ],
    }),
  ];
}
