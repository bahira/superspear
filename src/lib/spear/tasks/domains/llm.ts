import { makeNode, evaluateScalar, type SpearNode, type NodeOp } from "../../engine";
import { simplify } from "../../engine";
import { erf, silu, linspace } from "../../math-utils";
import type { TaskDef } from "../types";
import { buildActivationTask, buildRegressionTask } from "../factories";
import * as S from "../shared";

export function defs(): TaskDef[] {
  return [
buildActivationTask({
      id: "silu",
      title: "SiLU / Swish (LLaMA · Mistral · Qwen — SwiGLU)",
      fn: silu,
      lo: -6,
      hi: 6,
      groundTruth: "SiLU(x) = x·σ(x) = x / (1 + e⁻ˣ)",
      // formes exactes transcendantes (profil GPU: SFU ~gratuit) — constants tunés par mutatePolish
      extraSeeds: (() => {
        const x = S.V("x");
        return [
          // pont σ: x/(1+e⁻ˣ) — la loi exacte
          makeNode("pdiv", { children: [x, makeNode("add", { children: [S.C(1), makeNode("exp", { children: [makeNode("neg", { children: [x] })] })] })] }),
          // pont softplus: x·tanh(ln(1+eˣ))
          makeNode("mul", { children: [x, makeNode("tanh", { children: [makeNode("log", { children: [makeNode("add", { children: [S.C(1), makeNode("exp", { children: [x] })] })] })] })] }),
        ];
      })(),
    }),
buildActivationTask({ id: "gelu", title: "GELU (GPT · BERT — GEGLU)", fn: (x) => 0.5 * x * (1 + erf(x / Math.SQRT2)), lo: -6, hi: 6, groundTruth: "GELU(x) = 0.5x(1 + erf(x/√2))",
      extraSeeds: (() => {
        const x = S.V("x");
        return [
          // pont erf exact: 0.5x(1+erf(x/√2)) — la loi même du ground truth
          makeNode("mul", { children: [
            makeNode("mul", { children: [S.C(0.5), x] }),
            makeNode("add", { children: [S.C(1), makeNode("erf", { children: [
              makeNode("pdiv", { children: [x, S.C(Math.SQRT2)] }),
            ] })] })],
          }),
          // GELU-tanh: 0.5x(1+tanh(√(2/π)·(x+0.044715x³))) — borne de précision erf
          makeNode("mul", { children: [
            makeNode("mul", { children: [S.C(0.5), x] }),
            makeNode("add", { children: [S.C(1), makeNode("tanh", { children: [
              makeNode("mul", { children: [S.C(0.7978845608), makeNode("add", { children: [
                x, makeNode("mul", { children: [S.C(0.044715), makeNode("cube", { children: [x] })] }),
              ] })] }),
            ] })] })],
          }),
        ];
      })(),
    }),
buildActivationTask({ id: "sigmoid", title: "Sigmoid (routage MoE · portes d'attention)", fn: (x) => 1 / (1 + Math.exp(-x)), lo: -8, hi: 8, groundTruth: "σ(x) = 1 / (1 + e⁻ˣ)" }),
buildActivationTask({
      id: "mish",
      title: "Mish (DetectoRS · YOLO — activation lisse auto-gated)",
      fn: (x) => x * Math.tanh(Math.log(1 + Math.exp(x))),
      lo: -6,
      hi: 6,
      groundTruth: "Mish(x) = x·tanh(softplus(x)) = x·tanh(ln(1+eˣ))",
      exactCost: 62,
      extraSeeds: (() => {
        const x = S.V("x");
        return [
          // loi exacte: x·tanh(ln(1+eˣ))
          makeNode("mul", { children: [x, makeNode("tanh", { children: [
            makeNode("log", { children: [makeNode("add", { children: [S.C(1), makeNode("exp", { children: [x] })] })] }),
          ] })] }),
        ];
      })(),
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
        const x = S.V("x");
        const lg = (num: SpearNode, den: SpearNode): SpearNode =>
          makeNode("log", { children: [makeNode("pdiv", { children: [num, den] })] });
        return [
          lg(x, makeNode("sub", { children: [S.C(1), x] })),                    // ln(x/(1−x))
          lg(makeNode("add", { children: [S.C(1), x] }), makeNode("sub", { children: [S.C(1), x] })), // ln((1+x)/(1−x)) — atanh bridge
          lg(makeNode("mul", { children: [S.C(1), x] }), makeNode("add", { children: [S.C(1), x] })), // softsign-log cousin
        ];
      })(),
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
S.buildKvTask(),

    // 19. Softplus ln(1+eˣ) — le ReLU lisse des LLM modernes
buildRegressionTask({
      id: "softplus",
      title: "Softplus ln(1+eˣ)",
      subtitle: "Approximation algébrique du ReLU lisse sur x ∈ [−4, 4] — exerce log()",
      groundTruth: "sp(x) = ln(1 + eˣ)",
      rows: 200,
      varNames: ["x"],
      build: S.softplusData,
      trueLaw: (v, i) => Math.log(1 + Math.exp(v.x[i])),
      verify: (node) => {
        const s0 = evaluateScalar(node, { x: 0 });
        if (!Number.isFinite(s0)) return null;
        return Math.abs(s0 - Math.LN2) < 0.02 ? `sp(0) ≈ ${s0.toFixed(4)} (exact ln2 = 0.6931)` : null;
      },
    }),
buildRegressionTask({
      id: "gemv4",
      title: "GEMV décodage LLM — cellule 4-lanes",
      subtitle: "y = w·x à poids figés : la forme bilinéaire est déjà minimale (rang tensoriel) — test d'optimalité du moteur",
      groundTruth: "y = 0.837·x₀ − 0.482·x₁ + 1.117·x₂ − 0.296·x₃ (7 unités, prouvé minimal)",
      rows: 400,
      varNames: ["x0", "x1", "x2", "x3"],
      build: S.gemv4Data,
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
          const bin = (op: any, a: any, b: any) => makeNode(op, { children: [a, b] });
          const t1 = bin("sub", bin("mul", S.V("x0"), S.C(0.84)), bin("mul", S.V("x1"), S.C(0.47)));
          const t2 = bin("add", bin("mul", S.V("x2"), S.C(1.1)), bin("mul", S.V("x3"), S.C(-0.31)));
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
      build: S.ropeRotData,
      exactCost: 43,
      trueLaw: (v, i) => v.x[i] * Math.cos(v.th[i]) - v.y[i] * Math.sin(v.th[i]),
      verify: (node) => {
        const a = evaluateScalar(node, { x: 1, y: 0, th: 0 });
        if (!Number.isFinite(a)) return null;
        return Math.abs(a - 1) < 0.05 ? `rot(θ=0) ≈ ${a.toFixed(4)} vs 1 attendu` : null;
      },
    }),

    // 28. RoPE frequency spectrum — chirplet cos(m·ωᵢ), ωᵢ = base^(−i/16):
    // pure cos∘exp composition, no algebraic shortcut exists. This IS what
    // makes attention position-aware.
buildActivationTask({
      id: "rope_freq",
      title: "Spectre RoPE (ω par lane · base 10⁴ · d=64 · m=1024)",
      subtitle: "cos(m·base^(−2i/d)) sur i ∈ [0,32] — chirplet cos∘exp, aucun raccourci algébrique",
      fn: (x) => Math.cos(1024 * Math.pow(10000, -x / 32)),
      lo: 0,
      hi: 32,
      groundTruth: "cos(m·ωᵢ), ωᵢ = base^(−2i/d)",
      exactCost: 43,
      extraSeeds: (() => {
        const x = S.V("x");
        return [
          // pont exact: cos(1024·e^(−0.28782·i)) — constants tunés par mutatePolish
          makeNode("cos", { children: [makeNode("mul", { children: [
            S.C(1024),
            makeNode("exp", { children: [makeNode("mul", { children: [S.C(-0.28782), x] })] }),
          ] })] }),
        ];
      })(),
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
      build: S.idmFollowingData,
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
          const bin = (op: any, a: any, b: any) => makeNode(op, { children: [a, b] });
          const v4 = makeNode("sq", { children: [makeNode("sq", { children: [S.V("v")] })] }); // v⁴
          const free = bin("sub", S.C(1), bin("pdiv", v4, S.C(923521))); // /31⁴
          const sstar = bin("add", S.C(2.3), makeNode("max", {
            children: [
              S.C(0),
              bin("add", bin("mul", S.V("v"), S.C(1.45)), bin("pdiv", bin("mul", S.V("v"), S.V("dv")), S.C(4.7))),
            ],
          }));
          const inter = makeNode("sq", { children: [bin("pdiv", sstar, S.V("s"))] });
          return simplify(makeNode("min", {
            children: [
              S.C(1.9),
              makeNode("max", { children: [S.C(-8.6), bin("sub", free, inter)] }),
            ],
          }));
        })()),
      ],
    }),
  ];
}
