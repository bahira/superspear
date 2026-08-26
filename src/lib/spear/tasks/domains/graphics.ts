import { makeNode, evaluateScalar, type SpearNode, type NodeOp } from "../../engine";
import { simplify } from "../../engine";
import { erf, silu, linspace } from "../../math-utils";
import type { TaskDef } from "../types";
import { buildActivationTask, buildRegressionTask } from "../factories";
import * as S from "../shared";

export function defs(): TaskDef[] {
  return [

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

    // Blackbody red channel: color temperature -> normalized red. Piecewise
    // power law (Tanner Helland fit) — used by every physically-based light.
buildActivationTask({
      id: "blackbody_r",
      title: "Corps noir — canal rouge (éclairage PBR)",
      subtitle: "canal R normalisé vs température [1500K, 12000K] — loi en puissance par morceaux",
      fn: (t) => S.blackbodyRed(t),
      lo: 1500,
      hi: 12000,
      groundTruth: "R(T) — fit Tanner Helland",
      exactCost: 25,
    }),
buildActivationTask({
      id: "blackbody_g",
      title: "Corps noir — canal vert (éclairage PBR)",
      subtitle: "canal G normalisé vs température — branche ln puis branche puissance",
      fn: (t) => S.blackbodyGreen(t),
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
      fn: (t) => S.blackbodyBlue(t),
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
      fn: (x) => S.narkowiczAces(x),
      lo: 0,
      hi: 1.5,
      groundTruth: "aces(x) = clamp(x(2.51x+0.03)/(x(2.43x+0.59)+0.14))",
      exactCost: 8,
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

    // 9. Noyau de blur gaussien 1-D — remplacer exp() par forme fermée
buildRegressionTask({
      id: "gaussian_kernel",
      title: "Blur gaussien · noyau exp(−x²/2σ²)",
      subtitle: "Approximation du kernel gaussien (σ=1) sur x ∈ [−3, 3]",
      groundTruth: "G(x) = exp(−x²/2), σ=1",
      rows: 200,
      varNames: ["x"],
      build: S.gaussianKernelData,
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
      build: S.betaScheduleData,
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
      build: S.interpWeightData,
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
      build: S.temporalGradData,
      trueLaw: (v, i) => v.b[i] - v.a[i],
      verify: (node) => {
        const c = evaluateScalar(node, { a: 0.5, b: 0.53 });
        if (!Number.isFinite(c)) return null;
        const err = Math.abs(c - 0.03) / 0.03;
        return err < 0.2 ? `∂I/∂t ≈ ${c.toFixed(4)} (exact 0.0300)` : null;
      },
    }),

    // Bilateral range kernel × spatial Lorentzian — the edge-aware weight
    // behind every real-time denoiser/sharpen (SVGF, bilateral grids).
    // W(Δ) = exp(−Δ²/(2σr²)) / (1+Δ²/(2σs²)), σr=0.1, σs=0.5.
buildActivationTask({
      id: "bilateral_weight",
      title: "Poids bilatéral (débruitage temps réel · SVGF)",
      subtitle: "W(Δ)=exp(−Δ²/0.02)/(1+Δ²/0.5) sur Δ ∈ [0,1] — gaussienne×lorentzienne",
      fn: (x) => Math.exp(-(x * x) / 0.02) / (1 + (x * x) / 0.5),
      lo: 0,
      hi: 1,
      groundTruth: "W(Δ) = exp(−Δ²/(2σr²))/(1+Δ²/(2σs²))",
      exactCost: 45,
      extraSeeds: (() => {
        const x = S.V("x");
        return [
          // pont exact: exp(−x²/0.02)/(1+x²/0.5)
          makeNode("pdiv", { children: [
            makeNode("exp", { children: [makeNode("neg", { children: [
              makeNode("pdiv", { children: [makeNode("sq", { children: [x] }), S.C(0.02)] }),
            ] })] }),
            makeNode("add", { children: [
              S.C(1),
              makeNode("pdiv", { children: [makeNode("sq", { children: [x] }), S.C(0.5)] }),
            ] }),
          ] }),
        ];
      })(),
    }),
  ];
}
