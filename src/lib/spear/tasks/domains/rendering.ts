// RENDERING DOMAIN — GPU hot-loop kernels for p5.js / three.js pipelines.
// Every law here runs per-fragment or per-object in real engines, so cost
// units translate directly into frame budget.
import { makeNode, type SpearNode } from "../../engine";
import type { TaskDef } from "../types";
import { buildActivationTask, buildRegressionTask } from "../factories";
import { V, C } from "../shared";

// golden-ratio sampler shared by the regression builders below
function sample(rows: number, lo: number, hi: number, fn: (x: number) => number): { vars: Record<string, Float64Array>; y: Float64Array } {
  const xs = new Float64Array(rows);
  const ys = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    xs[i] = lo + (hi - lo) * ((i * 0.6180339887) % 1);
    ys[i] = fn(xs[i]);
  }
  return { vars: { x: xs }, y: ys };
}

export function defs(): TaskDef[] {
  return [
    // Fresnel-Schlick: F = F0 + (1-F0)(1-cosθ)^5 — every PBR fragment pays
    // this power. (1-u)^5 is an exact sq-chain: the engine must rediscover it.
    buildActivationTask({
      id: "fresnel_schlick",
      title: "Fresnel-Schlick (réflexion PBR)",
      subtitle: "F(u) = 0.04 + 0.96·(1−u)⁵ sur u ∈ [0,1] — payé par chaque fragment",
      fn: (u) => 0.04 + 0.96 * Math.pow(1 - u, 5),
      lo: 0,
      hi: 1,
      groundTruth: "F = 0.04 + 0.96·(1−u)⁵",
      exactCost: 12,
      // pow-chain scaffold: (1−u)⁵ = sq∘sq·t — the exact deployment shape
      extraSeeds: (() => {
        const u = V("u");
        const t = makeNode("sub", { children: [C(1), u] });
        const p5 = makeNode("mul", { children: [makeNode("sq", { children: [makeNode("sq", { children: [t] })] }), t] });
        return [makeNode("add", { children: [C(0.04), makeNode("mul", { children: [C(0.96), p5] })] })];
      })(),
    }),
    // Perlin's quintic fade — THE noise interpolation curve (smootherstep).
    // Horner form t³·(t·(6t−15)+10) is ~8 units: optimality-test material.
    buildActivationTask({
      id: "smootherstep",
      title: "Smootherstep (fade Perlin)",
      subtitle: "6t⁵ − 15t⁴ + 10t³ sur [0,1] — la C²-interpolation du bruit de Perlin",
      fn: (t) => t * t * t * (t * (6 * t - 15) + 10),
      lo: 0,
      hi: 1,
      groundTruth: "fade(t) = 6t⁵−15t⁴+10t³",
      exactCost: 9,
    }),
    // p5.js / easings.net Back ease-out with the magic overshoot constant
    // c1 = 1.70158 (c3 = c1+1). Pure polynomial — can refinement find 1.70158?
    buildActivationTask({
      id: "back_ease_out",
      title: "Back ease-out (easing p5.js)",
      subtitle: "1 + c₃(t−1)³ + c₁(t−1)², c₁ = 1.70158 — l'overshoot iconique",
      fn: (t) => {
        const s = t - 1;
        return 1 + 2.70158 * s * s * s + 1.70158 * s * s;
      },
      lo: 0,
      hi: 1,
      groundTruth: "easeOutBack = 1 + c₃s³ + c₁s², s = t−1",
      exactCost: 11,
    }),
    // Shadow-acne slope-scaled bias: tan(acos(z)) = √(1−z²)/z — pure rsqrt
    // family, exact form ~8 units.
    buildActivationTask({
      id: "bias_slope",
      title: "Biais de pente (acné d'ombre)",
      subtitle: "tan(acos(z)) = √(1−z²)/z sur z ∈ [0.3,1] — le biais des shadow maps",
      fn: (z) => Math.sqrt(1 - z * z) / z,
      lo: 0.3,
      hi: 1,
      groundTruth: "bias = √(1−z²)/z",
      exactCost: 8,
      // rsqrt-family scaffold: the exact closed form, constants tunable
      extraSeeds: [
        makeNode("pdiv", {
          children: [
            makeNode("sqrt", { children: [makeNode("sub", { children: [C(1), makeNode("sq", { children: [V("x")] })] })] }),
            V("x"),
          ],
        }),
      ],
    }),
    // Rayleigh phase function — sky scattering, one line of every atmosphere
    // shader. Constant × (1+cos²θ): exact at ~5 units.
    buildActivationTask({
      id: "rayleigh_phase",
      title: "Phase de Rayleigh (ciel atmosphérique)",
      subtitle: "(3/16π)(1+cos²θ) sur θ ∈ [0,π] — la diffusion moléculaire",
      fn: (x) => {
        const c = Math.cos(x);
        return 0.0596831 * (1 + c * c);
      },
      lo: 0,
      hi: Math.PI,
      groundTruth: "p(θ) = (3/16π)(1+cos²θ)",
      exactCost: 5,
    }),
    // three.js FogExp2: f = 1 − exp(−(density·depth)²). Served exp gives the
    // reference; the ALU-only hunt targets a Padé-grade fast slot.
    buildRegressionTask({
      id: "fog_exp2",
      title: "Brouillard exponentiel carré (three.js)",
      subtitle: "f = 1 − exp(−(d·z)²), d=1.2, z ∈ [0,3] — FogExp2 du moteur",
      groundTruth: "f = 1 − exp(−(dz)²)",
      rows: 400,
      varNames: ["x"],
      exactCost: 22,
      build: () => sample(400, 0, 3, (z) => 1 - Math.exp(-(1.2 * z) * (1.2 * z))),
      trueLaw: (v, i) => 1 - Math.exp(-(1.2 * v.x[i]) * (1.2 * v.x[i])),
      verify: () => null,
    }),
    // three.js punctual light attenuation:
    // pow(saturate(1 − (d/cutoff)⁴), 2) / d² — clamped window + inverse square.
    buildRegressionTask({
      id: "light_falloff_punctual",
      title: "Atténuation punctual (three.js)",
      subtitle: "saturate(1−(d/c)⁴)²/d², c=8, d ∈ [0.5,10] — chaque lumière, chaque frame",
      groundTruth: "att = sat(1−(d/8)⁴)²/d²",
      rows: 400,
      varNames: ["x"],
      exactCost: 20,
      build: () =>
        sample(400, 0.5, 10, (d) => {
          const w = Math.max(0, Math.min(1, 1 - Math.pow(d / 8, 4)));
          return (w * w) / (d * d);
        }),
      trueLaw: (v, i) => {
        const d = v.x[i];
        const w = Math.max(0, Math.min(1, 1 - Math.pow(d / 8, 4)));
        return (w * w) / (d * d);
      },
      verify: () => null,
    }),
    // Uncharted 2 filmic tonemap — the reference shoulder curve, six magic
    // constants inside one rational. Refinement stress test.
    buildRegressionTask({
      id: "uncharted2_tonemap",
      title: "Tonemap Uncharted 2 (filmic)",
      subtitle: "rationnel A=.15 B=.50 C=.10 D=.20 E=.02 F=.30 — le filmic de référence",
      groundTruth: "((x(Ax+CB)+DE)/(x(Ax+B)+DF))−E/F",
      rows: 400,
      varNames: ["x"],
      exactCost: 26,
      build: () => {
        const g = (x: number): number => {
          const A = 0.15, B = 0.5, Cst = 0.1, D = 0.2, E = 0.02, F = 0.3;
          return ((x * (A * x + Cst * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
        };
        const W = 11.2;
        const whiteScale = 1 / g(W);
        return sample(400, 0, 10, (x) => g(x * 2) * whiteScale);
      },
      trueLaw: (v, i) => {
        const A = 0.15, B = 0.5, Cst = 0.1, D = 0.2, E = 0.02, F = 0.3;
        const g = (x: number): number => ((x * (A * x + Cst * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
        return g(v.x[i] * 2) * (1 / g(11.2));
      },
      verify: () => null,
    }),
  ];
}

// garde-fou typage : les seeds restent optionnelles dans ce domaine
export type RenderingSeed = SpearNode;
void makeNode;
