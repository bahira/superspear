import { makeNode, type SpearNode } from "../../engine";
import { simplify } from "../../engine";
import type { TaskDef } from "../types";
import { buildActivationTask } from "../factories";
import * as S from "../shared";

export function defs(): TaskDef[] {
  return [
    // Mel scale — la carte fréquentielle de tous les spectrogrammes MFCC.
    // mel(f) = 2595·log₁₀(1+f/700) = 1127·ln(1+f/700) — pur log composé.
buildActivationTask({
      id: "mel_scale",
      title: "Échelle de Mel (MFCC · speech)",
      subtitle: "mel(f)=2595·log₁₀(1+f/700) sur l'audio audible [20Hz, 20kHz]",
      fn: (x) => 2595 * Math.log10(1 + x / 700),
      lo: 20,
      hi: 20000,
      groundTruth: "mel(f) = 2595·log₁₀(1+f/700)",
      exactCost: 23,
      extraSeeds: (() => {
        const f = S.V("x");
        return [
          // pont exact: ln10·log₁₀ → 1127.01·ln(1+f/700)
          makeNode("mul", { children: [
            S.C(1127.01048),
            makeNode("log", { children: [makeNode("add", { children: [
              S.C(1),
              makeNode("pdiv", { children: [f, S.C(700)] }),
            ] })] }),
          ] }),
        ];
      })(),
    }),

    // A-weighting IEC 61672 — la courbe de pondération acoustique universelle.
    // R_A(f) = 12194²·f⁴ / ((f²+20.6²)·√((f²+107.7²)(f²+737.9²))·(f²+12194²))
    // A(f) = 20·log₁₀(R_A) + 2. Cascade quadratique: zéro SFU sauf le √ et le log.
    buildActivationTask({
      id: "a_weighting",
      title: "Pondération A (acoustique · sonomètres)",
      subtitle: "A(f)=20log₁₀(R_A)+2 sur [20Hz,20kHz] — cascade de pôles quadratiques, défi structurel",
      fn: (x) => {
        const f2 = x * x;
        const num = 148691236 * f2 * f2; // 12194²
        const den = (f2 + 424.36) * Math.sqrt((f2 + 11599.29) * (f2 + 544496.41)) * (f2 + 148691236);
        return 20 * Math.log10(num / den) + 2.0;
      },
      lo: 20,
      hi: 20000,
      groundTruth: "A(f) = 20·log₁₀(R_A(f)) + 2 dB",
      exactCost: 55,
      extraSeeds: (() => {
        const x = S.V("x");
        const x2 = makeNode("sq", { children: [x] });
        const num = makeNode("mul", { children: [makeNode("mul", { children: [S.C(148691236), x2] }), x2] });
        const d1 = makeNode("add", { children: [x2, S.C(424.36)] });
        const d2 = makeNode("add", { children: [x2, S.C(11599.29)] });
        const d3 = makeNode("add", { children: [x2, S.C(544496.41)] });
        const d4 = makeNode("add", { children: [x2, S.C(148691236)] });
        const ra = makeNode("pdiv", { children: [
          num,
          makeNode("mul", { children: [
            d1,
            makeNode("mul", { children: [
              makeNode("sqrt", { children: [makeNode("mul", { children: [d2, d3] })] }),
              d4,
            ] }),
          ] }),
        ] });
        // A = (20/ln10)·ln(R_A) + 2
        return [makeNode("add", { children: [
          makeNode("mul", { children: [S.C(8.68588964), makeNode("log", { children: [ra] })] }),
          S.C(2),
        ] })];
      })(),
    }),
  ];
}