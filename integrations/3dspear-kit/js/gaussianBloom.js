// SPEAR shader-ready gaussian — student-t k3 approximant of e^(-x²/2).
// Provenance: tuned by SPEAR constant refinement. 9 ALU units, MSE 6.4e-4.
// Targets low-end GLSL profiles where transcendental-per-tap is expensive.
// NOTE: for static blurs prefer precomputed taps; use this for per-pixel
// variable-σ evaluation (DoF, local bloom radius).
export function gaussianFast(x) {
  const s = 0.207 * x * x + 1;
  return (1.02232 / (s * s * s));
}
