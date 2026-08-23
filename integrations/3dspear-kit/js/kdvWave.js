// SPEAR discovered kernel — KdV soliton crest (water surfaces, waves).
// Provenance: evolved by SPEAR (L2, MSE 1.7e-3 vs 2·sech² reference),
// ×7.57 cheaper in model units, ×1.96 measured wall-clock. 7 ALU units.
// Usage: per-vertex lake displacement, evaluated every frame.
export function kdvCrest(x) {
  return (
    -1.553578 * Math.max(Math.sqrt(Math.abs(6.660595 - Math.max(3.1671, x))), 0.603662) +
    2.906087
  );
}
