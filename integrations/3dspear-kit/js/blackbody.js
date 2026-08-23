// SPEAR discovered kernel — blackbody red channel vs color temperature.
// Provenance: evolved by SPEAR (L4, MSE 3.6e-6 vs Tanner Helland fit),
// ×1.79 faster than the piecewise power-law reference at 14 ALU units.
// Usage (PBR light color): light.color.setRGB(blackbodyRed(K), g, b);
export function blackbodyRed(tempK) {
  const x = tempK;
  return (
    12961264.054148 *
      ((6608.647704 / Math.max(x, 6608.647704)) /
        Math.pow(Math.max(Math.max(x, 6608.647704), 6449.745424), 2)) +
    0.703081
  );
}
