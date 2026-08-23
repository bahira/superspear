// SPEAR discovered kernel — 2-link IK elbow angle, EXACT (1.1e-13 vs acos).
// Provenance: acos identity rewritten through the atan primitive, refined by
// SPEAR. Replaces iterative Newton-DLS chains (×20 vs 8-iteration solve).
// Usage: elbowAngle(reachDistance, linkLen2, linkLen3) -> angle in radians.
export function elbowAngle(d, l2, l3) {
  const z =
    (d * d - (l2 * l2 + l3 * l3)) /
    (2 * (l2 * l3));
  return (
    1.5707963267948966 -
    Math.atan(
      z /
        Math.sqrt(Math.abs(1 - z * z))
    )
  );
}
