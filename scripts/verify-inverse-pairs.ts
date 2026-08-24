// Inverse-pair composition verification — a new task category: two discovered
// formulas are correct TOGETHER if f(g(x)) ≈ x and g(f(y)) ≈ y on the domain.
// Demonstrated on the registry's sigmoid/logit pair.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseNode, evaluateScalar } from "../src/lib/spear/engine";

async function main() {
  const led = JSON.parse(readFileSync(join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json"), "utf8")) as Record<string, { tree?: unknown }>;
  const sigmoid = parseNode(led["sigmoid"].tree as never);
  const logit = parseNode(led["logit_ml"].tree as never);

  let worst1 = 0, worst2 = 0;
  for (let i = 1; i < 400; i++) {
    const x = 0.02 + (0.96 * i) / 400;
    // logit then sigmoid: should return to x
    worst1 = Math.max(worst1, Math.abs(evaluateScalar(sigmoid, { x: evaluateScalar(logit, { x }) }) - x));
    const y = -6 + (12 * i) / 400;
    worst2 = Math.max(worst2, Math.abs(evaluateScalar(logit, { x: evaluateScalar(sigmoid, { y }) }) - y));
  }
  console.log(`sigmoid∘logit : max |f(g(x)) − x| = ${worst1.toExponential(3)} sur (0.02..0.98)`);
  console.log(`logit∘sigmoid : max |g(f(y)) − y| = ${worst2.toExponential(3)} sur (-6..6)`);
  console.log(worst1 < 0.02 && worst2 < 0.05 ? "PAIRE INVERSE VALIDÉE — les deux formes sont correctes ensemble" : "PAIRE DÉGRADÉE — au moins une dérive");
}
main();
