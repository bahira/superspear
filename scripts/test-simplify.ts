// Self-check for the reciprocal rewrite in simplify(): run via
//   npx tsx scripts/test-simplify.ts
import { makeNode, simplify, nodeToString, estimateCost } from "../src/lib/spear/engine";

const x = makeNode("var", { name: "x" });
const c = (v: number) => makeNode("const", { value: v });
const pdiv = (a: unknown, b: unknown) => makeNode("pdiv", { children: [a, b] as never });

// safe divisor: rewritten to mul-by-reciprocal (cost 4 -> 1)
const safe = simplify(pdiv(x, c(4)));
console.assert(nodeToString(safe) === "(x * 0.25)", `safe rewrite failed: ${nodeToString(safe)}`);
console.assert(estimateCost(safe) === 1, `cost should drop to 1, got ${estimateCost(safe)}`);

// protected divisor: left alone — the clamp rails ARE the formula's semantics
const prot = simplify(pdiv(x, c(3.44e-5)));
console.assert((prot as { op: string }).op === "pdiv", `protected div must survive: ${nodeToString(prot)}`);

// identity rules still intact
console.assert(nodeToString(simplify(pdiv(x, c(1)))) === "x", "x/1 must collapse");
console.assert(nodeToString(simplify(makeNode("add", { children: [c(2), c(3)] }))) === "5", "const fold broke");

console.log("test-simplify: all assertions passed");
