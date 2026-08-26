import { parseFormula, parseNode } from "../src/lib/spear/engine";
import { toWasmBytes } from "../src/lib/spear/wasm";
import { readFileSync, writeFileSync } from "node:fs";

const led = JSON.parse(readFileSync("spear-hall-of-fame.json", "utf8"));
const out = {};
for (const id of ["sigmoid", "gaussian_cdf", "hill"]) {
  const bytes = toWasmBytes(parseNode(led[id].tree));
  out[id] = { b64: Buffer.from(bytes).toString("base64") };
  console.log(id, "->", bytes.length, "bytes");
}
// fusion avec l'existant
const cur = JSON.parse(readFileSync("docs/spear-wasm.js", "utf8").replace(/^window\.SPEAR_WASM=/, "").replace(/;\s*$/, ""));
Object.assign(cur, out);
writeFileSync("docs/spear-wasm.js", "window.SPEAR_WASM=" + JSON.stringify(cur) + ";");
console.log("total kernels:", Object.keys(cur).join(", "));