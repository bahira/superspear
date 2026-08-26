import { parseNode, toMisraC } from "./src/lib/spear/engine.js";
const t = parseNode({ o: "erf", c: [{ o: "var", n: "x" }] });
const c = toMisraC(t, "t_erf");
console.log("ERF OK:", c.includes("erf"));
// et re-check mish complet sur l'état disque actuel
const led = JSON.parse(await import("node:fs").then(m => m.readFileSync("./spear-hall-of-fame.json", "utf8")));
const m = parseNode(led.mish.tree);
console.log("MISH OK:", toMisraC(m, "t_mish").length > 100);
