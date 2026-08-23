// Generate validation/gelu_policy_generated.py from the ledger GELU champion.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { parseNode, toPython } from "../src/lib/spear/engine";

const led = JSON.parse(readFileSync(join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json"), "utf8"));
const py = toPython(parseNode(led["gelu"].tree), "spear_gelu");
mkdirSync("validation", { recursive: true });
writeFileSync("validation/gelu_policy_generated.py", "# AUTO-GENERATED from spear-hall-of-fame.json\nimport math\n\n" + py + "\n");
console.log(py);
