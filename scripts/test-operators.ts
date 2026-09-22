// Tests unitaires numérotés par opérateur (issue #7).
// Pour chacun des 20 opérateurs : parité JS/WASM/C/torch sur des points de
// sonde (dont les rails), coût ALU/SFU annoncé dans OP_COST, comportement
// sur les rails (clamp exp, plancher pdiv, plancher log, clamp asin/acos).
// Assertion-based, pas de framework. Déviation au format demandé : un seul
// fichier avec une section numérotée par opérateur plutôt que 20 fichiers.
// Run: npx tsx scripts/test-operators.ts
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  estimateCost,
  evaluateScalar,
  makeNode,
  OP_COST,
  resetTranscendentalCost,
  setTranscendentalCost,
  toC,
  toPython,
  type NodeOp,
  type SpearNode,
} from "../src/lib/spear/engine";
import { instantiateSpearWasm, toWasmBytes } from "../src/lib/spear/wasm";

const v = (n: string): SpearNode => makeNode("var", { name: n });
const bin = (op: NodeOp): SpearNode => makeNode(op, { children: [v("x"), v("y")] });
const una = (op: NodeOp): SpearNode => makeNode(op, { children: [v("x")] });

const UNARY_OPS: CalcOp[] = ["relu", "abs", "sq", "sqrt", "cube", "exp", "sin", "cos", "log", "atan", "asin", "acos", "tanh", "erf"];
const BINARY_OPS: CalcOp[] = ["add", "sub", "mul", "pdiv", "max", "min"];
type CalcOp = Exclude<NodeOp, "var" | "const" | "neg">;

// points de sonde par opérateur — incluent volontairement les déclencheurs
// de rails (pdiv ~0, exp |>50|, log <0, asin/acos |>1|, sqrt <0)
const PROBES: Record<CalcOp, [number, number][]> = {
  add: [[2, 3], [-1.5, 0.25], [0, 0]],
  sub: [[2, 3], [-1.5, 0.25], [0, 0]],
  mul: [[2, 3], [-1.5, 0.25], [0, 0]],
  pdiv: [[7, 2.5], [1, 0], [1, 1e-5], [-1, 1e-5], [1, -1e-5], [0, 0]],
  max: [[2, 3], [-1.5, 0.25], [0, 0]],
  min: [[2, 3], [-1.5, 0.25], [0, 0]],
  relu: [[2, 0], [-3, 0], [0, 0]],
  abs: [[4, 0], [-4, 0], [0.25, 0]],
  sq: [[4, 0], [-4, 0], [0.25, 0]],
  sqrt: [[4, 0], [-4, 0], [0.25, 0]],
  cube: [[4, 0], [-4, 0], [0.25, 0]],
  exp: [[0, 0], [1000, 0], [-1000, 0], [3, 0]],
  sin: [[0.7, 0], [-2.1, 0], [0, 0]],
  cos: [[0.7, 0], [-2.1, 0], [0, 0]],
  log: [[2.5, 0], [-5, 0], [0, 0]],
  atan: [[0.7, 0], [-2.1, 0], [0, 0]],
  asin: [[0.3, 0], [2, 0], [-2, 0]],
  acos: [[0.3, 0], [2, 0], [-2, 0]],
  tanh: [[0.7, 0], [-2.1, 0], [0, 0]],
  erf: [[0.5, 0], [-1.2, 0], [0, 0]],
};

const nodeOf = (op: CalcOp): SpearNode => (BINARY_OPS.includes(op) ? bin(op) : una(op));

let failures = 0;
function check(label: string, cond: boolean, detail = ""): void {
  if (!cond) {
    failures++;
    console.log(`  ✗ ${label}${detail ? ` — ${detail}` : ""}`);
  }
}

// ------------------------------------------------------------------ harnais C
// toC émet du CUDA `__device__` — on strip le qualifier pour gcc. Un seul
// fichier avec les 20 fonctions + un main qui imprime toutes les évals.
function buildCSource(): string {
  const fns: string[] = [];
  const lines: string[] = [];
  // un littéral C float exige un point décimal : 2 -> "2.0"
  const cf = (n: number): string => (Number.isInteger(n) ? n.toFixed(1) : String(n));
  for (const op of [...BINARY_OPS, ...UNARY_OPS]) {
    const node = nodeOf(op);
    const decl = BINARY_OPS.includes(op) ? "const float x, const float y" : "const float x";
    fns.push(toC(node, `spear_${op}`, decl).replace("__device__ ", "static "));
    for (const [a, b] of PROBES[op]) {
      lines.push(
        BINARY_OPS.includes(op)
          ? `printf("${op} %.17g\\n", (double)spear_${op}(${cf(a)}f, ${cf(b)}f));`
          : `printf("${op} %.17g\\n", (double)spear_${op}(${cf(a)}f));`,
      );
    }
  }
  return ["#include <stdio.h>", "#include <math.h>", ...fns, "int main(void) {", ...lines, "  return 0;", "}"].join("\n");
}

// ------------------------------------------------------------- harnais torch
function buildPySource(): string {
  const fns: string[] = [];
  const pairs: { op: string; pts: [number, number][]; binary: boolean }[] = [];
  for (const op of [...BINARY_OPS, ...UNARY_OPS]) {
    fns.push(toPython(nodeOf(op), `spear_${op}`).replace("import torch\n\n", ""));
    pairs.push({ op, pts: PROBES[op], binary: BINARY_OPS.includes(op) });
  }
  return [
    "import torch, json",
    ...fns,
    `PAIRS = json.loads(${JSON.stringify(JSON.stringify(pairs))})`,
    "out = {}",
    "for p in PAIRS:",
    "  fn = globals()['spear_' + p['op']]",
    "  vals = []",
    "  for a, b in p['pts']:",
    "    ta = torch.tensor(float(a), dtype=torch.float64)",
    "    tb = torch.tensor(float(b), dtype=torch.float64)",
    "    vals.append(float(fn(ta, tb)) if p['binary'] else float(fn(ta)))",
    "  out[p['op']] = vals",
    "print(json.dumps(out))",
  ].join("\n");
}

async function main() {
  const total = 20;
  const tmp = mkdtempSync(join(tmpdir(), "spear-test-ops-"));
  try {
    // ---- backends C et torch : une exécution chacun, résultats parsés
    writeFileSync(join(tmp, "ops.c"), buildCSource());
    execFileSync("gcc", ["-O2", "-o", join(tmp, "ops.exe"), join(tmp, "ops.c")]);
    const cOut = execFileSync(join(tmp, "ops.exe"), { encoding: "utf8" });
    const cVals: Record<string, number[]> = {};
    for (const line of cOut.trim().split("\n")) {
      const [op, val] = line.split(" ");
      (cVals[op] ??= []).push(Number(val));
    }

    writeFileSync(join(tmp, "ops.py"), buildPySource());
    const pyRaw = execFileSync("python", [join(tmp, "ops.py")], { encoding: "utf8" });
    const pyVals = JSON.parse(pyRaw.trim()) as Record<string, number[]>;

    for (let i = 0; i < total; i++) {
      const op = [...BINARY_OPS, ...UNARY_OPS][i];
      const node = nodeOf(op);
      const pts = PROBES[op];
      console.log(`[${String(i + 1).padStart(2, "0")}/${total} ${op}]`);

      // -- coût ALU/SFU annoncé
      check("coût OP_COST", estimateCost(node) === OP_COST[op], `estimé=${estimateCost(node)} annoncé=${OP_COST[op]}`);

      for (let p = 0; p < pts.length; p++) {
        const scope = { x: pts[p][0], y: pts[p][1] };
        const js = evaluateScalar(node, scope);
        const args = BINARY_OPS.includes(op) ? [scope.x, scope.y] : [scope.x];

        // -- parité JS/WASM (f64, tolérance serrée)
        let wasm: number | null = null;
        try {
          const fn = await instantiateSpearWasm(Buffer.from(toWasmBytes(node)).toString("base64"));
          wasm = fn(args);
        } catch (e) {
          check(`wasm p${p}`, false, String(e).slice(0, 80));
        }
        if (wasm !== null) {
          check(`parité js/wasm p${p}`, Math.abs(js - wasm) <= 1e-12 * (Math.abs(js) + 1), `js=${js} wasm=${wasm}`);
        }

        // -- parité JS/C (f32 côté C → tolérance large ; erf: approx A&S vs libm)
        const cv = cVals[op]?.[p];
        check(`parité js/c p${p}`, cv !== undefined && (op === "erf"
          ? Math.abs(js - cv) <= 2e-7
          : Math.abs(js - cv) <= 1e-5 * (Math.abs(js) + 1)), `js=${js} c=${cv}`);

        // -- parité JS/torch (f64 ; erf: approx A&S vs torch.erf exact)
        const pv = pyVals[op]?.[p];
        check(`parité js/torch p${p}`, pv !== undefined && (op === "erf"
          ? Math.abs(js - pv) <= 2e-7
          : Math.abs(js - pv) <= 1e-12 * (Math.abs(js) + 1)), `js=${js} torch=${pv}`);
      }
    }

    // ---- [21] rails — comportement total hors domaine
    console.log(`[21/${total} rails]`);
    check("clamp exp haut", evaluateScalar(makeNode("exp", { children: [makeNode("const", { value: 1000 })] }), {}) === Math.exp(50));
    check("clamp exp bas", evaluateScalar(makeNode("exp", { children: [makeNode("const", { value: -1000 })] }), {}) === Math.exp(-50));
    check("plancher pdiv 0", evaluateScalar(bin("pdiv"), { x: 1, y: 0 }) === 1e4);
    check("clamp pdiv haut", evaluateScalar(bin("pdiv"), { x: 1, y: 1e-5 }) === 1e4);
    check("plancher pdiv signé −", evaluateScalar(bin("pdiv"), { x: 1, y: -1e-5 }) === -1e4);
    check("plancher log", evaluateScalar(una("log"), { x: -5 }) === Math.log(1e-30));
    check("clamp asin", Math.abs(evaluateScalar(una("asin"), { x: 2 }) - Math.PI / 2) < 1e-12);
    check("clamp acos", evaluateScalar(una("acos"), { x: 2 }) === 0);
    check("sqrt pair", evaluateScalar(una("sqrt"), { x: -4 }) === 2);

    // ---- [22] knob de repricing transcendantal (profil GPU vs défaut)
    console.log(`[22/${total} coût]`);
    setTranscendentalCost(1);
    check("knob GPU exp=1", OP_COST.exp === 1 && estimateCost(una("exp")) === 1);
    check("knob GPU add inchangé", OP_COST.add === 1);
    resetTranscendentalCost();
    check("reset exp=20", OP_COST.exp === 20 && estimateCost(una("exp")) === 20);

    console.log(`\nopérateurs: ${total - failures}/${total} sections OK`);
    if (failures > 0) process.exit(1);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
