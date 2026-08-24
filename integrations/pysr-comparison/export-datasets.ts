/**
 * Export SPEAR registry tasks as PySR-friendly CSVs (X columns + y column),
 * plus a manifest describing each dataset and the current SPEAR champion.
 *
 * Data is regenerated deterministically from the repo's own task builders:
 *   - activation tasks expose exactFn (the true generating function) and are
 *     sampled on their registry grid (linspace(lo, hi, 400));
 *   - regression tasks expose exactRefNode (the generating law as an AST,
 *     identical to EXACT_LAWS) evaluated on their registry grids.
 * A self-check re-scores each ledger champion against the exported rows and
 * compares with the ledger metric — large drift would mean the regeneration
 * no longer matches the engine's evaluation data.
 *
 * Run: npx tsx integrations/pysr-comparison/export-datasets.ts
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildTasks } from "../../src/lib/spear/benchmarks";
import {
  estimateCost,
  evaluateNode,
  nodeToString,
  parseNode,
  type SpearNode,
} from "../../src/lib/spear/engine";
import { linspace, mse } from "../../src/lib/spear/math-utils";

const HERE = join(fileURLToPath(import.meta.url), "..");
const OUT_DIR = join(HERE, "datasets");
const LEDGER_PATH = join(HERE, "..", "..", "spear-hall-of-fame.json");

/** Registry sampling domains (must mirror the builders in benchmarks.ts). */
const SPECS: Record<string, { domain: Record<string, [number, number]>; rows: number }> = {
  silu:              { domain: { x: [-6, 6] },   rows: 400 },
  gelu:              { domain: { x: [-6, 6] },   rows: 400 },
  sigmoid:           { domain: { x: [-8, 8] },   rows: 400 },
  softplus:          { domain: { x: [-4, 4] },   rows: 200 },
  tanh_sat:          { domain: { x: [-3, 3] },   rows: 400 },
  atan_unit:         { domain: { x: [-1, 1] },   rows: 400 },
  srgb_gamma:        { domain: { x: [0, 1] },    rows: 400 },
  gaussian_cdf:      { domain: { x: [-6, 6] },   rows: 400 },
  hill:              { domain: { c: [0.05, 5] }, rows: 200 },
  logistic_growth:   { domain: { t: [0, 4] },    rows: 200 },
  damped_oscillation:{ domain: { t: [0, 6] },    rows: 250 },
  kdv_soliton:       { domain: { x: [-10, 10], t: [0, 2] }, rows: 400 },
};

function main(): void {
  const tasks = new Map(buildTasks().map((t) => [t.id, t]));
  const ledger = JSON.parse(readFileSync(LEDGER_PATH, "utf8")) as Record<
    string,
    { metric?: number; formula?: string; tree?: unknown }
  >;
  mkdirSync(OUT_DIR, { recursive: true });

  const manifestTasks: unknown[] = [];

  for (const [id, spec] of Object.entries(SPECS)) {
    const def = tasks.get(id);
    if (!def) throw new Error(`task ${id} not in registry`);
    const varNames = def.variables;
    if (JSON.stringify(Object.keys(spec.domain)) !== JSON.stringify(varNames)) {
      throw new Error(`${id}: domain keys ${Object.keys(spec.domain)} != variables ${varNames}`);
    }

    const vars: Record<string, Float64Array> = {};
    for (const [name, [lo, hi]] of Object.entries(spec.domain)) {
      vars[name] = linspace(lo, hi, spec.rows);
    }

    // Target column: exactFn for activation tasks, generating AST otherwise.
    let y: Float64Array;
    if (varNames.length === 1 && def.exactFn) {
      const xv = vars[varNames[0]];
      y = Float64Array.from({ length: spec.rows }, (_, i) => def.exactFn!(xv[i]));
    } else if (def.exactRefNode) {
      y = evaluateNode(def.exactRefNode, vars, spec.rows);
    } else {
      throw new Error(`${id}: no exactFn / exactRefNode to generate targets`);
    }

    // CSV: X columns then y, full round-trip float precision.
    const header = [...varNames, "y"].join(",");
    const lines: string[] = [header];
    for (let i = 0; i < spec.rows; i++) {
      lines.push([...varNames.map((v) => String(vars[v][i])), String(y[i])].join(","));
    }
    writeFileSync(join(OUT_DIR, `${id}.csv`), lines.join("\n") + "\n");

    // Champion block: parsed ledger tree, cost in engine units, MSE on THESE rows.
    const entry = ledger[id];
    let champion: {
      formula: string;
      costUnits: number;
      mse?: number;
      ledgerMetric?: number;
    } | null = null;
    let check = "";
    if (entry?.tree) {
      let node: SpearNode;
      try {
        node = parseNode(entry.tree as never);
      } catch {
        node = null as never;
      }
      if (node) {
        let champMse: number | null = null;
        try {
          const pred = evaluateNode(node, vars, spec.rows);
          if (Array.from(pred).every(Number.isFinite)) champMse = mse(pred, y);
        } catch {
          /* champion references an unknown variable — leave mse null */
        }
        champion = {
          formula: entry.formula ?? nodeToString(node),
          costUnits: estimateCost(node),
          ...(champMse !== null && { mse: champMse }),
        };
        if (champMse !== null && typeof entry.metric === "number") {
          const ratio = champMse / Math.max(entry.metric, 1e-300);
          check = ` | self-check ledger=${entry.metric.toExponential(2)} recomputed=${champMse.toExponential(2)} (×${ratio.toFixed(3)})`;
        }
      }
    }

    manifestTasks.push({
      id,
      varNames,
      domain: spec.domain,
      rows: spec.rows,
      groundTruth: def.groundTruth,
      spearChampion: champion,
    });

    console.log(
      `${id.padEnd(20)} ${String(spec.rows).padStart(4)} rows × ${varNames.length} var(s)` +
        ` | costUnits=${champion?.costUnits ?? "?"}${check}`,
    );
  }

  const manifest = {
    description:
      "SPEAR vs PySR head-to-head datasets. Rows regenerate the registry evaluation grids exactly; spearChampion.mse is the ledger champion re-scored on these rows.",
    generatedAt: new Date().toISOString(),
    costUnitsNote: "SPEAR cost units are GPU ALU/SFU op weights (add/mul=1, div=4, sqrt=2, transcendental=20), NOT FLOPs.",
    tasks: manifestTasks,
  };
  writeFileSync(join(OUT_DIR, "manifest.json"), JSON.stringify(manifest, null, 2));
  console.log(`\n${manifestTasks.length} CSVs + manifest.json written to ${OUT_DIR}`);
}

main();
