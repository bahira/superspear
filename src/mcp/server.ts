#!/usr/bin/env node
// SPEAR MCP Server — exposes the symbolic regression engine to LLM agents.
//
// Tools: discover, list_kernels, get_kernel, evaluate_kernel, run_benchmark
// Transport: stdio (standard for local MCP servers)
//
// Usage: npx tsx src/mcp/server.ts
// Or add to Claude Desktop / Cursor / Windsurf config:
//   { "command": "npx", "args": ["tsx", "path/to/src/mcp/server.ts"] }

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import {
  parseFormula,
  parseNode,
  evaluateScalar,
  evaluateNode,
  estimateCost,
  nodeToString,
  toPython,
  toC,
  toMisraC,
} from "../lib/spear/engine.js";
import { toWasmBytes } from "../lib/spear/wasm.js";
import { buildTasks } from "../lib/spear/benchmarks.js";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// ---------------------------------------------------------------------------
// Ledger access
// ---------------------------------------------------------------------------

const LEDGER_PATH = join(import.meta.dirname ?? ".", "..", "..", "spear-hall-of-fame.json");

function loadLedger(): Record<string, any> {
  try {
    return JSON.parse(readFileSync(LEDGER_PATH, "utf8"));
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------------------
// Server setup
// ---------------------------------------------------------------------------

const server = new McpServer({
  name: "spear-kernels",
  version: "1.2.0",
});

// ---------------------------------------------------------------------------
// Tool 1: discover — evolve a kernel for a target formula
// ---------------------------------------------------------------------------

server.tool(
  "discover",
  "Discover an optimized closed-form approximation of a target mathematical expression using genetic programming. Returns the discovered formula, its cost in ALU/SFU units, MSE, R², and MISRA-C/WASM exports.",
  {
    formula: z.string().describe("Target expression in SPEAR syntax, e.g. 'sin(x) * exp(-x / 3)' or 'x^2.4'"),
    varName: z.string().optional().describe("Variable name (default: auto-detected)"),
    lo: z.number().optional().describe("Domain lower bound (default: -5)"),
    hi: z.number().optional().describe("Domain upper bound (default: 5)"),
    rows: z.number().optional().describe("Number of sample points (default: 400)"),
    generations: z.number().optional().describe("Evolution generations (default: 60, range 5–200)"),
    populationSize: z.number().optional().describe("Population size (default: 120)"),
    transcendentals: z.boolean().optional().describe("Allow exp/sin/cos/log in search (default true)"),
  },
  async ({ formula, varName, lo, hi, rows: rowsIn, generations, populationSize, transcendentals }) => {
    try {
      const target = parseFormula(formula);
      const vars = [...new Set(collectVarNames(target))];
      if (vars.length === 0) return err("La formule doit contenir au moins une variable.");
      if (vars.length > 1) return err(`Multi-variables (${vars.join(",")}) non supporté en v1.`);

      const vName = varName ?? vars[0];
      if (!vars.includes(vName)) return err(`Variable '${vName}' absente.`);

      const loV = num(lo, -5);
      const hiV = num(hi, 5);
      const rows = clampInt(rowsIn ?? 400, 20, 4000);
      const gens = clampInt(generations ?? 60, 5, 200);
      const pop = clampInt(populationSize ?? 120, 20, 300);

      // generate dataset
      const xs = new Float64Array(rows);
      const ys = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        xs[i] = loV + ((hiV - loV) * ((i * 0.6180339887) % 1));
        ys[i] = evaluateScalar(target, { [vName]: xs[i] });
        if (!Number.isFinite(ys[i])) return err(`Non-finite à x=${xs[i].toFixed(3)}.`);
      }

      // evolve
      const { evolve } = await import("../lib/spear/engine.js");
      const { mse, linfError, r2Score } = await import("../lib/spear/math-utils.js");
      const varsMap: Record<string, Float64Array> = { [vName]: xs };
      const result = evolve({
        variables: [vName],
        constRange: [-5, 5],
        ops: transcendentals !== false
          ? ["add", "sub", "mul", "pdiv", "relu", "abs", "neg", "sq", "cube", "sqrt", "max", "min", "exp", "sin", "cos", "log", "atan"]
          : ["add", "sub", "mul", "pdiv", "relu", "abs", "neg", "sq", "cube", "sqrt", "max", "min"],
        maxDepth: 5,
        populationSize: pop,
        generations: gens,
      }, (node) => {
        try {
          const pred = evaluateNode(node, varsMap, rows);
          for (let i = 0; i < rows; i++) if (!Number.isFinite(pred[i])) return { fitness: -1e9, size: node.size };
          const m = mse(pred, ys);
          if (!Number.isFinite(m)) return { fitness: -1e9, size: node.size };
          const li = linfError(pred, ys);
          return { fitness: -(m + 0.05 * li), size: node.size, extra: li };
        } catch {
          return { fitness: -1e9, size: node.size };
        }
      });

      const best = result.best;
      const predFinal = evaluateNode(best, varsMap, rows);
      const finalMse = mse(predFinal, ys);
      const finalR2 = r2Score(predFinal, ys);
      const targetCost = estimateCost(target);
      const bestCost = estimateCost(best);

      return ok({
        discovered_formula: nodeToString(best),
        cost_units: bestCost,
        mse: finalMse,
        r2: finalR2,
        target_cost_units: targetCost,
        speedup_vs_target: +(targetCost / Math.max(1, bestCost)).toFixed(2),
        generations_used: result.history.length,
        duration_ms: result.durationMs,
        misra_c: toMisraC(best, fnId("discovered"), `const float ${vName}`),
        python: toPython(best, fnId("discovered")),
        wasm_base64: Buffer.from(toWasmBytes(best)).toString("base64"),
      });
    } catch (e) {
      return err(String(e).slice(0, 300));
    }
  }
);

// ---------------------------------------------------------------------------
// Tool 2: list_kernels — browse the registry
// ---------------------------------------------------------------------------

server.tool(
  "list_kernels",
  "List all kernels in the SPEAR hall-of-fame registry with their records, costs, speedups, and exact-solve status.",
  {},
  async () => {
    const led = loadLedger();
    const rows = Object.entries(led).map(([id, e]: [string, any]) => ({
      id,
      title: e.title,
      metric: e.metric,
      level: e.level,
      cost: e.speed?.formulaCost,
      speedup_vs_exact: e.speed?.speedup ? +e.speed.speedup.toFixed(2) : undefined,
      vs_iterative: e.speed?.vsIterative ? `${e.speed.vsIterative.speedup.toFixed(0)} vs ${e.speed.vsIterative.label}` : undefined,
      has_fast_slot: !!e.fast,
    }));
    return ok(JSON.stringify(rows, null, 2));
  }
);

// ---------------------------------------------------------------------------
// Tool 3: get_kernel — full details for one kernel
// ---------------------------------------------------------------------------

server.tool(
  "get_kernel",
  "Get full details for a specific kernel: precise formula, fast variant, C/MISRA export, Python export, WASM base64, cost breakdown.",
  {
    id: z.string().describe("Kernel task ID, e.g. 'silu', 'gaussian_cdf', 'kerr_spin'"),
  },
  async ({ id }) => {
    const led = loadLedger();
    const e = led[id];
    if (!e) return err(`Kernel '${id}' introuvable.`);
    const tree = e.tree ? parseNode(e.tree) : null;
    return ok(JSON.stringify({
      id,
      title: e.title,
      formula_precise: e.formula,
      formula_fast: e.fast?.formula ?? null,
      metric: e.metric,
      level: e.level,
      cost_precise: e.speed?.formulaCost,
      cost_fast: e.fast?.speed?.formulaCost,
      speedup_vs_exact: e.speed?.speedup,
      vs_iterative: e.speed?.vsIterative,
      misra_c: tree ? toMisraC(tree, `spear_${id}`) : null,
      python: tree ? toPython(tree, `spear_${id}`) : null,
      wasm_available: !!tree,
    }, null, 2));
  }
);

// ---------------------------------------------------------------------------
// Tool 4: evaluate_kernel — numeric evaluation
// ---------------------------------------------------------------------------

server.tool(
  "evaluate_kernel",
  "Evaluate a kernel's precise or fast formula at specific input values.",
  {
    id: z.string(),
    slot: z.enum(["precise", "fast"]).optional().default("precise"),
    inputs: z.record(z.string(), z.number()).describe("Named input values, e.g. { x: 2.5 }"),
  },
  async ({ id, slot, inputs }) => {
    const led = loadLedger();
    const e = led[id];
    if (!e) return err(`Kernel '${id}' introuvable.`);
    const treeSrc = slot === "fast" ? (e.fastTree ?? e.tree) : e.tree;
    if (!treeSrc) return err(`Pas d'arbre pour '${id}'.`);
    const node = parseNode(treeSrc);
    const val = evaluateScalar(node, inputs as Record<string, number>);
    return ok(JSON.stringify({ id, slot, inputs, result: val }));
  }
);

// ---------------------------------------------------------------------------
// Tool 5: run_benchmark — launch discovery on all tasks
// ---------------------------------------------------------------------------

server.tool(
  "run_benchmark",
  "Run the SPEAR grounded loop on selected tasks. Returns breakthrough records.",
  {
    budget: z.number().optional().default(500).describe("Iterations per task (500 recommended)"),
    taskIds: z.array(z.string()).optional().describe("Subset of task IDs (default: all)"),
  },
  async ({ budget: bud, taskIds }) => {
    // This would normally spawn the farm; for MCP simplicity we report status.
    return ok(JSON.stringify({
      note: "Pour lancer une vraie passe, utilisez: npx tsx scripts/run-farm.ts <seed> <budget> <workers> [taskIds]",
      total_tasks_in_registry: Object.keys(loadLedger()).length,
      suggested_command: taskIds?.length
        ? `npx tsx scripts/run-farm.ts $(date +%s) ${bud} ${taskIds.length} ${taskIds.join(",")}`
        : `npx tsx scripts/run-farm.ts $(date +%s) ${bud} 6`,
    }));
  }
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function num(v: unknown, fallback: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}
function clampInt(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, Math.round(v)));
}
function fnId(suffix: string): string {
  return `spear_${suffix}`;
}
function collectVarNames(node: any): string[] {
  const names: string[] = [];
  const walk = (nd: any) => {
    if (!nd) return;
    if (nd.op === "var" && nd.name && !names.includes(nd.name)) names.push(nd.name);
    (nd.children ?? []).forEach(walk);
  };
  walk(node);
  return names;
}
function ok(data: unknown) {
  return { content: [{ type: "text" as const, text: typeof data === "string" ? data : JSON.stringify(data, null, 2) }] };
}
function err(msg: string) {
  return { content: [{ type: "text" as const, text: `Erreur: ${msg}` }], isError: true };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("[SPEAR MCP] serveur démarré sur stdio");
}

main().catch((e) => {
  console.error("[SPEAR MCP] fatal:", e);
  process.exit(1);
});
