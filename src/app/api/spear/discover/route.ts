import { NextRequest, NextResponse } from "next/server";
import {
  parseFormula,
  evaluateScalar,
  evaluateNode,
  nodeToString,
  estimateCost,
  toMisraC,
  collectVarNames,
  type SpearNode,
} from "@/lib/spear/engine";
import { mse, linfError, r2Score } from "@/lib/spear/math-utils";
import { toWasmBytes } from "@/lib/spear/wasm";
import { evolve, type EvolveConfig } from "@/lib/spear/engine";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

interface DiscoverBody {
  /** Target curve as a printed SPEAR formula, e.g. "sin(x) * exp(-x / 3)" */
  formula: string;
  varName?: string;
  lo?: number;
  hi?: number;
  rows?: number;
  populationSize?: number;
  generations?: number;
  maxDepth?: number;
  /** allow transcendental ops in the search (default true) */
  transcendentals?: boolean;
}

function num(v: unknown, fallback: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

const BASE_OPS = ["add", "sub", "mul", "pdiv", "relu", "abs", "neg", "sq", "cube", "sqrt", "max", "min"] as const;
const TRASC_OPS = ["exp", "sin", "cos", "log", "atan"] as const;

export async function POST(req: NextRequest) {
  let body: DiscoverBody;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Corps JSON invalide." }, { status: 400 });
  }

  if (!body.formula || typeof body.formula !== "string") {
    return NextResponse.json({ error: "Champ 'formula' requis (formule cible SPEAR)." }, { status: 400 });
  }

  // Parse the user's target curve — this validates syntax AND builds the law.
  let targetNode: SpearNode;
  try {
    targetNode = parseFormula(body.formula);
  } catch (e) {
    return NextResponse.json({ error: `Formule cible invalide: ${String(e).slice(0, 200)}` }, { status: 400 });
  }

  const vars = [...new Set(collectVarNames(targetNode))];
  if (vars.length === 0) {
    return NextResponse.json({ error: "La formule cible doit contenir au moins une variable." }, { status: 400 });
  }

  const lo = num(body.lo, -5);
  const hi = num(body.hi, 5);
  const rows = clamp(num(body.rows, 400), 20, 4000);
  const varName = body.varName && typeof body.varName === "string" ? body.varName : vars[0];
  if (!vars.includes(varName)) {
    return NextResponse.json({ error: `Variable '${varName}' absente de la formule.` }, { status: 400 });
  }
  if (vars.length > 1) {
    return NextResponse.json(
      { error: `Multi-variables détecté (${vars.join(",")}) — v1 du service supporte une seule variable.` },
      { status: 400 },
    );
  }

  // dataset: quasi-uniform grid over [lo,hi], target evaluated from the law
  const x = new Float64Array(rows);
  const y = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    x[i] = lo + ((hi - lo) * ((i * 0.6180339887) % 1));
    y[i] = evaluateScalar(targetNode, { [varName]: x[i] });
    if (!Number.isFinite(y[i])) {
      return NextResponse.json({ error: `La formule produit une valeur non finie à x=${x[i]}.` }, { status: 400 });
    }
  }

  const gpCfg: EvolveConfig = {
    variables: [varName],
    constRange: [-5, 5],
    ops: [
      ...(body.transcendentals === false ? [] : TRASC_OPS),
      ...BASE_OPS,
    ],
    maxDepth: clamp(num(body.maxDepth, 5), 2, 7),
    populationSize: clamp(num(body.populationSize, 120), 20, 300),
    generations: clamp(num(body.generations, 60), 5, 200),
  };

  const varsMap: Record<string, Float64Array> = { [varName]: x };
  const fitnessFn = (node: SpearNode) => {
    try {
      const pred = evaluateNode(node, varsMap, rows);
      for (let i = 0; i < rows; i++) if (!Number.isFinite(pred[i])) return { fitness: -1e9, size: node.size };
      const m = mse(pred, y);
      if (!Number.isFinite(m)) return { fitness: -1e9, size: node.size };
      const li = linfError(pred, y);
      return { fitness: -(m + 0.05 * li), size: node.size, extra: li };
    } catch {
      return { fitness: -1e9, size: node.size };
    }
  };

  const result = evolve(gpCfg, fitnessFn);
  const best = result.best;

  const predFinal = evaluateNode(best, varsMap, rows);
  const finalMse = mse(predFinal, y);
  const finalLinf = linfError(predFinal, y);
  const finalR2 = r2Score(predFinal, y);

  const targetCost = estimateCost(targetNode);
  const bestCost = estimateCost(best);

  // verified exports
  const fnId = "spear_discovered";
  const params = `const float ${varName}`;
  const c99 = toMisraC(best, fnId, params);
  const wasmBase64 = Buffer.from(toWasmBytes(best)).toString("base64");

  return NextResponse.json({
    discovery: {
      formula: nodeToString(best),
      treeSize: best.size,
      costUnits: bestCost,
      mse: finalMse,
      linfError: finalLinf,
      r2: finalR2,
      generations: result.history.length,
      durationMs: result.durationMs,
    },
    target: {
      formula: nodeToString(targetNode),
      costUnits: targetCost,
      speedupVsTarget: targetCost / Math.max(1, bestCost),
    },
    exports: {
      misraC: c99,
      wasmBase64,
      torch: null, // use /api/spear/run custom_regression for torch export today
    },
    history: result.history.slice(-40),
  });
}
