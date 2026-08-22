import { NextRequest, NextResponse } from "next/server";
import { db } from "@/db";
import { spearRuns } from "@/db/schema";
import {
  runActivationPreset,
  runCustomRegressionPreset,
  runKvCachePreset,
  type ActivationTarget,
} from "@/lib/spear/presets";

export const runtime = "nodejs";

const ACTIVATION_TARGETS: ActivationTarget[] = ["silu", "gelu", "tanh", "sigmoid", "gaussian_cdf", "rl_distillation"];

interface RunBody {
  preset?: string;
  label?: string;
  config?: Record<string, unknown>;
}

function num(v: unknown, fallback: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export async function POST(req: NextRequest) {
  let body: RunBody;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Corps JSON invalide." }, { status: 400 });
  }

  const preset = body.preset;
  const rawConfig = body.config ?? {};

  try {
    if (preset === "activation") {
      const target = ACTIVATION_TARGETS.includes(rawConfig.target as ActivationTarget)
        ? (rawConfig.target as ActivationTarget)
        : "silu";
      const config = {
        target,
        populationSize: num(rawConfig.populationSize, 90),
        generations: num(rawConfig.generations, 35),
        maxDepth: num(rawConfig.maxDepth, 4),
        domainMin: num(rawConfig.domainMin, -4),
        domainMax: num(rawConfig.domainMax, 4),
        points: num(rawConfig.points, 400),
      };
      const result = runActivationPreset(config);
      const label = body.label?.trim() || `${target.toUpperCase()} — Synthèse d'activation LLM`;
      const [row] = await db
        .insert(spearRuns)
        .values({
          preset: "activation",
          label,
          config,
          status: "completed",
          formulaText: result.formulaText,
          formulaTree: result.formulaTree,
          fitness: result.fitness,
          mse: result.mse,
          linfError: result.linfError,
          treeSize: result.treeSize,
          durationMs: result.durationMs,
          history: result.history,
          metrics: result.metrics,
          chartData: result.chartData,
        })
        .returning();
      return NextResponse.json({ run: row });
    }

    if (preset === "kv_cache") {
      const config = {
        populationSize: num(rawConfig.populationSize, 70),
        generations: num(rawConfig.generations, 20),
        maxDepth: num(rawConfig.maxDepth, 3),
        seqLen: num(rawConfig.seqLen, 512),
        keepBudget: num(rawConfig.keepBudget, 64),
        numSamples: num(rawConfig.numSamples, 10),
      };
      const result = runKvCachePreset(config);
      const label = body.label?.trim() || "Éviction de KV-Cache — découverte de règle";
      const [row] = await db
        .insert(spearRuns)
        .values({
          preset: "kv_cache",
          label,
          config,
          status: "completed",
          formulaText: result.formulaText,
          formulaTree: result.formulaTree,
          fitness: result.fitness,
          mse: result.mse,
          linfError: result.linfError,
          treeSize: result.treeSize,
          durationMs: result.durationMs,
          history: result.history,
          metrics: result.metrics,
          chartData: result.chartData,
        })
        .returning();
      return NextResponse.json({ run: row });
    }

    if (preset === "custom") {
      const csv = String(rawConfig.csv ?? "").slice(0, 200_000);
      const config = {
        csv,
        populationSize: num(rawConfig.populationSize, 140),
        generations: num(rawConfig.generations, 45),
        maxDepth: num(rawConfig.maxDepth, 4),
      };
      const result = runCustomRegressionPreset(config);
      const label = body.label?.trim() || "Régression symbolique personnalisée";
      const [row] = await db
        .insert(spearRuns)
        .values({
          preset: "custom",
          label,
          config,
          status: "completed",
          formulaText: result.formulaText,
          formulaTree: result.formulaTree,
          fitness: result.fitness,
          mse: result.mse,
          linfError: result.linfError,
          treeSize: result.treeSize,
          durationMs: result.durationMs,
          history: result.history,
          metrics: result.metrics,
          chartData: result.chartData,
        })
        .returning();
      return NextResponse.json({ run: row });
    }

    return NextResponse.json({ error: `Preset inconnu: ${String(preset)}` }, { status: 400 });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Échec de l'exécution SPEAR.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
