"use client";

import { useState } from "react";
import { XyChart } from "./Charts";
import { NumberField, RunButton, MetricCard } from "./FormControls";
import { fmtMs, fmtPct } from "./format";
import type { SpearRunRecord } from "./types";

export function KvCacheLab({ onRunComplete }: { onRunComplete: (run: SpearRunRecord) => void }) {
  const [seqLen, setSeqLen] = useState(512);
  const [keepBudget, setKeepBudget] = useState(64);
  const [numSamples, setNumSamples] = useState(10);
  const [populationSize, setPopulationSize] = useState(70);
  const [generations, setGenerations] = useState(20);
  const [maxDepth, setMaxDepth] = useState(3);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SpearRunRecord | null>(null);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/spear/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          preset: "kv_cache",
          config: { seqLen, keepBudget, numSamples, populationSize, generations, maxDepth },
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Échec de l'exécution.");
      setResult(data.run);
      onRunComplete(data.run);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur inconnue.");
    } finally {
      setLoading(false);
    }
  }

  const metrics = (result?.metrics ?? {}) as Record<string, number>;

  return (
    <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
      <div className="flex flex-col gap-4 rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
        <div>
          <h3 className="text-base font-semibold text-slate-100">🧠 Éviction de KV-Cache</h3>
          <p className="mt-1 text-sm text-slate-400">
            SPEAR évolue une règle de score Score(A, P, S, R) à partir de l&apos;attention (A), la position (P),
            l&apos;indicateur de token puits (S) et la récence (R), pour ne conserver qu&apos;une fraction du cache.
          </p>
        </div>
        <NumberField label="Longueur de séquence" value={seqLen} onChange={setSeqLen} min={128} max={768} step={64} />
        <NumberField label="Budget de tokens conservés" value={keepBudget} onChange={setKeepBudget} min={8} max={256} step={8} />
        <NumberField label="Échantillons synthétiques" value={numSamples} onChange={setNumSamples} min={4} max={12} />
        <NumberField label="Taille de population" value={populationSize} onChange={setPopulationSize} min={20} max={90} />
        <NumberField label="Générations" value={generations} onChange={setGenerations} min={5} max={25} />
        <NumberField label="Profondeur max. de l'arbre" value={maxDepth} onChange={setMaxDepth} min={2} max={5} />
        <RunButton onClick={run} loading={loading}>
          Lancer SPEAR
        </RunButton>
        {error ? <p className="text-sm text-rose-400">{error}</p> : null}
      </div>

      <div className="flex flex-col gap-4">
        {result ? (
          <>
            <div className="rounded-2xl border border-cyan-500/30 bg-cyan-500/5 p-4">
              <p className="text-xs uppercase tracking-wide text-cyan-400">Règle de score découverte</p>
              <p className="mt-1 break-all font-mono text-sm text-cyan-200">Score = {result.formulaText}</p>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricCard
                label="Masse d'attention captée"
                value={fmtPct(metrics.attentionMassCapturedPct)}
                accent="text-cyan-400"
              />
              <MetricCard label="Compression" value={metrics.compressionRatio ? `${metrics.compressionRatio.toFixed(1)}×` : "—"} />
              <MetricCard label="Réduction mémoire" value={fmtPct(metrics.memoryReductionPct)} />
              <MetricCard label="Tokens conservés" value={`${metrics.keepBudget ?? "—"} / ${metrics.seqLen ?? "—"}`} />
              <MetricCard label="Temps de recherche" value={fmtMs(result.durationMs)} />
              <MetricCard label="Taille de la règle" value={String(result.treeSize ?? "—")} />
              <MetricCard label="Échantillons" value={String(metrics.numSamples ?? "—")} />
              <MetricCard label="Fitness" value={result.fitness?.toFixed(4) ?? "—"} />
            </div>
            <XyChart
              title="Convergence de la fitness (masse d'attention captée)"
              series={[
                {
                  name: "Meilleure fitness",
                  color: "#22d3ee",
                  points: (result.history ?? []).map((h) => ({ x: h.generation, y: h.bestFitness })),
                },
              ]}
            />
          </>
        ) : (
          <div className="flex h-full min-h-[300px] items-center justify-center rounded-2xl border border-dashed border-slate-700 text-sm text-slate-500">
            Lancez une exécution pour découvrir une règle d&apos;éviction de cache.
          </div>
        )}
      </div>
    </div>
  );
}
