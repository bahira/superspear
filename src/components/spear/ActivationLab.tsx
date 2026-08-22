"use client";

import { useState } from "react";
import { XyChart } from "./Charts";
import { NumberField, RunButton, SelectField, MetricCard } from "./FormControls";
import { fmtMs, fmtNum } from "./format";
import type { SpearRunRecord } from "./types";

const TARGET_OPTIONS = [
  { value: "silu", label: "SiLU / Swish (LLaMA, Mistral, Qwen)" },
  { value: "gelu", label: "GELU (GPT, BERT)" },
  { value: "tanh", label: "tanh" },
  { value: "sigmoid", label: "Sigmoid" },
];

export function ActivationLab({ onRunComplete }: { onRunComplete: (run: SpearRunRecord) => void }) {
  const [target, setTarget] = useState("silu");
  const [populationSize, setPopulationSize] = useState(90);
  const [generations, setGenerations] = useState(35);
  const [maxDepth, setMaxDepth] = useState(4);
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
          preset: "activation",
          config: { target, populationSize, generations, maxDepth, domainMin: -4, domainMax: 4, points: 400 },
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
          <h3 className="text-base font-semibold text-slate-100">⚡ Synthèse d&apos;activation LLM</h3>
          <p className="mt-1 text-sm text-slate-400">
            SPEAR évolue une expression algébrique sans exponentielle/erf pour remplacer une fonction
            d&apos;activation transcendante utilisée dans les couches FFN (SwiGLU / GEGLU).
          </p>
        </div>
        <SelectField label="Fonction cible" value={target} onChange={setTarget} options={TARGET_OPTIONS} />
        <NumberField label="Taille de population" value={populationSize} onChange={setPopulationSize} min={20} max={400} />
        <NumberField label="Générations" value={generations} onChange={setGenerations} min={5} max={100} />
        <NumberField label="Profondeur max. de l'arbre" value={maxDepth} onChange={setMaxDepth} min={2} max={6} />
        <RunButton onClick={run} loading={loading}>
          Lancer SPEAR
        </RunButton>
        {error ? <p className="text-sm text-rose-400">{error}</p> : null}
      </div>

      <div className="flex flex-col gap-4">
        {result ? (
          <>
            <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/5 p-4">
              <p className="text-xs uppercase tracking-wide text-emerald-400">Formule découverte</p>
              <p className="mt-1 break-all font-mono text-sm text-emerald-200">{result.formulaText}</p>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricCard label="MSE" value={fmtNum(result.mse, 6)} />
              <MetricCard label="Erreur L∞" value={fmtNum(result.linfError)} />
              <MetricCard label="Taille arbre" value={String(result.treeSize ?? "—")} />
              <MetricCard
                label="Accélération"
                value={metrics.speedup ? `${metrics.speedup.toFixed(2)}×` : "—"}
                accent="text-emerald-400"
              />
              <MetricCard label="Temps SPEAR (calcul exact)" value={fmtMs(metrics.exactTimeMs)} />
              <MetricCard label="Temps SPEAR (formule)" value={fmtMs(metrics.spearTimeMs)} />
              <MetricCard label="Temps de recherche" value={fmtMs(result.durationMs)} />
              <MetricCard label="Éléments testés" value={metrics.benchElements ? metrics.benchElements.toLocaleString("fr-FR") : "—"} />
            </div>
            <XyChart
              title="Cible exacte vs. formule SPEAR"
              series={[
                { name: "Cible exacte", color: "#38bdf8", points: (result.chartData ?? []).map((p) => ({ x: p.x, y: p.target ?? 0 })) },
                { name: "SPEAR (approx.)", color: "#34d399", points: (result.chartData ?? []).map((p) => ({ x: p.x, y: p.predicted ?? 0 })) },
              ]}
            />
            <XyChart
              title="Convergence de la fitness par génération"
              series={[
                {
                  name: "Meilleure fitness",
                  color: "#f472b6",
                  points: (result.history ?? []).map((h) => ({ x: h.generation, y: h.bestFitness })),
                },
              ]}
            />
          </>
        ) : (
          <div className="flex h-full min-h-[300px] items-center justify-center rounded-2xl border border-dashed border-slate-700 text-sm text-slate-500">
            Lancez une exécution pour voir la formule découverte et ses métriques.
          </div>
        )}
      </div>
    </div>
  );
}
