"use client";

import { useState } from "react";
import { XyChart } from "./Charts";
import { NumberField, RunButton, MetricCard } from "./FormControls";
import { fmtMs, fmtNum } from "./format";
import type { SpearRunRecord } from "./types";

function buildExampleCsv(): string {
  // Free-fall distance law: d = 0.5 * g * t^2 (+ small measurement noise)
  const g = 9.81;
  const rows = ["t,y"];
  for (let i = 0; i <= 40; i++) {
    const t = i * 0.1;
    const noise = (Math.random() - 0.5) * 0.05;
    const d = 0.5 * g * t * t + noise;
    rows.push(`${t.toFixed(2)},${d.toFixed(4)}`);
  }
  return rows.join("\n");
}

export function CustomRegressionLab({ onRunComplete }: { onRunComplete: (run: SpearRunRecord) => void }) {
  const [csv, setCsv] = useState("");
  const [populationSize, setPopulationSize] = useState(140);
  const [generations, setGenerations] = useState(45);
  const [maxDepth, setMaxDepth] = useState(4);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SpearRunRecord | null>(null);

  async function run() {
    if (!csv.trim()) {
      setError("Collez un jeu de données CSV avec une colonne cible 'y'.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/spear/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset: "custom", config: { csv, populationSize, generations, maxDepth } }),
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

  const metrics = (result?.metrics ?? {}) as Record<string, unknown>;

  return (
    <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
      <div className="flex flex-col gap-4 rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
        <div>
          <h3 className="text-base font-semibold text-slate-100">🧮 Régression symbolique générique</h3>
          <p className="mt-1 text-sm text-slate-400">
            Collez un CSV avec un en-tête (variables + colonne cible <code>y</code>). SPEAR découvre une
            formule fermée qui explique vos données — utile pour redécouvrir des lois physiques ou métier.
          </p>
        </div>
        <textarea
          value={csv}
          onChange={(e) => setCsv(e.target.value)}
          placeholder={"x,y\n0,0\n1,1\n2,4\n3,9\n..."}
          rows={10}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 font-mono text-xs text-slate-100 outline-none focus:border-emerald-500"
        />
        <button
          type="button"
          onClick={() => setCsv(buildExampleCsv())}
          className="self-start rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:border-emerald-500 hover:text-emerald-300"
        >
          Charger un exemple (chute libre d = ½gt²)
        </button>
        <NumberField label="Taille de population" value={populationSize} onChange={setPopulationSize} min={20} max={300} />
        <NumberField label="Générations" value={generations} onChange={setGenerations} min={5} max={80} />
        <NumberField label="Profondeur max. de l'arbre" value={maxDepth} onChange={setMaxDepth} min={2} max={6} />
        <RunButton onClick={run} loading={loading}>
          Lancer SPEAR
        </RunButton>
        {error ? <p className="text-sm text-rose-400">{error}</p> : null}
      </div>

      <div className="flex flex-col gap-4">
        {result ? (
          <>
            <div className="rounded-2xl border border-fuchsia-500/30 bg-fuchsia-500/5 p-4">
              <p className="text-xs uppercase tracking-wide text-fuchsia-400">Formule découverte</p>
              <p className="mt-1 break-all font-mono text-sm text-fuchsia-200">y ≈ {result.formulaText}</p>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricCard label="MSE" value={fmtNum(result.mse, 6)} />
              <MetricCard label="Erreur L∞" value={fmtNum(result.linfError)} />
              <MetricCard
                label="R²"
                value={typeof metrics.r2 === "number" ? metrics.r2.toFixed(4) : "—"}
                accent="text-fuchsia-400"
              />
              <MetricCard label="Lignes" value={String(metrics.rows ?? "—")} />
              <MetricCard label="Taille arbre" value={String(result.treeSize ?? "—")} />
              <MetricCard label="Temps de recherche" value={fmtMs(result.durationMs)} />
            </div>
            <XyChart
              title="Données réelles vs. prédiction SPEAR"
              series={[
                { name: "y réel", color: "#38bdf8", points: (result.chartData ?? []).map((p) => ({ x: p.x, y: p.target ?? 0 })) },
                { name: "y prédit", color: "#e879f9", points: (result.chartData ?? []).map((p) => ({ x: p.x, y: p.predicted ?? 0 })) },
              ]}
            />
            <XyChart
              title="Convergence de la fitness par génération"
              series={[
                {
                  name: "Meilleure fitness",
                  color: "#fbbf24",
                  points: (result.history ?? []).map((h) => ({ x: h.generation, y: h.bestFitness })),
                },
              ]}
            />
          </>
        ) : (
          <div className="flex h-full min-h-[300px] items-center justify-center rounded-2xl border border-dashed border-slate-700 text-sm text-slate-500">
            Collez un jeu de données puis lancez SPEAR pour découvrir la formule sous-jacente.
          </div>
        )}
      </div>
    </div>
  );
}
