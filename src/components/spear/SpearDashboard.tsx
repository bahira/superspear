"use client";

import { useMemo, useState } from "react";
import { ActivationLab } from "./ActivationLab";
import { KvCacheLab } from "./KvCacheLab";
import { CustomRegressionLab } from "./CustomRegressionLab";
import { RunHistory } from "./RunHistory";
import { GroundedLoopConsole } from "./GroundedLoopConsole";
import type { SpearRunRecord } from "./types";

type Tab = "loop" | "activation" | "kv_cache" | "custom";

const TABS: { id: Tab; label: string; emoji: string }[] = [
  { id: "loop", label: "Boucle grounded", emoji: "🔬" },
  { id: "activation", label: "Activations LLM", emoji: "⚡" },
  { id: "kv_cache", label: "Éviction KV-Cache", emoji: "🧠" },
  { id: "custom", label: "Régression symbolique", emoji: "🧮" },
];

export function SpearDashboard({ initialRuns }: { initialRuns: SpearRunRecord[] }) {
  const [tab, setTab] = useState<Tab>("loop");
  const [runs, setRuns] = useState<SpearRunRecord[]>(initialRuns);

  function handleRunComplete(run: SpearRunRecord) {
    setRuns((prev) => [run, ...prev].slice(0, 50));
  }

  function handleDelete(id: number) {
    setRuns((prev) => prev.filter((r) => r.id !== id));
  }

  const stats = useMemo(() => {
    const total = runs.length;
    const activationRuns = runs.filter((r) => r.preset === "activation" && typeof r.metrics?.speedup === "number");
    const avgSpeedup =
      activationRuns.length > 0
        ? activationRuns.reduce((s, r) => s + Number(r.metrics!.speedup), 0) / activationRuns.length
        : null;
    const totalGenerations = runs.reduce((s, r) => s + (r.history?.length ?? 0), 0);
    return { total, avgSpeedup, totalGenerations };
  }, [runs]);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-8 px-6 py-10">
      <header className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-emerald-500/15 text-2xl">
            🎯
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-emerald-400">
              SPEAR Lab · Symbolic Pareto Evolutionary Algorithm for Research
            </p>
            <h1 className="text-2xl font-bold text-slate-50 sm:text-3xl">
              Utilisez SPEAR pour remplacer le calcul lourd par des formules découvertes automatiquement
            </h1>
          </div>
        </div>
        <p className="max-w-3xl text-sm text-slate-400">
          SPEAR est un moteur de programmation génétique multi-objectifs (précision vs. taille de formule) qui
          évolue des expressions symboliques fermées pour remplacer des calculs coûteux : fonctions
          d&apos;activation transcendantes des LLM, règles d&apos;éviction de KV-cache pour l&apos;inférence
          longue-contexte, ou toute loi cachée dans un jeu de données. Chaque exécution est journalisée en base
          PostgreSQL ci-dessous.
        </p>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
            <p className="text-xs uppercase tracking-wide text-slate-500">Exécutions enregistrées</p>
            <p className="mt-1 text-2xl font-bold text-slate-100">{stats.total}</p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
            <p className="text-xs uppercase tracking-wide text-slate-500">Accélération moyenne (activations)</p>
            <p className="mt-1 text-2xl font-bold text-emerald-400">
              {stats.avgSpeedup ? `${stats.avgSpeedup.toFixed(2)}×` : "—"}
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
            <p className="text-xs uppercase tracking-wide text-slate-500">Générations évoluées (cumulé)</p>
            <p className="mt-1 text-2xl font-bold text-slate-100">{stats.totalGenerations.toLocaleString("fr-FR")}</p>
          </div>
        </div>
      </header>

      <nav className="flex flex-wrap gap-2 border-b border-slate-800 pb-2">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`rounded-t-lg px-4 py-2 text-sm font-medium transition ${
              tab === t.id
                ? "bg-slate-900 text-emerald-300 shadow-[inset_0_-2px_0_0_theme(colors.emerald.400)]"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {t.emoji} {t.label}
          </button>
        ))}
      </nav>

      <section>
        {tab === "loop" ? <GroundedLoopConsole /> : null}
        {tab === "activation" ? <ActivationLab onRunComplete={handleRunComplete} /> : null}
        {tab === "kv_cache" ? <KvCacheLab onRunComplete={handleRunComplete} /> : null}
        {tab === "custom" ? <CustomRegressionLab onRunComplete={handleRunComplete} /> : null}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-slate-100">Historique des exécutions</h2>
        <RunHistory runs={runs} onDelete={handleDelete} />
      </section>
    </div>
  );
}
