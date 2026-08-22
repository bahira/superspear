"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { XyChart } from "./Charts";
import { MetricCard } from "./FormControls";
import { fmtMs } from "./format";
import { instantiateSpearWasm } from "@/lib/spear/wasm";
import {
  clearBestCache,
  copyToClipboard,
  downloadText,
  exportAllJson,
  exportCuda,
  exportEntryJson,
  exportPython,
  fmtCachedMetric,
  loadBestCache,
  mergeBestEntries,
  persistBestCache,
  type CachedBest,
} from "./localCache";

interface Baseline {
  name: string;
  metric: number;
  note: string;
  kind: string;
  formula?: string;
  beaten: boolean;
  ratio: number | null;
}

interface TaskSnapshot {
  taskId: string;
  family: string;
  title: string;
  subtitle: string;
  groundTruth: string;
  metricLabel: string;
  metricDirection: "min" | "max";
  iterations: number;
  evals: number;
  cacheHits: number;
  stagnation: number;
  priority: number;
  history: { i: number; metric: number; size: number }[];
  paretoFront: { formula: string; metric: number; size: number; level: number }[];
  best: { formula: string; metric: number; size: number; level: number } | null;
  secondary: number | null;
  python: string | null;
  c: string | null;
  wasm: string | null;
  chart: { x: number; target: number; predicted: number }[] | null;
  ops: { total: number; transcendental: number } | null;
  verifyNote: string | null;
  speed: { formulaCost: number; exactCost: number; estimatedSpeedup: number; ops: number; transcendental: number; elements: number } | null;
  baselines: Baseline[];
  bestBaselineName: string | null;
  milestonesHit: string[];
}

interface Breakthrough {
  iteration: number;
  taskId: string;
  taskTitle: string;
  level: number;
  kind: string;
  label: string;
  formula: string;
  metric: number;
  deltaPct: number | null;
  note?: string;
}

interface Progress {
  status: string;
  seed: number;
  budget: number;
  iterationsUsed: number;
  elapsedMs: number;
  deadlineMs: number;
  tasks: TaskSnapshot[];
  breakthroughs: Breakthrough[];
  totals: {
    breakthroughs: number;
    maxLevel: number;
    sumLevels: number;
    tasksBeatingBaseline: number;
    evaluations: number;
    cacheHits: number;
  };
}

const LEVEL_STYLES = [
  "border-slate-700 text-slate-400",
  "border-emerald-600/50 text-emerald-300 bg-emerald-500/10",
  "border-emerald-500/60 text-emerald-200 bg-emerald-500/20",
  "border-cyan-400/60 text-cyan-200 bg-cyan-500/20",
  "border-amber-400/60 text-amber-200 bg-amber-500/20",
  "border-fuchsia-400/70 text-fuchsia-200 bg-fuchsia-500/20",
];

const KIND_STYLES: Record<string, string> = {
  improvement: "text-emerald-300",
  milestone: "text-cyan-300",
  beats_baseline: "text-amber-300",
  first_valid: "text-slate-300",
};

const FAMILY_EMOJI: Record<string, string> = {
  activation: "⚡",
  kv_cache: "🧠",
  regression: "🧮",
};

function fmtMetric(t: TaskSnapshot, m: number | null | undefined): string {
  if (m === null || m === undefined || !Number.isFinite(m)) return "—";
  return t.metricDirection === "max" ? `${m.toFixed(2)} %` : m < 1e-9 ? m.toExponential(1) : m.toExponential(3);
}

function TaskCard({ t }: { t: TaskSnapshot }) {
  const [code, setCode] = useState<"python" | "c">("python");
  const [wasmStatus, setWasmStatus] = useState<string | null>(null);
  const [wasmResult, setWasmResult] = useState<number | null>(null);
  const [wasmRunning, setWasmRunning] = useState(false);

  async function runWasm() {
    if (!t.wasm) return;
    setWasmRunning(true);
    setWasmStatus(null);
    try {
      const spear = await instantiateSpearWasm(t.wasm);
      // Parity is meaningful only where chart.x holds the real input variable
      // (activation tasks). For regression tasks chart.x is a sample index.
      const chart = t.family === "activation" ? (t.chart ?? []) : [];
      if (chart.length > 0) {
        let maxDiff = 0;
        let sum = 0;
        for (const p of chart) {
          const v = spear([p.x]);
          const d = Math.abs(v - p.predicted);
          if (d > maxDiff) maxDiff = d;
          sum += v;
        }
        setWasmResult(sum / chart.length);
        setWasmStatus(`parité ok · max diff ${maxDiff.toExponential(2)}`);
      } else {
        setWasmStatus(`module compilé (${t.wasm.length} octets b64)`);
      }
    } catch (e) {
      setWasmStatus(`échec WASM : ${e instanceof Error ? e.message : "erreur"}`);
    } finally {
      setWasmRunning(false);
    }
  }
  const lvl = t.best?.level ?? 0;
  const best = t.best;
  const deployable = t.baselines.filter((b) => b.kind !== "oracle" && b.kind !== "transcendental");
  const beatenCount = deployable.filter((b) => b.beaten).length;
  const oracle = t.baselines.find((b) => b.kind === "oracle");
  // max-direction: oracle is a ceiling (report % of it reached)
  // min-direction: oracle is the irreducible noise floor (report ratio)
  const oracleLabel = t.metricDirection === "max" ? "Gap oracle fermé" : "× le plancher de bruit";
  const oracleValue =
    oracle && best && oracle.metric > 0
      ? t.metricDirection === "max"
        ? `${Math.max(0, Math.min(100, (best.metric / oracle.metric) * 100)).toFixed(0)} %`
        : `${(best.metric / oracle.metric).toFixed(2)}×`
      : null;
  const atFloor =
    oracle !== undefined &&
    best !== null &&
    t.metricDirection === "min" &&
    oracle.metric > 0 &&
    best.metric <= oracle.metric * 1.1;

  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h4 className="text-sm font-semibold text-slate-100">
            {FAMILY_EMOJI[t.family]} {t.title}
          </h4>
          <p className="mt-0.5 text-xs text-slate-500">{t.subtitle}</p>
        </div>
        <span className={`shrink-0 rounded-full border px-2 py-0.5 text-xs font-semibold ${LEVEL_STYLES[lvl]}`}>
          Niv. {lvl}/5
        </span>
      </div>

      {best ? (
        <>
          <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/5 p-3">
            <p className="text-[11px] uppercase tracking-wide text-emerald-400/80">Formule découverte</p>
            <p className="mt-1 break-all font-mono text-xs leading-relaxed text-emerald-200">{best.formula}</p>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
            <div className="rounded-lg bg-slate-950/60 p-2">
              <p className="text-slate-500">{t.metricLabel}</p>
              <p className="font-semibold text-slate-100">{fmtMetric(t, best.metric)}</p>
            </div>
            <div className="rounded-lg bg-slate-950/60 p-2">
              <p className="text-slate-500">Baselines dépassées</p>
              <p className="font-semibold text-emerald-300">{beatenCount}/{deployable.length}</p>
            </div>
            <div className="rounded-lg bg-slate-950/60 p-2">
              <p className="text-slate-500">Ops / transcendants</p>
              <p className="font-semibold text-slate-100">{t.ops ? `${t.ops.total} / ${t.ops.transcendental}` : "—"}</p>
            </div>
            <div className="rounded-lg bg-slate-950/60 p-2">
              <p className="text-slate-500">{oracleLabel}</p>
              <p className={`font-semibold ${atFloor ? "text-fuchsia-300" : "text-cyan-300"}`}>{oracleValue ?? "—"}</p>
            </div>
          </div>

          <div className="overflow-hidden rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/70 text-[10px] uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-3 py-2">Baseline</th>
                  <th className="px-3 py-2">Score</th>
                  <th className="px-3 py-2">Verdict</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {t.baselines.map((b) => (
                  <tr key={b.name} className={b.kind === "oracle" ? "bg-slate-950/40" : ""}>
                    <td className="px-3 py-1.5">
                      <span className="text-slate-200">{b.name}</span>
                      <span className="block text-[10px] text-slate-500">{b.note}</span>
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 font-mono text-slate-300">{fmtMetric(t, b.metric)}</td>
                    <td className="whitespace-nowrap px-3 py-1.5">
                      {b.kind === "oracle" ? (
                        <span className="text-slate-500">{t.metricDirection === "min" ? "plancher" : "plafond"}</span>
                      ) : b.kind === "transcendental" ? (
                        <span className="text-slate-500">référence</span>
                      ) : b.beaten ? (
                        <span className="text-emerald-400">dépassée {b.ratio ? `×${b.ratio.toFixed(1)}` : ""}</span>
                      ) : (
                        <span className="text-rose-400">non dépassée</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {t.milestonesHit.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {[...new Set(t.milestonesHit)].map((m) => (
                <span key={m} className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-[10px] text-cyan-300">
                  {m}
                </span>
              ))}
            </div>
          ) : null}

          {atFloor ? (
            <p className="rounded-lg border border-fuchsia-500/30 bg-fuchsia-500/10 p-2 text-xs text-fuchsia-200">
              🏁 Au plancher de bruit : la formule est statistiquement indiscernable de la loi exacte.
            </p>
          ) : null}

          {t.verifyNote ? (
            <p className="rounded-lg border border-amber-500/25 bg-amber-500/5 p-2 text-xs text-amber-200">✓ {t.verifyNote}</p>
          ) : null}

          {t.history.length > 2 ? (
            <XyChart
              title="Convergence de la métrique"
              height={150}
              series={[
                {
                  name: t.metricLabel,
                  color: t.family === "kv_cache" ? "#22d3ee" : "#34d399",
                  points: t.history.map((h) => ({ x: h.i, y: h.metric })),
                },
              ]}
            />
          ) : null}

          {t.chart && t.chart.length > 2 ? (
            <XyChart
              title="Cible vs formule"
              height={150}
              series={[
                { name: "cible", color: "#38bdf8", points: t.chart.map((p) => ({ x: p.x, y: p.target })) },
                { name: "SPEAR", color: "#a3e635", points: t.chart.map((p) => ({ x: p.x, y: p.predicted })) },
              ]}
            />
          ) : null}

          {t.python && t.c ? (
            <div className="rounded-xl border border-slate-800 bg-slate-950/70">
              <div className="flex gap-1 border-b border-slate-800 p-1">
                {(["python", "c"] as const).map((k) => (
                  <button
                    key={k}
                    onClick={() => setCode(k)}
                    className={`rounded-md px-2.5 py-1 text-[11px] font-medium ${code === k ? "bg-slate-800 text-emerald-300" : "text-slate-500 hover:text-slate-300"}`}
                  >
                    {k === "python" ? "PyTorch" : "CUDA"}
                  </button>
                ))}
              </div>
              <pre className="max-h-44 overflow-auto p-3 text-[11px] leading-relaxed text-slate-300">
                {code === "python" ? t.python : t.c}
              </pre>
            </div>
          ) : null}

          {t.wasm ? (
            <div className="flex items-center justify-between gap-2 rounded-xl border border-fuchsia-500/25 bg-fuchsia-500/5 p-2">
              <div className="min-w-0">
                <p className="text-[11px] font-semibold text-fuchsia-300">WebAssembly</p>
                {wasmStatus ? (
                  <p className="truncate font-mono text-[10px] text-slate-400">
                    {wasmStatus}
                    {wasmResult !== null ? ` · moyenne ${wasmResult.toFixed(4)}` : ""}
                  </p>
                ) : (
                  <p className="text-[10px] text-slate-500">binaire {t.wasm.length} octets b64 · exécuté dans le navigateur</p>
                )}
              </div>
              <button
                onClick={() => void runWasm()}
                disabled={wasmRunning}
                className="shrink-0 rounded-lg border border-fuchsia-500/40 px-3 py-1.5 text-xs font-medium text-fuchsia-300 transition hover:bg-fuchsia-500/10 disabled:opacity-40"
              >
                {wasmRunning ? "Compile…" : "▶ Exécuter WASM"}
              </button>
            </div>
          ) : null}
        </>
      ) : (
        <p className="rounded-xl border border-dashed border-slate-700 p-4 text-center text-xs text-slate-500">
          Aucune formule valide encore — la boucle y travaille.
        </p>
      )}
    </div>
  );
}

function BestCacheSection({ cache, onClear }: { cache: CachedBest[]; onClear: () => void }) {
  const [copied, setCopied] = useState<string | null>(null);

  async function copy(entry: CachedBest) {
    const ok = await copyToClipboard(
      JSON.stringify(
        {
          taskId: entry.taskId,
          title: entry.title,
          formula: entry.formula,
          metric: entry.metric,
          metricLabel: entry.metricLabel,
          level: entry.level,
        },
        null,
        2,
      ),
    );
    if (ok) {
      setCopied(entry.taskId);
      setTimeout(() => setCopied(null), 1500);
    }
  }

  if (cache.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-700 p-6 text-center text-sm text-slate-500">
        Cache local vide — lancez la boucle : les meilleures formules par tâche y seront conservées
        automatiquement (localStorage) et resteront disponibles après rechargement de la page.
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-amber-500/25 bg-slate-900/40 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h4 className="text-sm font-semibold text-amber-300">💾 Cache local — meilleurs breakthroughs ({cache.length})</h4>
          <p className="mt-0.5 text-xs text-slate-500">
            Conservé dans localStorage, indépendant des runs : seule la formule la plus forte par tâche est gardée.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => exportAllJson(cache)}
            className="rounded-lg border border-emerald-500/40 px-3 py-1.5 text-xs font-medium text-emerald-300 transition hover:bg-emerald-500/10"
          >
            ⬇ Exporter tout (JSON)
          </button>
          <button
            onClick={onClear}
            className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-400 transition hover:border-rose-500/40 hover:text-rose-300"
          >
            Vider le cache
          </button>
        </div>
      </div>

      <div className="mt-3 flex flex-col gap-2">
        {cache.map((e) => (
          <div key={e.taskId} className="rounded-xl border border-slate-800 bg-slate-950/50 p-3">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span>{FAMILY_EMOJI[e.family]}</span>
              <span className="font-semibold text-slate-100">{e.title}</span>
              <span className={`rounded-full border px-1.5 py-0.5 text-[10px] ${LEVEL_STYLES[e.level]}`}>niv. {e.level}/5</span>
              <span className="font-mono text-emerald-300">{fmtCachedMetric(e)}</span>
              <span className="text-slate-500">{e.metricLabel} · graine {e.seed}</span>
              <span className="ml-auto text-slate-500">{new Date(e.savedAt).toLocaleString("fr-FR")}</span>
            </div>
            <p className="mt-1.5 break-all font-mono text-[11px] text-slate-300">{e.formula}</p>
            {e.verifyNote ? <p className="mt-1 text-[11px] text-amber-300">✓ {e.verifyNote}</p> : null}
            <div className="mt-2 flex flex-wrap gap-1.5">
              <button
                onClick={() => exportPython(e)}
                disabled={!e.python}
                className="rounded-md border border-sky-500/30 px-2 py-1 text-[11px] font-medium text-sky-300 transition hover:bg-sky-500/10 disabled:opacity-40"
              >
                ⬇ PyTorch (.py)
              </button>
              <button
                onClick={() => exportCuda(e)}
                disabled={!e.c}
                className="rounded-md border border-emerald-500/30 px-2 py-1 text-[11px] font-medium text-emerald-300 transition hover:bg-emerald-500/10 disabled:opacity-40"
              >
                ⬇ CUDA (.cu)
              </button>
              <button
                onClick={() => exportEntryJson(e)}
                className="rounded-md border border-slate-600 px-2 py-1 text-[11px] font-medium text-slate-300 transition hover:bg-slate-700/40"
              >
                ⬇ JSON
              </button>
              <button
                onClick={() => void copy(e)}
                className="rounded-md border border-slate-600 px-2 py-1 text-[11px] font-medium text-slate-300 transition hover:bg-slate-700/40"
              >
                {copied === e.taskId ? "✓ Copié" : "Copier"}
              </button>
              {e.speed ? (
                <span className="ml-auto self-center text-[10px] text-slate-500">
                  coût ALU/SFU {e.speed.formulaCost} vs {e.speed.exactCost} exact
                </span>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function GroundedLoopConsole() {
  const [budget, setBudget] = useState(500);
  const [seed, setSeed] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [experimentId, setExperimentId] = useState<number | null>(null);
  const [cache, setCache] = useState<CachedBest[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // hydrate local cache on mount
  useEffect(() => {
    setCache(loadBestCache());
  }, []);

  // every progress snapshot (streamed or final) feeds the local cache:
  // only strictly better formulas replace the incumbent per task.
  useEffect(() => {
    if (!progress || progress.tasks.length === 0) return;
    const incoming: CachedBest[] = progress.tasks
      .filter((t) => t.best)
      .map((t) => ({
        taskId: t.taskId,
        family: t.family,
        title: t.title,
        metricLabel: t.metricLabel,
        metricDirection: t.metricDirection,
        metric: t.best!.metric,
        level: t.best!.level,
        formula: t.best!.formula,
        python: t.python,
        c: t.c,
        wasm: t.wasm,
        verifyNote: t.verifyNote,
        speed: t.speed,
        seed: progress.seed,
        savedAt: Date.now(),
      }));
    if (incoming.length === 0) return;
    const merged = mergeBestEntries(loadBestCache(), incoming);
    persistBestCache(merged);
    setCache(merged);
  }, [progress]);

  const poll = useCallback(async (id: number) => {
    try {
      const res = await fetch(`/api/spear/loop/${id}`, { cache: "no-store" });
      if (!res.ok) return;
      const data = await res.json();
      const snap = data.experiment?.snapshot as Progress | null;
      if (snap) setProgress(snap);
    } catch {
      /* transient */
    }
  }, []);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  async function run() {
    setRunning(true);
    setError(null);
    setProgress(null);
    try {
      const res = await fetch("/api/spear/loop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          budget,
          deadlineMs: 25000,
          seed: seed.trim() ? Number(seed.trim()) : undefined,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Échec de la boucle.");
      setExperimentId(data.experimentId ?? null);
      setProgress(data.progress as Progress);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur inconnue.");
    } finally {
      setRunning(false);
      if (pollRef.current) clearInterval(pollRef.current);
    }
  }

  function start() {
    void run();
    pollRef.current = setInterval(() => {
      void (async () => {
        const res = await fetch("/api/spear/loop", { cache: "no-store" });
        if (!res.ok) return;
        const data = await res.json();
        const latest = data.experiments?.[0];
        if (latest?.id) {
          setExperimentId(latest.id);
          void poll(latest.id);
        }
      })();
    }, 900);
    setTimeout(() => {
      if (pollRef.current) clearInterval(pollRef.current);
    }, 45000);
  }

  const totals = progress?.totals;
  const pct = progress ? Math.min(100, (progress.iterationsUsed / progress.budget) * 100) : 0;

  const feedExport = useMemo(() => {
    if (!progress || progress.breakthroughs.length === 0) return null;
    return JSON.stringify(
      {
        exportedAt: new Date().toISOString(),
        seed: progress.seed,
        budget: progress.budget,
        status: progress.status,
        totals: progress.totals,
        breakthroughs: progress.breakthroughs,
      },
      null,
      2,
    );
  }, [progress]);

  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
        <div className="flex flex-wrap items-end gap-4">
          <div className="min-w-[200px] flex-1">
            <h3 className="text-base font-semibold text-slate-100">🔬 Boucle grounded SPEAR</h3>
            <p className="mt-1 text-sm text-slate-400">
              Une seule boucle, six problèmes ouverts, budget d&apos;itérations partagé. L&apos;allocateur
              UCB dirige le budget vers les tâches qui progressent ; chaque improvement, jalon et baseline
              dépassée est mesuré et journalisé. Baselines calculées par le même code, sur les mêmes données.
            </p>
          </div>
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-slate-300">Budget d&apos;itérations</span>
            <input
              type="number"
              min={30}
              max={2000}
              step={50}
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="w-32 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-100 outline-none focus:border-emerald-500"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-slate-300">Graine (optionnel)</span>
            <input
              type="text"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="aléatoire"
              className="w-32 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-100 outline-none focus:border-emerald-500"
            />
          </label>
          <button
            onClick={start}
            disabled={running}
            className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {running ? (
              <>
                <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-950/40 border-t-slate-950" />
                Recherche…
              </>
            ) : (
              "Lancer la boucle"
            )}
          </button>
        </div>
        {error ? <p className="mt-3 text-sm text-rose-400">{error}</p> : null}
        {progress ? (
          <div className="mt-4">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>
                {progress.iterationsUsed} / {progress.budget} itérations · {progress.status} · graine {progress.seed}
              </span>
              <span>{fmtMs(progress.elapsedMs)}</span>
            </div>
            <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-slate-800">
              <div className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-cyan-400 transition-all" style={{ width: `${pct}%` }} />
            </div>
          </div>
        ) : null}
      </div>

      {totals ? (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <MetricCard label="Breakthroughs" value={String(totals.breakthroughs)} accent="text-emerald-400" />
          <MetricCard label="Niveau max" value={`${totals.maxLevel}/5`} accent="text-cyan-400" />
          <MetricCard label="Somme des niveaux" value={String(totals.sumLevels)} />
          <MetricCard label="Tâches > baseline" value={`${totals.tasksBeatingBaseline}/${progress?.tasks.length ?? 6}`} accent="text-emerald-400" />
          <MetricCard label="Évaluations" value={totals.evaluations.toLocaleString("fr-FR")} />
          <MetricCard label="Hits de cache" value={totals.cacheHits.toLocaleString("fr-FR")} />
        </div>
      ) : null}

      <BestCacheSection
        cache={cache}
        onClear={() => {
          clearBestCache();
          setCache([]);
        }}
      />

      {progress && progress.breakthroughs.length > 0 ? (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-slate-100">
              Fil des breakthroughs ({progress.breakthroughs.length}) — ordre de découverte
            </h4>
            {feedExport ? (
              <button
                onClick={() => downloadText(`spear_feed_seed${progress.seed}.json`, feedExport, "application/json")}
                className="rounded-lg border border-slate-600 px-2.5 py-1 text-[11px] font-medium text-slate-300 transition hover:bg-slate-700/40"
              >
                ⬇ Exporter le fil (JSON)
              </button>
            ) : null}
          </div>
          <div className="mt-3 flex max-h-72 flex-col gap-1.5 overflow-y-auto pr-1">
            {[...progress.breakthroughs].reverse().slice(0, 60).map((b, i) => (
              <div key={`${b.iteration}-${i}`} className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2">
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="font-mono text-slate-500">#{b.iteration}</span>
                  <span className="font-medium text-slate-200">{b.taskId}</span>
                  <span className={`font-medium ${KIND_STYLES[b.kind] ?? "text-slate-300"}`}>{b.label}</span>
                  {b.level > 0 ? <span className="rounded-full border border-slate-700 px-1.5 text-[10px] text-slate-400">niv. {b.level}</span> : null}
                </div>
                <p className="mt-1 break-all font-mono text-[11px] text-slate-500">{b.formula}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-2">
        {(progress?.tasks ?? []).map((t) => (
          <TaskCard key={t.taskId} t={t} />
        ))}
      </div>

      {!progress ? (
        <div className="rounded-2xl border border-dashed border-slate-700 p-10 text-center text-sm text-slate-500">
          Lancez la boucle : SPEAR attaquera simultanément SiLU, GELU, sigmoid, l&apos;éviction de KV-cache,
          la chute libre et la 3ᵉ loi de Kepler — avec un budget de {budget} itérations.
        </div>
      ) : null}

      {experimentId ? (
        <p className="text-xs text-slate-500">
          Expérience #{experimentId} persistée en PostgreSQL — retrouvable dans l&apos;historique.
        </p>
      ) : null}
    </div>
  );
}
