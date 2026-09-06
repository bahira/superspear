"use client";

import { useEffect, useMemo, useState } from "react";

/**
 * The Hall of Fame browser.
 *
 * The 89 audited kernels are the project's actual output, and they had no UI at
 * all — you had to read JSON in the repo. This view makes them explorable, and
 * shows the caveats next to the wins rather than only the headline numbers:
 * measured speedup (not the cost model's guess), provenance, and whether a
 * kernel is safe outside its training band.
 */

interface Kernel {
  id: string;
  title: string;
  formula: string;
  mse: number;
  level: number;
  holdout: number | null;
  exact: boolean;
  speedup: number | null;
  speedupModelled: number | null;
  speedupMeasured: number | null;
  speedUnresolvable: boolean | null;
  vsIterative: { label: string; speedup: number } | null;
  cost: number | null;
  fast: { formula: string; mse: number; cost: number | null; speedupMeasured: number | null } | null;
  provenance: string | null;
  r2: number | null;
  risky: string[];
}

interface Payload {
  total: number;
  auditAvailable: boolean;
  summary: Record<string, number>;
  kernels: Kernel[];
}

type Sort = "speed" | "accuracy" | "name";
type Filter = "all" | "exact" | "fast" | "solver" | "search" | "risky";

const FILTERS: { id: Filter; label: string; hint: string }[] = [
  { id: "all", label: "Tous", hint: "les 89 kernels du ledger" },
  { id: "exact", label: "Exacts", hint: "MSE < 1e-25 — précision machine" },
  { id: "fast", label: "Fast slot utile", hint: "gain vitesse mesuré ≥ 1.2×" },
  { id: "solver", label: "Remplace un solveur", hint: "forme fermée au lieu d'une boucle itérative" },
  { id: "search", label: "Trouvés par la recherche", hint: "ni la loi semée, ni une réécriture de celle-ci" },
  { id: "risky", label: "À ne pas déployer", hint: "extrapolent mal hors de leur bande d'entraînement" },
];

function fmt(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  if (n === 0) return "0";
  if (Math.abs(n) < 1e-3 || Math.abs(n) >= 1e5) return n.toExponential(digits);
  return n.toFixed(digits);
}

const PROVENANCE_LABEL: Record<string, { text: string; tone: string }> = {
  search: { text: "trouvé par la recherche", tone: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30" },
  "rewrite-of-law": { text: "réécriture de la loi", tone: "bg-sky-500/15 text-sky-300 ring-sky-500/30" },
  "recovered-seed": { text: "loi semée retrouvée", tone: "bg-slate-500/15 text-slate-300 ring-slate-500/30" },
};

export function HallOfFame() {
  const [data, setData] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<Sort>("speed");
  const [filter, setFilter] = useState<Filter>("all");
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch("/api/spear/kernels")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: Payload) => alive && setData(d))
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, []);

  const rows = useMemo(() => {
    if (!data) return [];
    const needle = q.trim().toLowerCase();
    let out = data.kernels.filter((k) => {
      if (needle && !(k.id.toLowerCase().includes(needle) || k.title.toLowerCase().includes(needle) || k.formula.toLowerCase().includes(needle))) {
        return false;
      }
      switch (filter) {
        case "exact": return k.exact;
        case "fast": return (k.fast?.speedupMeasured ?? 0) >= 1.2;
        case "solver": return Boolean(k.vsIterative);
        case "search": return k.provenance === "search";
        case "risky": return k.risky.length > 0;
        default: return true;
      }
    });
    out = [...out].sort((a, b) => {
      if (sort === "name") return a.id.localeCompare(b.id);
      if (sort === "accuracy") return a.mse - b.mse;
      const av = a.fast?.speedupMeasured ?? a.speedupMeasured ?? 0;
      const bv = b.fast?.speedupMeasured ?? b.speedupMeasured ?? 0;
      return bv - av;
    });
    return out;
  }, [data, q, sort, filter]);

  if (error) {
    return (
      <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-6 text-sm text-red-200">
        Impossible de charger le Hall of Fame : {error}
      </div>
    );
  }
  if (!data) {
    return <div className="animate-pulse rounded-xl border border-slate-800 bg-slate-900/40 p-6 text-sm text-slate-400">Chargement des kernels…</div>;
  }

  const s = data.summary;

  return (
    <section className="flex flex-col gap-5">
      {/* honest headline: what is verified, and what the caveats are */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { k: "exact", label: "exacts", hint: "MSE < 1e-25" },
          { k: "safe", label: "déployables", hint: "aucune violation hors-bande" },
          { k: "fastSlotsWorthIt", label: "fast slots utiles", hint: "gain mesuré ≥ 1.2×" },
          { k: "searchFound", label: "trouvés par recherche", hint: "hors lois semées et réécritures" },
        ].map((c) => (
          <div key={c.k} className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
            <div className="text-2xl font-semibold text-slate-100">{s[c.k] ?? "—"}</div>
            <div className="text-xs font-medium text-slate-300">{c.label}</div>
            <div className="mt-0.5 text-[11px] text-slate-500">{c.hint}</div>
          </div>
        ))}
      </div>

      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Rechercher un kernel, un titre, une formule…"
            className="min-w-[240px] flex-1 rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-emerald-500/50 focus:outline-none"
          />
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as Sort)}
            className="rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2 text-sm text-slate-200 focus:border-emerald-500/50 focus:outline-none"
          >
            <option value="speed">Trier : vitesse mesurée</option>
            <option value="accuracy">Trier : précision</option>
            <option value="name">Trier : nom</option>
          </select>
        </div>

        <div className="flex flex-wrap gap-2">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              onClick={() => setFilter(f.id)}
              title={f.hint}
              className={`rounded-full px-3 py-1.5 text-xs font-medium ring-1 transition ${
                filter === f.id
                  ? "bg-emerald-500/20 text-emerald-200 ring-emerald-500/40"
                  : "bg-slate-900/60 text-slate-400 ring-slate-800 hover:text-slate-200"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <p className="text-xs text-slate-500">{FILTERS.find((f) => f.id === filter)?.hint} — {rows.length} résultat{rows.length > 1 ? "s" : ""}</p>
      </div>

      <div className="overflow-hidden rounded-xl border border-slate-800">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-900/80 text-[11px] uppercase tracking-wide text-slate-400">
            <tr>
              <th className="px-4 py-2.5 font-medium">Kernel</th>
              <th className="px-3 py-2.5 font-medium">MSE</th>
              <th className="px-3 py-2.5 font-medium">Vitesse mesurée</th>
              <th className="px-3 py-2.5 font-medium">Origine</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {rows.map((k) => {
              const isOpen = open === k.id;
              const fastX = k.fast?.speedupMeasured ?? null;
              const prov = k.provenance ? PROVENANCE_LABEL[k.provenance] : null;
              return (
                <>
                  <tr
                    key={k.id}
                    onClick={() => setOpen(isOpen ? null : k.id)}
                    className="cursor-pointer bg-slate-950/40 transition hover:bg-slate-900/60"
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[13px] text-slate-100">{k.id}</span>
                        {k.exact && <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-300">exact</span>}
                        {k.risky.length > 0 && (
                          <span title={k.risky.join(", ")} className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-amber-300">
                            hors-bande
                          </span>
                        )}
                      </div>
                      <div className="mt-0.5 line-clamp-1 text-[11px] text-slate-500">{k.title}</div>
                    </td>
                    <td className="px-3 py-3 font-mono text-[12px] text-slate-300">{fmt(k.mse)}</td>
                    <td className="px-3 py-3">
                      {fastX ? (
                        <span className="font-mono text-[12px] text-emerald-300">×{fastX.toFixed(2)}<span className="ml-1 text-[10px] text-slate-500">fast</span></span>
                      ) : k.speedupMeasured ? (
                        <span className="font-mono text-[12px] text-slate-300">×{k.speedupMeasured.toFixed(2)}</span>
                      ) : (
                        <span className="text-[11px] text-slate-600">non mesurable</span>
                      )}
                    </td>
                    <td className="px-3 py-3">
                      {prov ? (
                        <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium ring-1 ${prov.tone}`}>{prov.text}</span>
                      ) : (
                        <span className="text-[11px] text-slate-600">—</span>
                      )}
                    </td>
                  </tr>
                  {isOpen && (
                    <tr key={`${k.id}-d`} className="bg-slate-900/40">
                      <td colSpan={4} className="px-4 py-4">
                        <div className="flex flex-col gap-3 text-[12px]">
                          <div>
                            <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-500">Formule (précise)</div>
                            <code className="block overflow-x-auto rounded-lg bg-slate-950/80 p-3 font-mono text-[11px] text-emerald-200">{k.formula}</code>
                          </div>
                          {k.fast && (
                            <div>
                              <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-500">
                                Fast slot — MSE {fmt(k.fast.mse)} (erreur ≈ {fmt(Math.sqrt(Math.max(k.fast.mse, 0)))})
                              </div>
                              <code className="block overflow-x-auto rounded-lg bg-slate-950/80 p-3 font-mono text-[11px] text-amber-200">{k.fast.formula}</code>
                            </div>
                          )}
                          <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 sm:grid-cols-4">
                            <Detail label="niveau" value={`L${k.level}`} />
                            <Detail label="holdout" value={fmt(k.holdout)} />
                            <Detail label="r²" value={k.r2 !== null ? fmt(k.r2, 4) : "—"} />
                            <Detail label="coût ALU" value={k.cost !== null ? String(k.cost) : "—"} />
                            {k.speedupModelled !== null && (
                              <Detail label="modélisé" value={`×${k.speedupModelled.toFixed(2)}`} />
                            )}
                            {k.vsIterative && (
                              <Detail label="remplace" value={`${k.vsIterative.label}`} />
                            )}
                          </div>
                          {k.speedUnresolvable && (
                            <p className="text-[11px] text-slate-500">
                              Vitesse non résoluble : les deux kernels sont au plancher du coût d&apos;appel, le gain réel est ≈ ×1.
                            </p>
                          )}
                          {k.risky.length > 0 && (
                            <p className="rounded-lg bg-amber-500/10 px-3 py-2 text-[11px] text-amber-200">
                              À ne pas déployer hors de la bande d&apos;entraînement ({k.risky.join(", ")}).
                            </p>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              );
            })}
          </tbody>
        </table>
        {rows.length === 0 && <div className="bg-slate-950/40 px-4 py-10 text-center text-sm text-slate-500">Aucun kernel ne correspond.</div>}
      </div>

      <p className="text-[11px] leading-relaxed text-slate-500">
        Les vitesses affichées sont <strong className="text-slate-400">mesurées</strong> (gcc -O2, médiane de 9×40 passes), pas prédites par le
        modèle de coût — celui-ci s&apos;est révélé faux dans les deux sens sur plusieurs kernels. « Loi semée retrouvée » signifie que le
        champion est la loi de référence elle-même : une validation du moteur, pas une découverte.
      </p>
    </section>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="font-mono text-[12px] text-slate-200">{value}</div>
    </div>
  );
}
