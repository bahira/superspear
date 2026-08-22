"use client";

import { useState } from "react";
import { fmtDate, fmtMs, PRESET_COLORS, PRESET_LABELS } from "./format";
import type { SpearRunRecord } from "./types";

export function RunHistory({
  runs,
  onDelete,
}: {
  runs: SpearRunRecord[];
  onDelete: (id: number) => void;
}) {
  const [deletingId, setDeletingId] = useState<number | null>(null);

  async function handleDelete(id: number) {
    setDeletingId(id);
    try {
      const res = await fetch(`/api/spear/runs/${id}`, { method: "DELETE" });
      if (res.ok) onDelete(id);
    } finally {
      setDeletingId(null);
    }
  }

  if (runs.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-700 p-8 text-center text-sm text-slate-500">
        Aucune exécution SPEAR enregistrée pour le moment. Lancez un laboratoire ci-dessus.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-2xl border border-slate-800">
      <table className="w-full min-w-[720px] text-left text-sm">
        <thead className="bg-slate-900/70 text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-4 py-3">Laboratoire</th>
            <th className="px-4 py-3">Libellé</th>
            <th className="px-4 py-3">Formule</th>
            <th className="px-4 py-3">Durée</th>
            <th className="px-4 py-3">Date</th>
            <th className="px-4 py-3" />
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {runs.map((run) => (
            <tr key={run.id} className="text-slate-300">
              <td className="px-4 py-3">
                <span
                  className={`rounded-full border px-2 py-0.5 text-xs font-medium ${
                    PRESET_COLORS[run.preset] ?? "border-slate-600 text-slate-300"
                  }`}
                >
                  {PRESET_LABELS[run.preset] ?? run.preset}
                </span>
              </td>
              <td className="px-4 py-3">{run.label}</td>
              <td className="max-w-[280px] truncate px-4 py-3 font-mono text-xs text-slate-400" title={run.formulaText ?? ""}>
                {run.formulaText ?? "—"}
              </td>
              <td className="px-4 py-3 text-slate-400">{fmtMs(run.durationMs)}</td>
              <td className="px-4 py-3 text-slate-400">{fmtDate(run.createdAt)}</td>
              <td className="px-4 py-3 text-right">
                <button
                  onClick={() => handleDelete(run.id)}
                  disabled={deletingId === run.id}
                  className="rounded-md px-2 py-1 text-xs text-rose-400 transition hover:bg-rose-500/10 disabled:opacity-50"
                >
                  Supprimer
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
