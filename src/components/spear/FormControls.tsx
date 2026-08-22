"use client";

import type { ReactNode } from "react";

export function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  hint,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
}) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="font-medium text-slate-300">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-100 outline-none focus:border-emerald-500"
      />
      {hint ? <span className="text-xs text-slate-500">{hint}</span> : null}
    </label>
  );
}

export function SelectField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="font-medium text-slate-300">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-100 outline-none focus:border-emerald-500"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function RunButton({
  onClick,
  loading,
  children,
}: {
  onClick: () => void;
  loading: boolean;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
    >
      {loading ? (
        <>
          <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-950/40 border-t-slate-950" />
          Évolution en cours…
        </>
      ) : (
        children
      )}
    </button>
  );
}

export function MetricCard({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
      <p className="text-[11px] uppercase tracking-wide text-slate-500">{label}</p>
      <p className={`mt-1 text-lg font-semibold ${accent ?? "text-slate-100"}`}>{value}</p>
    </div>
  );
}
