export function fmtNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1000) return v.toFixed(1);
  return v.toFixed(digits);
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return `${v.toFixed(digits)}%`;
}

export function fmtMs(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v < 1000) return `${v.toFixed(0)} ms`;
  return `${(v / 1000).toFixed(2)} s`;
}

export function fmtDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString("fr-FR", {
      day: "2-digit",
      month: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export const PRESET_LABELS: Record<string, string> = {
  activation: "Activation LLM",
  kv_cache: "KV-Cache",
  custom: "Régression custom",
};

export const PRESET_COLORS: Record<string, string> = {
  activation: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  kv_cache: "bg-cyan-500/15 text-cyan-300 border-cyan-500/30",
  custom: "bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/30",
};
