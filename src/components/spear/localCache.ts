"use client";

// ---------------------------------------------------------------------------
// Local cache of the best SPEAR breakthroughs (localStorage, survives reloads)
// + export helpers (PyTorch / CUDA / JSON / clipboard).
// ---------------------------------------------------------------------------

export interface CachedBest {
  taskId: string;
  family: string;
  title: string;
  metricLabel: string;
  metricDirection: "min" | "max";
  metric: number;
  level: number;
  formula: string;
  python: string | null;
  c: string | null;
  wasm: string | null;
  verifyNote: string | null;
  speed: { formulaCost: number; exactCost: number; estimatedSpeedup: number; ops: number; transcendental: number; elements: number } | null;
  seed: number;
  savedAt: number;
}

const CACHE_KEY = "spear.best.breakthroughs.v1";

export const TASK_ORDER = ["silu", "gelu", "sigmoid", "kv_cache", "free_fall", "kepler", "gaussian_cdf", "european_call", "damped_pendulum", "rl_distillation", "lambert_w", "rc_circuit", "layernorm_scale", "gaussian_kernel", "diffusion_beta", "bilinear_interp", "temporal_grad"];

export function loadBestCache(): CachedBest[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(CACHE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as CachedBest[]) : [];
  } catch {
    return [];
  }
}

function isBetterMetric(dir: "min" | "max", candidate: number, incumbent: number): boolean {
  if (!Number.isFinite(candidate)) return false;
  if (!Number.isFinite(incumbent)) return true;
  return dir === "min" ? candidate < incumbent : candidate > incumbent;
}

/** Merge freshly discovered bests into the local cache, keeping only the
 *  strongest formula per task. Returns the merged list. */
export function mergeBestEntries(existing: CachedBest[], incoming: CachedBest[]): CachedBest[] {
  const byTask = new Map<string, CachedBest>();
  for (const e of existing) byTask.set(e.taskId, e);
  for (const e of incoming) {
    const incumbent = byTask.get(e.taskId);
    if (!incumbent || isBetterMetric(e.metricDirection, e.metric, incumbent.metric)) {
      byTask.set(e.taskId, e);
    }
  }
  return TASK_ORDER.map((id) => byTask.get(id)).filter((e): e is CachedBest => Boolean(e))
    .concat([...byTask.values()].filter((e) => !TASK_ORDER.includes(e.taskId)));
}

export function persistBestCache(entries: CachedBest[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(CACHE_KEY, JSON.stringify(entries));
  } catch {
    // quota exceeded or private mode — cache is best-effort by design
  }
}

export function clearBestCache(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(CACHE_KEY);
  } catch {
    /* best-effort */
  }
}

// ---------------------------------------------------------------- exports

export function downloadText(filename: string, content: string, mime = "text/plain"): void {
  const blob = new Blob([content], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function slug(text: string): string {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 40);
}

export function exportPython(entry: CachedBest): void {
  if (!entry.python) return;
  downloadText(`spear_${entry.taskId}.py`, entry.python, "text/x-python");
}

export function exportCuda(entry: CachedBest): void {
  if (!entry.c) return;
  downloadText(`spear_${entry.taskId}.cu`, entry.c, "text/x-c");
}

export function exportEntryJson(entry: CachedBest): void {
  downloadText(
    `spear_${entry.taskId}_breakthrough.json`,
    JSON.stringify(
      {
        exportedAt: new Date().toISOString(),
        engine: "SPEAR v2 — Symbolic Pareto Evolutionary Algorithm for Research",
        taskId: entry.taskId,
        title: entry.title,
        formula: entry.formula,
        metricLabel: entry.metricLabel,
        metric: entry.metric,
        level: entry.level,
        seed: entry.seed,
        verifyNote: entry.verifyNote,
        costModel: entry.speed,
        code: { python: entry.python, cuda: entry.c, wasm: entry.wasm },
      },
      null,
      2,
    ),
    "application/json",
  );
}

export function exportAllJson(entries: CachedBest[]): void {
  downloadText(
    `spear_breakthroughs_${slug(new Date().toISOString())}.json`,
    JSON.stringify(
      {
        exportedAt: new Date().toISOString(),
        engine: "SPEAR v2",
        count: entries.length,
        entries: entries.map((e) => ({
          taskId: e.taskId,
          title: e.title,
          formula: e.formula,
          metricLabel: e.metricLabel,
          metric: e.metric,
          level: e.level,
          seed: e.seed,
          savedAt: new Date(e.savedAt).toISOString(),
          verifyNote: e.verifyNote,
          code: { python: e.python, cuda: e.c, wasm: e.wasm },
        })),
      },
      null,
      2,
    ),
    "application/json",
  );
}

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      return true;
    } catch {
      return false;
    }
  }
}

export function fmtCachedMetric(entry: CachedBest): string {
  const m = entry.metric;
  if (!Number.isFinite(m)) return "—";
  return entry.metricDirection === "max" ? `${m.toFixed(2)} %` : m < 1e-9 ? m.toExponential(1) : m.toExponential(3);
}
