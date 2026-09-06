import { desc } from "drizzle-orm";
import { db, isDbEnabled } from "@/db";
import { spearRuns } from "@/db/schema";
import { SpearDashboard } from "@/components/spear/SpearDashboard";
import type { SpearRunRecord } from "@/components/spear/types";

export const dynamic = "force-dynamic";

async function loadRuns() {
  return db.select().from(spearRuns).orderBy(desc(spearRuns.createdAt)).limit(50);
}

export default async function HomePage() {
  // Run history is a nice-to-have; the Hall of Fame is the point. Never let a
  // missing or unreachable database take the whole page down.
  let rows: Awaited<ReturnType<typeof loadRuns>> = [];
  if (isDbEnabled) {
    try {
      rows = await loadRuns();
    } catch (e) {
      console.error("run history unavailable:", (e as Error).message);
    }
  }

  const initialRuns: SpearRunRecord[] = rows.map((r) => ({
    id: r.id,
    preset: r.preset,
    label: r.label,
    config: (r.config as Record<string, unknown>) ?? {},
    status: r.status,
    error: r.error,
    formulaText: r.formulaText,
    fitness: r.fitness,
    mse: r.mse,
    linfError: r.linfError,
    treeSize: r.treeSize,
    durationMs: r.durationMs,
    history: (r.history as SpearRunRecord["history"]) ?? null,
    metrics: (r.metrics as Record<string, unknown>) ?? null,
    chartData: (r.chartData as SpearRunRecord["chartData"]) ?? null,
    createdAt: r.createdAt.toISOString(),
  }));

  return (
    <main className="min-h-screen bg-slate-950">
      <SpearDashboard initialRuns={initialRuns} historyEnabled={isDbEnabled} />
    </main>
  );
}
