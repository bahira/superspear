import { desc } from "drizzle-orm";
import { db } from "@/db";
import { spearRuns } from "@/db/schema";
import { SpearDashboard } from "@/components/spear/SpearDashboard";
import type { SpearRunRecord } from "@/components/spear/types";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const rows = await db.select().from(spearRuns).orderBy(desc(spearRuns.createdAt)).limit(50);

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
      <SpearDashboard initialRuns={initialRuns} />
    </main>
  );
}
