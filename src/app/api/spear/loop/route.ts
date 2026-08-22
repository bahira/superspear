import { NextRequest, NextResponse } from "next/server";
import { desc, eq } from "drizzle-orm";
import { db } from "@/db";
import { spearBreakthroughs, spearExperiments } from "@/db/schema";
import { runGroundedLoop, type LoopProgress } from "@/lib/spear/loop";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function GET() {
  const rows = await db.select().from(spearExperiments).orderBy(desc(spearExperiments.id)).limit(25);
  return NextResponse.json({ experiments: rows });
}

export async function POST(req: NextRequest) {
  let body: { seed?: number; budget?: number; deadlineMs?: number; label?: string } = {};
  try {
    body = await req.json();
  } catch {
    body = {};
  }

  const budget = Math.max(30, Math.min(2000, Math.floor(Number(body.budget) || 500)));
  const deadlineMs = Math.max(4000, Math.min(40_000, Math.floor(Number(body.deadlineMs) || 25_000)));
  const seed = Number.isFinite(Number(body.seed)) && Number(body.seed) > 0 ? Math.floor(Number(body.seed)) : Math.floor(Math.random() * 1e9);
  const label = (body.label ?? "").trim().slice(0, 180) || `Boucle grounded — budget ${budget}`;

  const [experiment] = await db
    .insert(spearExperiments)
    .values({ label, seed, budget, deadlineMs, status: "running" })
    .returning();

  let lastPersist = 0;
  const persist = async (p: LoopProgress) => {
    const now = Date.now();
    if (now - lastPersist < 450 && p.status === "running") return;
    lastPersist = now;
    await db
      .update(spearExperiments)
      .set({
        status: p.status,
        iterationsUsed: p.iterationsUsed,
        elapsedMs: Math.round(p.elapsedMs),
        snapshot: p,
        totals: p.totals,
        breakthroughCount: p.breakthroughs.length,
        maxLevel: p.totals.maxLevel,
        sumLevels: p.totals.sumLevels,
        tasksBeatingBaseline: p.totals.tasksBeatingBaseline,
        updatedAt: new Date(),
      })
      .where(eq(spearExperiments.id, experiment.id));
  };

  try {
    const progress = await runGroundedLoop({ seed, budget, deadlineMs, onProgress: persist });
    if (progress.breakthroughs.length > 0) {
      await db.insert(spearBreakthroughs).values(
        progress.breakthroughs.slice(-400).map((b) => ({
          experimentId: experiment.id,
          iteration: b.iteration,
          taskId: b.taskId,
          taskTitle: b.taskTitle.slice(0, 200),
          level: b.level,
          kind: b.kind,
          label: b.label,
          formula: b.formula,
          metric: Number.isFinite(b.metric) ? b.metric : null,
          deltaPct: b.deltaPct !== null && b.deltaPct !== undefined && Number.isFinite(b.deltaPct) ? b.deltaPct : null,
          note: b.note ?? null,
        })),
      );
    }
    await db
      .update(spearExperiments)
      .set({
        status: progress.status,
        iterationsUsed: progress.iterationsUsed,
        elapsedMs: Math.round(progress.elapsedMs),
        snapshot: progress,
        totals: progress.totals,
        breakthroughCount: progress.breakthroughs.length,
        maxLevel: progress.totals.maxLevel,
        sumLevels: progress.totals.sumLevels,
        tasksBeatingBaseline: progress.totals.tasksBeatingBaseline,
        updatedAt: new Date(),
      })
      .where(eq(spearExperiments.id, experiment.id));
    return NextResponse.json({ experimentId: experiment.id, progress });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Échec de la boucle.";
    await db
      .update(spearExperiments)
      .set({ status: "failed", updatedAt: new Date() })
      .where(eq(spearExperiments.id, experiment.id));
    return NextResponse.json({ error: message, experimentId: experiment.id }, { status: 500 });
  }
}
