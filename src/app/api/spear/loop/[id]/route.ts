import { NextResponse } from "next/server";
import { desc, eq } from "drizzle-orm";
import { db } from "@/db";
import { spearBreakthroughs, spearExperiments } from "@/db/schema";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const numericId = Number(id);
  if (!Number.isInteger(numericId)) {
    return NextResponse.json({ error: "Identifiant invalide." }, { status: 400 });
  }
  const [experiment] = await db.select().from(spearExperiments).where(eq(spearExperiments.id, numericId));
  if (!experiment) return NextResponse.json({ error: "Expérience introuvable." }, { status: 404 });
  const breakthroughs = await db
    .select()
    .from(spearBreakthroughs)
    .where(eq(spearBreakthroughs.experimentId, numericId))
    .orderBy(desc(spearBreakthroughs.id))
    .limit(200);
  return NextResponse.json({ experiment, breakthroughs });
}

export async function DELETE(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const numericId = Number(id);
  if (!Number.isInteger(numericId)) {
    return NextResponse.json({ error: "Identifiant invalide." }, { status: 400 });
  }
  await db.delete(spearBreakthroughs).where(eq(spearBreakthroughs.experimentId, numericId));
  await db.delete(spearExperiments).where(eq(spearExperiments.id, numericId));
  return NextResponse.json({ ok: true });
}
