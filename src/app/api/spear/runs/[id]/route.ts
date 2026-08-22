import { NextResponse } from "next/server";
import { eq } from "drizzle-orm";
import { db } from "@/db";
import { spearRuns } from "@/db/schema";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function DELETE(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const numericId = Number(id);
  if (!Number.isInteger(numericId)) {
    return NextResponse.json({ error: "Identifiant invalide." }, { status: 400 });
  }
  await db.delete(spearRuns).where(eq(spearRuns.id, numericId));
  return NextResponse.json({ ok: true });
}
