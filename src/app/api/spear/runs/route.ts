import { NextResponse } from "next/server";
import { desc } from "drizzle-orm";
import { db } from "@/db";
import { spearRuns } from "@/db/schema";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const rows = await db.select().from(spearRuns).orderBy(desc(spearRuns.createdAt)).limit(50);
  return NextResponse.json({ runs: rows });
}
