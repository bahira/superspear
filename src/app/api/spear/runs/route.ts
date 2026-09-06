import { NextResponse } from "next/server";
import { desc } from "drizzle-orm";
import { db, isDbEnabled } from "@/db";
import { spearRuns } from "@/db/schema";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  // No database configured is a valid deployment, not an error: run history is
  // optional and the rest of the app does not depend on it.
  if (!isDbEnabled) return NextResponse.json({ runs: [], historyEnabled: false });
  try {
    const rows = await db.select().from(spearRuns).orderBy(desc(spearRuns.createdAt)).limit(50);
    return NextResponse.json({ runs: rows, historyEnabled: true });
  } catch (e) {
    console.error("run history unavailable:", (e as Error).message);
    return NextResponse.json({ runs: [], historyEnabled: false });
  }
}
