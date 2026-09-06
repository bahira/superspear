import { db, isDbEnabled } from "@/db";
import { sql } from "drizzle-orm";

export const dynamic = "force-dynamic";

export async function GET() {
  // The app is healthy without Postgres — only run history needs it. Reporting
  // 500 here made every uptime check fail on a perfectly working deployment.
  if (!isDbEnabled) return Response.json({ ok: true, database: "not-configured", history: false });
  try {
    await db.execute(sql`select 1`);
    return Response.json({ ok: true, database: "up", history: true });
  } catch (e) {
    return Response.json({ ok: true, database: "unreachable", history: false, detail: (e as Error).message });
  }
}
