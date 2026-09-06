import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from "pg";

// Postgres is OPTIONAL.
//
// It only ever stored the run history of the interactive labs. The actual
// asset — 89 audited kernels — lives in ledger/*.json and needs no database.
// But `throw new Error("DATABASE_URL is required")` at module scope meant a
// fresh clone answered HTTP 500 with a stacktrace on the home page: the whole
// app, including the read-only Hall of Fame, was unreachable without setting
// up a server nobody had told you about.
//
// Now the absence of DATABASE_URL degrades gracefully: history is disabled,
// everything else works. Call sites use `isDbEnabled` to decide.
const databaseUrl = process.env.DATABASE_URL;

export const isDbEnabled = Boolean(databaseUrl);

const globalForDb = globalThis as typeof globalThis & {
  __arenaNextJsPostgresqlPool?: Pool;
};

export const pool = databaseUrl
  ? (globalForDb.__arenaNextJsPostgresqlPool ??
    new Pool({
      connectionString: databaseUrl,
    }))
  : null;

if (process.env.NODE_ENV !== "production" && pool) {
  globalForDb.__arenaNextJsPostgresqlPool = pool;
}

/**
 * Drizzle client. Throws only if actually used while Postgres is unconfigured,
 * so importing this module is always safe.
 */
export const db = pool
  ? drizzle(pool)
  : (new Proxy(
      {},
      {
        get() {
          throw new Error(
            "DATABASE_URL is not set — run history is disabled. The Hall of Fame and all labs work without it.",
          );
        },
      },
    ) as ReturnType<typeof drizzle>);
