import {
  doublePrecision,
  integer,
  jsonb,
  pgTable,
  serial,
  text,
  timestamp,
  varchar,
} from "drizzle-orm/pg-core";

/** One autonomous SPEAR research loop (multi-task, budgeted, grounded). */
export const spearExperiments = pgTable("spear_experiments", {
  id: serial("id").primaryKey(),
  label: varchar("label", { length: 180 }).notNull(),
  seed: integer("seed").notNull(),
  budget: integer("budget").notNull(),
  deadlineMs: integer("deadline_ms").notNull(),
  status: varchar("status", { length: 24 }).notNull().default("running"),
  iterationsUsed: integer("iterations_used").notNull().default(0),
  elapsedMs: integer("elapsed_ms").notNull().default(0),
  snapshot: jsonb("snapshot"),
  totals: jsonb("totals"),
  breakthroughCount: integer("breakthrough_count").notNull().default(0),
  maxLevel: integer("max_level").notNull().default(0),
  sumLevels: integer("sum_levels").notNull().default(0),
  tasksBeatingBaseline: integer("tasks_beating_baseline").notNull().default(0),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

export type SpearExperiment = typeof spearExperiments.$inferSelect;

/** Every measured breakthrough emitted by a loop, in order of discovery. */
export const spearBreakthroughs = pgTable("spear_breakthroughs", {
  id: serial("id").primaryKey(),
  experimentId: integer("experiment_id").notNull(),
  iteration: integer("iteration").notNull(),
  taskId: varchar("task_id", { length: 40 }).notNull(),
  taskTitle: varchar("task_title", { length: 200 }).notNull(),
  level: integer("level").notNull().default(0),
  kind: varchar("kind", { length: 24 }).notNull(),
  label: text("label").notNull(),
  formula: text("formula").notNull(),
  metric: doublePrecision("metric"),
  deltaPct: doublePrecision("delta_pct"),
  note: text("note"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export type SpearBreakthrough = typeof spearBreakthroughs.$inferSelect;

/** Legacy: manual single-preset lab runs (kept for the per-lab tabs). */
export const spearRuns = pgTable("spear_runs", {
  id: serial("id").primaryKey(),
  preset: varchar("preset", { length: 32 }).notNull(),
  label: varchar("label", { length: 160 }).notNull(),
  config: jsonb("config").notNull(),
  status: varchar("status", { length: 16 }).notNull().default("completed"),
  error: text("error"),
  formulaText: text("formula_text"),
  formulaTree: jsonb("formula_tree"),
  fitness: doublePrecision("fitness"),
  mse: doublePrecision("mse"),
  linfError: doublePrecision("linf_error"),
  treeSize: integer("tree_size"),
  durationMs: integer("duration_ms"),
  history: jsonb("history"),
  metrics: jsonb("metrics"),
  chartData: jsonb("chart_data"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export type SpearRun = typeof spearRuns.$inferSelect;
