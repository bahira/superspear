import type { TaskDef } from "./types";
import * as llm from "./domains/llm";
import * as quantum from "./domains/quantum";
import * as finance from "./domains/finance";
import * as graphics from "./domains/graphics";
import * as physics from "./domains/physics";
import * as rendering from "./domains/rendering";
export { taskOpProfile } from "./shared";

/** Registre extensible: ajouter un domaine = un fichier + une ligne ici. */
const SOURCES = [llm, quantum, finance, graphics, physics, rendering];

export function buildTasks(): TaskDef[] {
  const all: TaskDef[] = SOURCES.flatMap((m) => m.defs());
  const filter = process.env.SPEAR_TASKS;
  if (!filter) return all;
  const keep = new Set(filter.split(","));
  return all.filter((t) => keep.has(t.id));
}