/**
 * Export pour le crate Rust `spear-rs` (spear-rs/data/laws.json).
 *
 * Le crate Rust n'a pas accès aux closures `build()` des factories : on exporte
 * donc ce qui suffit à rejouer un kernel chez soi sans le registre TS :
 *   - `variables` : noms de variables dans l'ordre du gpConfig
 *   - `domain`    : intervalle d'échantillonnage déclaré (TaskDef.domain),
 *                   null quand la tâche tire ses entrées d'une loi (aléatoire,
 *                   grille implicite, monde de simulation)
 *   - `law`       : la loi génératrice en AST sérialisé (TaskDef.exactRefNode,
 *                   c'est-à-dire EXACT_LAWS ou l'exactLaw de la tâche)
 *
 * Cible absente => `spear-rs verify` saute la tâche et le dit. Aucune valeur
 * n'est inventée ici : le fichier est une transcription, pas une approximation.
 *
 * Run: npx tsx scripts/export-rust-data.ts
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { buildTasks } from "../src/lib/spear/tasks/index";
import { serializeNode, type SpearNode } from "../src/lib/spear/engine";

const ROOT = join(import.meta.dirname ?? ".", "..");
const OUT_DIR = join(ROOT, "spear-rs", "data");

function main(): void {
  const tasks = buildTasks();
  mkdirSync(OUT_DIR, { recursive: true });

  const entries = tasks.map((t) => ({
    id: t.id,
    family: t.family,
    variables: t.variables,
    domain: t.domain ?? null,
    law: t.exactRefNode ? serializeNode(t.exactRefNode as SpearNode) : null,
  }));

  const withLaw = entries.filter((e) => e.law !== null).length;
  const withDomain = entries.filter((e) => e.domain !== null).length;

  writeFileSync(
    join(OUT_DIR, "laws.json"),
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        note: "Transcription des TaskDef vivantes (voir scripts/export-rust-data.ts). law = loi génératrice en AST; null = pas de forme close.",
        taskCount: entries.length,
        withLaw,
        withDomain,
        tasks: entries,
      },
      null,
      2,
    ) + "\n",
  );

  console.log(`${entries.length} tâches écrites -> ${join(OUT_DIR, "laws.json")}`);
  console.log(`  avec loi AST : ${withLaw}/${entries.length}`);
  console.log(`  avec domaine : ${withDomain}/${entries.length}`);
  const noLaw = entries.filter((e) => e.law === null).map((e) => e.id);
  if (noLaw.length) console.log(`  sans loi (verify les sautera) : ${noLaw.join(", ")}`);
}

main();
