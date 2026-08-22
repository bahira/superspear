// SPEAR Hall of Fame — agrège tous les runs et sort le meilleur par tâche.
// Mode 1 (rebuild): npx tsx scripts/hall-of-fame.ts            → parse les logs run*.txt
// Mode 2 (live):    npx tsx scripts/hall-of-fame.ts <seed> <budget> [deadlineMs]
import { readFileSync, writeFileSync, readdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { runGroundedLoop } from "../src/lib/spear/loop";

interface Finding {
  taskId: string;
  title: string;
  direction: "min" | "max";
  metric: number;
  level: number;
  formula: string;
  seed: number;
  iteration: number;
}
type Ledger = Record<string, Finding>;

const LOG_DIR = "C:/Users/Yuri/AppData/Local/Temp/opencode";
const OUT = join(import.meta.dirname ?? ".", "..", "spear-hall-of-fame.json");
const MAX_TASKS: Record<string, "min" | "max"> = { kv_cache: "max" };

function better(dir: "min" | "max", a: number, b: number): boolean {
  return dir === "min" ? a < b : a > b;
}

/** Parse un log texte produit par run-loop.ts → trouvailles par tâche. */
function parseLog(text: string, fallbackSeed: number): { seed: number; finds: Map<string, Finding> } {
  let seed = fallbackSeed;
  const m = text.match(/seed=(\d+) status=/);
  if (m) seed = Number(m[1]);
  const finds = new Map<string, Finding>();
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const hit = lines[i].match(/^\[it\s+(\d+)\]\ (.+?)\s+\|\s+lvl=(\d+)\s+kind=\w+\s+metric=(-?[\d.e+-]+)$/);
    if (!hit) continue;
    const [, it, title, lvl, metricStr] = hit;
    // la ligne "formula:" suit dans les 3 lignes suivantes
    let formula = "";
    for (let j = i + 1; j <= i + 3 && j < lines.length; j++) {
      const fm = lines[j].match(/^\s+formula:\ (.+)$/);
      if (fm) { formula = fm[1]; break; }
    }
    if (!formula) continue;
    const id = titleToId(title);
    const dir = MAX_TASKS[id] ?? "min";
    const metric = Number(metricStr);
    const prev = finds.get(id);
    if (!prev || better(dir, metric, prev.metric)) {
      finds.set(id, { taskId: id, title, direction: dir, metric, level: Number(lvl), formula, seed, iteration: Number(it) });
    }
  }
  return { seed, finds };
}

function titleToId(title: string): string {
  const t = title.toLowerCase();
  if (t.includes("silu")) return "silu";
  if (t.includes("gelu")) return "gelu";
  if (t.includes("sigmoid")) return "sigmoid";
  if (t.includes("gaussienne")) return "gaussian_cdf";
  if (t.includes("kv-cache") || t.includes("éviction")) return "kv_cache";
  if (t.includes("chute libre")) return "free_fall";
  if (t.includes("kepler")) return "kepler";
  if (t.includes("call européen")) return "european_call";
  if (t.includes("pendule")) return "damped_pendulum";
  if (t.includes("distillation")) return "rl_distillation";
  if (t.includes("lambert")) return "lambert_w";
  if (t.includes("circuit rc")) return "rc_circuit";
  if (t.includes("layernorm")) return "layernorm_scale";
  if (t.includes("blur")) return "gaussian_kernel";
  if (t.includes("diffusion")) return "diffusion_beta";
  if (t.includes("upsampling") || t.includes("bilinéaire")) return "bilinear_interp";
  if (t.includes("vidéo")) return "temporal_grad";
  return title.slice(0, 24);
}

function mergeInto(ledger: Ledger, finds: Map<string, Finding>): number {
  let added = 0;
  for (const f of finds.values()) {
    const prev = ledger[f.taskId];
    if (!prev || better(f.direction, f.metric, prev.metric)) {
      ledger[f.taskId] = f;
      added++;
    }
  }
  return added;
}

async function main() {
  const ledger: Ledger = existsSync(OUT) ? JSON.parse(readFileSync(OUT, "utf8")) : {};
  const live = process.argv[2] && process.argv[3];

  if (live) {
    const seed = Number(process.argv[2]);
    const budget = Number(process.argv[3]);
    const deadlineMs = process.argv[4] ? Number(process.argv[4]) : 45_000;
    console.log(`▶ run live seed=${seed} budget=${budget}...`);
    const p = await runGroundedLoop({ seed, budget, deadlineMs });
    const finds = new Map<string, Finding>();
    for (const t of p.tasks) {
      if (!t.best) continue;
      finds.set(t.taskId, {
        taskId: t.taskId,
        title: t.title,
        direction: MAX_TASKS[t.taskId] ?? "min",
        metric: t.best.metric,
        level: t.best.level,
        formula: t.best.formula,
        seed,
        iteration: t.iterations,
      });
    }
    const n = mergeInto(ledger, finds);
    console.log(`terminé: ${p.breakthroughs.length} breakthroughs, ${n} nouveaux records au HOF\n`);
  } else {
    let runs = 0;
    for (const f of readdirSync(LOG_DIR)) {
      if (!/^run.*\.txt$/.test(f)) continue;
      try {
        const text = readFileSync(join(LOG_DIR, f), "utf8");
        if (!text.includes("========== RESULT")) continue;
        const { finds } = parseLog(text, 0);
        mergeInto(ledger, finds);
        runs++;
      } catch { /* log illisible, on passe */ }
    }
    console.log(`${runs} logs historiques fusionnés\n`);
  }

  writeFileSync(OUT, JSON.stringify(ledger, null, 2));

  // ---- affichage
  const rows = Object.values(ledger).sort((a, b) => a.taskId.localeCompare(b.taskId));
  console.log("╔═══════════════════════ HALL OF FAME SPEAR ═══════════════════════╗");
  for (const r of rows) {
    const metricStr = r.direction === "min" ? r.metric.toExponential(2) : r.metric.toFixed(1) + "%";
    console.log(`\n[${r.taskId}] ${r.title}`);
    console.log(`  best=${metricStr}  L${r.level}  trouvé: seed ${r.seed}, it ${r.iteration}`);
    console.log(`  ${r.formula}`);
  }
  console.log(`\n${rows.length} tâches · ledger sauvegardé: spear-hall-of-fame.json`);
}

main().catch((e) => { console.error(e); process.exit(1); });
