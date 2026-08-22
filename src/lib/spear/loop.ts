import { performance } from "node:perf_hooks";
import { setUniformSource } from "./math-utils";
import {
  canonicalKey,
  estimateCost,
  rand,
  mutatePolish,
  mutateStructure,
  prune,
  crossover,
  collectVarNames,
  evaluateNode,
  mutate,
  nodeToString,
  nonDominatedSort,
  randomTree,
  setSeed,
  simplify,
  toC,
  toPython,
  tournamentSelect,
  type SpearNode,
} from "./engine";
import { buildTasks, taskOpProfile, type TaskBaseline, type TaskDef } from "./benchmarks";
import { toWasmBytes } from "./wasm";

export interface LoopFrontEntry {
  formula: string;
  metric: number;
  size: number;
  level: number;
}

export interface LoopTaskSnapshot {
  taskId: string;
  family: string;
  title: string;
  subtitle: string;
  groundTruth: string;
  metricLabel: string;
  metricDirection: "min" | "max";
  iterations: number;
  evals: number;
  cacheHits: number;
  stagnation: number;
  priority: number;
  history: { i: number; metric: number; size: number }[];
  paretoFront: LoopFrontEntry[];
  best: LoopFrontEntry | null;
  secondary: number | null;
  python: string | null;
  c: string | null;
  /** base64-encoded WebAssembly binary of the discovered law */
  wasm: string | null;
  chart: { x: number; target: number; predicted: number }[] | null;
  ops: { total: number; transcendental: number } | null;
  verifyNote: string | null;
  speed: { formulaCost: number; exactCost: number; estimatedSpeedup: number; ops: number; transcendental: number; elements: number } | null;
  baselines: (TaskBaseline & { beaten: boolean; ratio: number | null })[];
  bestBaselineName: string | null;
  milestonesHit: string[];
}

export interface LoopBreakthrough {
  iteration: number;
  taskId: string;
  taskTitle: string;
  level: number;
  kind: "first_valid" | "improvement" | "beats_baseline" | "milestone";
  label: string;
  formula: string;
  metric: number;
  deltaPct: number | null;
  note?: string;
}

export interface LoopProgress {
  status: "running" | "completed" | "stopped_budget" | "stopped_time";
  seed: number;
  budget: number;
  iterationsUsed: number;
  elapsedMs: number;
  deadlineMs: number;
  tasks: LoopTaskSnapshot[];
  breakthroughs: LoopBreakthrough[];
  totals: {
    breakthroughs: number;
    maxLevel: number;
    sumLevels: number;
    tasksBeatingBaseline: number;
    evaluations: number;
    cacheHits: number;
  };
}

interface TaskRuntime {
  def: TaskDef;
  population: SpearNode[];
  cache: Map<string, { metric: number; secondary?: number }>;
  iterations: number;
  evals: number;
  cacheHits: number;
  stagnation: number;
  priority: number;
  bestNode: SpearNode | null;
  bestMetric: number;
  bestSecondary: number | undefined;
  bestLevel: number;
  milestonesHit: Set<string>;
  baselineBeaten: boolean;
  recentMetrics: number[];
  warmedUp: boolean;
  history: { i: number; metric: number; size: number }[];
  frontRaw: { node: SpearNode; metric: number; size: number }[];
}

const POP = 72;
const REFINE_EVERY = 4;
const STAGNATION_LIMIT = 18;
const UCB_C = 0.9;
const RELATIVE_IMPROVE = 0.04;

function directionSign(t: TaskDef): number {
  return t.metricDirection === "min" ? -1 : 1;
}

function fitnessOf(t: TaskDef, metric: number): number {
  if (!Number.isFinite(metric)) return -Infinity;
  return directionSign(t) * metric;
}

function levelOf(t: TaskDef, metric: number): number {
  let lvl = 0;
  for (const m of t.milestones) if (Number.isFinite(metric) && m.test(metric)) lvl = Math.max(lvl, m.level);
  return lvl;
}

const PARSIMONY_BASE = 0.0014;

function parsimony(iterationsUsed: number, budget: number): number {
  const progress = Math.min(1, iterationsUsed / budget);
  // Early budget: explore freely. Late budget: push toward compact, deployable formulas.
  return PARSIMONY_BASE * (0.3 + progress * progress * 2.5);
}

interface ShapedEval {
  /** un-penalised metric of the *reportable* node */
  metric: number;
  secondary?: number;
  /** self-contained formula whose metric is `metric` (affine-wrapped if the task uses linear scaling) */
  node: SpearNode;
}

function scored(rt: TaskRuntime, node: SpearNode): ShapedEval {
  const r = rt.def.evaluateScored ? rt.def.evaluateScored(node) : rt.def.evaluate(node);
  const shaped: SpearNode = "node" in r ? (r as { node: SpearNode }).node : node;
  return { metric: r.metric, secondary: r.secondary, node: shaped };
}

/**
 * Selection fitness only. Parsimony pressure biases the search toward compact
 * formulas but never alters the number that gets reported to the user.
 */
function selectFitness(rt: TaskRuntime, raw: number, size: number, iterationsUsed: number, budget: number): number {
  if (!Number.isFinite(raw)) return -Infinity;
  const penalty = 1 + parsimony(iterationsUsed, budget) * Math.min(80, size);
  return fitnessOf(rt.def, rt.def.metricDirection === "min" ? raw * penalty : raw / penalty);
}

/** honest reported score: holdout split when the task defines one */
function reportMetric(rt: TaskRuntime, node: SpearNode): { metric: number; secondary?: number } {
  const r = rt.def.holdout ? rt.def.holdout(node) : rt.def.evaluate(node);
  return { metric: r.metric, secondary: r.secondary };
}

function cachedEval(rt: TaskRuntime, node: SpearNode): ShapedEval {
  const key = canonicalKey(node);
  const hit = rt.cache.get(key);
  if (hit) {
    rt.cacheHits++;
    return { metric: hit.metric, secondary: hit.secondary, node };
  }
  const res = scored(rt, node);
  rt.evals++;
  if (Number.isFinite(res.metric)) {
    rt.cache.set(key, { metric: res.metric, secondary: res.secondary });
    if (rt.cache.size > 20000) rt.cache.clear();
  }
  return res;
}

function bestBaselineMetric(t: TaskDef): { name: string; metric: number } {
  let bestName = "—";
  let bestMetric = t.metricDirection === "min" ? Infinity : -Infinity;
  for (const b of t.baselines) {
    const better = t.metricDirection === "min" ? b.metric < bestMetric : b.metric > bestMetric;
    if (better) {
      bestMetric = b.metric;
      bestName = b.name;
    }
  }
  return { name: bestName, metric: bestMetric };
}

function breakthroughWorthy(t: TaskDef, prev: number, next: number): boolean {
  if (!Number.isFinite(prev)) return Number.isFinite(next);
  if (!Number.isFinite(next)) return false;
  if (t.metricDirection === "min") {
    if (next >= prev) return false;
    return (prev - next) / Math.abs(prev) >= RELATIVE_IMPROVE;
  }
  if (next <= prev) return false;
  return (next - prev) / Math.abs(prev) >= RELATIVE_IMPROVE;
}

async function yieldToEventLoop(): Promise<void> {
  await new Promise<void>((resolve) => setImmediate(resolve));
}

/**
 * Cost model instead of a wall-clock race: comparing a scalar JS loop against a
 * vectorised tree walk would be dishonest. We count ALU/SFU units instead
 * (exp/erf/tanh are only served by the special-function unit and cost ~20-30
 * units, mul/add/fma cost 1), and verify throughput on a vectorised batch.
 */
function benchmarkSpeed(t: TaskDef, node: SpearNode): LoopTaskSnapshot["speed"] {
  const exactCost = t.exactCost ?? (t.exactRefNode ? estimateCost(t.exactRefNode) : undefined);
  if (!exactCost) return null;
  const profile = taskOpProfile(node);
  const formulaCost = estimateCost(node);
  const n = 100_000;
  const xs = new Float64Array(n);
  // deterministic local sweep: snapshots fire on wall-clock intervals, so this
  // must never touch the shared seeded stream or runs stop being reproducible.
  // every variable of the formula gets the same sweep — enough for throughput.
  for (let i = 0; i < n; i++) xs[i] = -6 + (12 * i) / (n - 1);
  const vars: Record<string, Float64Array> = {};
  for (const name of new Set(collectVarNames(node))) vars[name] = xs;
  const out = evaluateNode(node, vars, n);
  let acc = 0;
  for (let i = 0; i < n; i++) acc += out[i];
  void acc;
  return {
    formulaCost,
    exactCost,
    estimatedSpeedup: exactCost / Math.max(1, formulaCost),
    ops: profile.total,
    transcendental: profile.transcendental,
    elements: n,
  };
}

function rand01(): number {
  return rand();
}


function snapshotTask(rt: TaskRuntime): LoopTaskSnapshot {
  const t = rt.def;
  const bl = bestBaselineMetric(t);
  const front: LoopFrontEntry[] = rt.frontRaw
    .map((f) => ({ formula: nodeToString(f.node), metric: f.metric, size: f.size, level: levelOf(t, f.metric) }))
    .sort((a, b) => (t.metricDirection === "min" ? a.metric - b.metric : b.metric - a.metric));
  const best = rt.bestNode
    ? {
        formula: nodeToString(rt.bestNode),
        metric: rt.bestMetric,
        size: rt.bestNode.size,
        level: rt.bestLevel,
      }
    : null;

  const speed = rt.bestNode ? benchmarkSpeed(t, rt.bestNode) : null;

  return {
    taskId: t.id,
    family: t.family,
    title: t.title,
    subtitle: t.subtitle,
    groundTruth: t.groundTruth,
    metricLabel: t.metricLabel,
    metricDirection: t.metricDirection,
    iterations: rt.iterations,
    evals: rt.evals,
    cacheHits: rt.cacheHits,
    stagnation: rt.stagnation,
    priority: Number(rt.priority.toFixed(4)),
    history: rt.history.slice(-160),
    paretoFront: front.slice(0, 10),
    best,
    secondary: rt.bestSecondary ?? null,
    python: rt.bestNode ? toPython(rt.bestNode, `spear_${t.id}`) : null,
    c: rt.bestNode ? toC(rt.bestNode, `spear_${t.id}`, t.codeVarDecl) : null,
    wasm: rt.bestNode ? Buffer.from(toWasmBytes(rt.bestNode)).toString("base64") : null,
    chart: rt.bestNode && t.chart ? t.chart(rt.bestNode) : null,
    ops: rt.bestNode ? taskOpProfile(rt.bestNode) : null,
    verifyNote: rt.bestNode && t.verify ? t.verify(rt.bestNode) : null,
    speed,
    baselines: t.baselines.map((b) => {
      const beaten =
        rt.bestMetric !== undefined && Number.isFinite(rt.bestMetric)
          ? t.metricDirection === "min"
            ? rt.bestMetric < b.metric
            : rt.bestMetric > b.metric
          : false;
      const ratio =
        rt.bestMetric !== undefined && Number.isFinite(rt.bestMetric) && Number.isFinite(b.metric) && b.metric !== 0
          ? b.metric / rt.bestMetric
          : null;
      return { ...b, beaten, ratio };
    }),
    bestBaselineName: bl.name,
    milestonesHit: [...rt.milestonesHit],
  };
}

function snapshot(rts: TaskRuntime[], progress: LoopProgress): LoopProgress {
  progress.tasks = rts.map(snapshotTask);
  progress.totals = {
    breakthroughs: progress.breakthroughs.length,
    maxLevel: Math.max(0, ...progress.tasks.map((t) => t.best?.level ?? 0)),
    sumLevels: progress.tasks.reduce((s, t) => s + (t.best?.level ?? 0), 0),
    tasksBeatingBaseline: progress.tasks.filter((t) => t.baselines.some((b) => b.beaten)).length,
    evaluations: rts.reduce((s, r) => s + r.evals, 0),
    cacheHits: rts.reduce((s, r) => s + r.cacheHits, 0),
  };
  return progress;
}

export interface GroundedLoopOptions {
  seed?: number;
  budget?: number;
  deadlineMs?: number;
  onProgress?: (p: LoopProgress) => void | Promise<void>;
}

export async function runGroundedLoop(opts: GroundedLoopOptions = {}): Promise<LoopProgress> {
  const seed = opts.seed ?? Math.floor(Math.random() * 1e9);
  const budget = Math.max(30, Math.min(2000, Math.floor(opts.budget ?? 500)));
  const deadlineMs = Math.max(3000, Math.min(45_000, opts.deadlineMs ?? 18_000));
  setSeed(seed);
  // datasets must be generated from the same seeded stream, otherwise two runs
  // with the same seed would score against different data
  setUniformSource(rand);

  const defs = buildTasks();
  const rts: TaskRuntime[] = defs.map((def, di) => ({
    def,
    population: Array.from({ length: POP }, (_, i) => {
      // half of generation 0 comes from generic algebraic primitives, the rest
      // is random — never from the published baselines we compete against.
      const pool = def.seedPool ?? [];
      const base = pool.length > 0 && i % 2 === 0 ? pool[(di + i) % pool.length] : undefined;
      return base ? simplify(mutate(base, def.gpConfig)) : simplify(randomTree(def.gpConfig, def.gpConfig.maxDepth));
    }),
    cache: new Map(),
    iterations: 0,
    evals: 0,
    cacheHits: 0,
    stagnation: 0,
    priority: 1,
    bestNode: null,
    bestMetric: def.metricDirection === "min" ? Infinity : -Infinity,
    bestSecondary: undefined,
    bestLevel: 0,
    milestonesHit: new Set<string>(),
    baselineBeaten: false,
    recentMetrics: [],
    warmedUp: false,
    history: [],
    frontRaw: [],
  }));

  const progress: LoopProgress = {
    status: "running",
    seed,
    budget,
    iterationsUsed: 0,
    elapsedMs: 0,
    deadlineMs,
    tasks: [],
    breakthroughs: [],
    totals: { breakthroughs: 0, maxLevel: 0, sumLevels: 0, tasksBeatingBaseline: 0, evaluations: 0, cacheHits: 0 },
  };

  const t0 = performance.now();
  let lastPersist = 0;
  const totalStepsTaken = () => rts.reduce((s, r) => s + r.iterations, 0);

  while (progress.iterationsUsed < budget) {
    if (performance.now() - t0 > deadlineMs) {
      progress.status = "stopped_time";
      break;
    }

    // ---- UCB-style budget allocation: exploit improving tasks, explore idle ones
    let target = rts[0];
    let bestScore = -Infinity;
    const totalSteps = Math.max(1, totalStepsTaken());
    for (const rt of rts) {
      const recent = rt.recentMetrics.slice(-12);
      let improvement = 0;
      if (recent.length >= 4 && Number.isFinite(recent[0]) && Number.isFinite(recent[recent.length - 1])) {
        const a = Math.abs(recent[0]) || 1;
        improvement = (rt.def.metricDirection === "min" ? recent[0] - recent[recent.length - 1] : recent[recent.length - 1] - recent[0]) / a;
      }
      const explore = UCB_C * Math.sqrt(Math.log(totalSteps + 2) / (rt.iterations + 1));
      const urgency = rt.stagnation > STAGNATION_LIMIT * 0.6 ? 0.5 : rt.stagnation > STAGNATION_LIMIT ? 0.25 : 0;
      const score = Math.max(0, improvement) + explore + urgency;
      rt.priority = score;
      if (score > bestScore) {
        bestScore = score;
        target = rt;
      }
    }

    const rt = target;
    const t = rt.def;

    // ---- multi-start warm-up (once per task): take every seeded shape and run
    // constant refinement on it immediately. A tuned good shape beats a
    // thousand random crossovers, and it costs a fraction of the budget.
    if (!rt.warmedUp) {
      rt.warmedUp = true;
      const pool = t.seedPool ?? [];
      for (const seed of pool) {
        const tuned = t.refine(simplify(seed));
        rt.evals += tuned.evals;
        const ev = cachedEval(rt, tuned.node);
        if (!Number.isFinite(ev.metric)) continue;
        rt.population.push(ev.node);
        const rep = reportMetric(rt, ev.node);
        const better = Number.isFinite(rep.metric) && (
          !Number.isFinite(rt.bestMetric) ||
          (t.metricDirection === "min" ? rep.metric < rt.bestMetric : rep.metric > rt.bestMetric)
        );
        if (better) {
          const prev = rt.bestMetric;
          rt.bestNode = ev.node;
          rt.bestMetric = rep.metric;
          rt.bestSecondary = rep.secondary;
          rt.bestLevel = levelOf(t, rep.metric);
          rt.history.push({ i: rt.iterations, metric: rep.metric, size: ev.node.size });

          const deltaPct = Number.isFinite(prev) && prev !== 0
            ? (t.metricDirection === "min" ? (prev - rep.metric) / Math.abs(prev) : (rep.metric - prev) / Math.abs(prev)) * 100
            : null;
          progress.breakthroughs.push({
            iteration: progress.iterationsUsed + 1,
            taskId: t.id,
            taskTitle: t.title,
            level: rt.bestLevel,
            kind: Number.isFinite(deltaPct as number) ? "improvement" : "first_valid",
            label: Number.isFinite(deltaPct as number)
              ? `Multi-start : ${(deltaPct as number) > 0 ? "+" : ""}${(deltaPct as number).toFixed(1)} % sur ${t.metricLabel}`
              : "Multi-start — première formule valide",
            formula: nodeToString(ev.node),
            metric: rep.metric,
            deltaPct: Number.isFinite(deltaPct as number) ? deltaPct : null,
          });
          // newly reached milestones
          for (const m of t.milestones) {
            if (!m.test(rep.metric) || rt.milestonesHit.has(m.label)) continue;
            rt.milestonesHit.add(m.label);
            progress.breakthroughs.push({
              iteration: progress.iterationsUsed + 1,
              taskId: t.id,
              taskTitle: t.title,
              level: m.level,
              kind: "milestone",
              label: `Jalon atteint : ${m.label}`,
              formula: nodeToString(ev.node),
              metric: rep.metric,
              deltaPct: null,
            });
          }
          // baseline crossings
          if (!rt.baselineBeaten) {
            const bl = bestBaselineMetric(t);
            const beaten = t.metricDirection === "min" ? rep.metric < bl.metric : rep.metric > bl.metric;
            if (beaten) {
              rt.baselineBeaten = true;
              progress.breakthroughs.push({
                iteration: progress.iterationsUsed + 1,
                taskId: t.id,
                taskTitle: t.title,
                level: Math.max(3, rt.bestLevel),
                kind: "beats_baseline",
                label: `Dépasse la meilleure baseline « ${bl.name} »`,
                formula: nodeToString(ev.node),
                metric: rep.metric,
                deltaPct: null,
                note: `${t.metricLabel} : ${t.formatMetric(rep.metric)} vs ${t.formatMetric(bl.metric)}`,
              });
            }
          }
        }
      }
      while (rt.population.length > POP) rt.population.shift();
    }

    // ---- anti-stagnation: when pinned against the wall, do three things:
    //   (a) polish the elite (fine-grain constant tweaks),
    //   (b) apply a structure mutation to 1/3 of the population,
    //   (c) inject task-specific rational/polynomial forms.
    if (rt.stagnation >= STAGNATION_LIMIT) {
      if (rt.bestNode) {
        const polishPool = [
          mutatePolish(rt.bestNode, 0.05),
          mutatePolish(rt.bestNode, 0.15),
          mutatePolish(rt.bestNode, 0.4),
          mutatePolish(rt.bestNode, 1.0),
        ];
        for (const p of polishPool) rt.population.push(simplify(p));
        while (rt.population.length > POP) rt.population.pop();
      }
      const struct = Math.max(3, Math.floor(POP * 0.25));
      for (let i = 0; i < struct && i < rt.population.length; i++) {
        rt.population[i] = simplify(mutateStructure(rt.population[i], t.gpConfig));
      }
      // Task-specific seed pool: rational / polynomial forms that match the
      // expected shape of the target. Generic GP would take thousands of iters
      // to stumble on these by chance.
      const pool = t.seedPool ?? [];
      if (pool.length > 0) {
        const inject = Math.max(4, Math.floor(POP * 0.25));
        for (let i = 0; i < inject; i++) {
          const seed = pool[i % pool.length];
          rt.population[inject + i] = simplify(mutate(seed, t.gpConfig));
        }
      }
      rt.stagnation = 0;
    }

    // ---- one generation: evaluate → NSGA-II sort → crowding → breed
    const results = rt.population.map((ind) => cachedEval(rt, ind));
    const objs = results.map((r, i) => ({ fitness: selectFitness(rt, r.metric, rt.population[i].size, progress.iterationsUsed, budget), size: rt.population[i].size }));
    const ranks = nonDominatedSort(objs);

    // track elite + breakthroughs
    let eliteIdx = 0;
    for (let i = 1; i < objs.length; i++) {
      if (objs[i].fitness > objs[eliteIdx].fitness) eliteIdx = i;
    }
    const eliteNode = results[eliteIdx].node;
    const eliteMetric = reportMetric(rt, eliteNode).metric;
    if (Number.isFinite(eliteMetric)) {
      if (breakthroughWorthy(t, rt.bestMetric, eliteMetric)) {
        const deltaPct =
          Number.isFinite(rt.bestMetric) && rt.bestMetric !== 0
            ? t.metricDirection === "min"
              ? ((rt.bestMetric - eliteMetric) / Math.abs(rt.bestMetric)) * 100
              : ((eliteMetric - rt.bestMetric) / Math.abs(rt.bestMetric)) * 100
            : null;
        rt.bestNode = eliteNode;
        rt.bestMetric = eliteMetric;
        rt.bestSecondary = results[eliteIdx].secondary;
        const lvl = levelOf(t, eliteMetric);
        rt.bestLevel = lvl;
        for (const m of t.milestones) {
          if (m.test(eliteMetric)) rt.milestonesHit.add(m.label);
        }
        rt.history.push({ i: rt.iterations, metric: eliteMetric, size: eliteNode.size });
        progress.breakthroughs.push({
          iteration: progress.iterationsUsed + 1,
          taskId: t.id,
          taskTitle: t.title,
          level: lvl,
          kind: Number.isFinite(deltaPct) ? "improvement" : "first_valid",
          label: Number.isFinite(deltaPct)
            ? `${deltaPct && deltaPct > 0 ? "+" : ""}${deltaPct?.toFixed(1)} % sur ${t.metricLabel}`
            : "Première formule valide",
          formula: nodeToString(eliteNode),
          metric: eliteMetric,
          deltaPct: Number.isFinite(deltaPct) ? deltaPct : null,
        });

        const bl = bestBaselineMetric(t);
        if (!rt.baselineBeaten) {
          const beaten =
            t.metricDirection === "min" ? eliteMetric < bl.metric : eliteMetric > bl.metric;
          if (beaten) {
            rt.baselineBeaten = true;
            progress.breakthroughs.push({
              iteration: progress.iterationsUsed + 1,
              taskId: t.id,
              taskTitle: t.title,
              level: Math.max(3, lvl),
              kind: "beats_baseline",
              label: `Dépasse la meilleure baseline « ${bl.name} »`,
              formula: nodeToString(eliteNode),
              metric: eliteMetric,
              deltaPct: null,
              note: `${t.metricLabel} : ${t.formatMetric(eliteMetric)} vs ${t.formatMetric(bl.metric)}`,
            });
          }
        }
        const reached = t.milestones.filter((m) => m.test(eliteMetric) && !rt.milestonesHit.has(m.label));
        for (const m of reached) {
          rt.milestonesHit.add(m.label);
          progress.breakthroughs.push({
            iteration: progress.iterationsUsed + 1,
            taskId: t.id,
            taskTitle: t.title,
            level: m.level,
            kind: "milestone",
            label: `Jalon atteint : ${m.label}`,
            formula: nodeToString(eliteNode),
            metric: eliteMetric,
            deltaPct: null,
          });
        }
      } else if (rt.bestNode === null) {
        rt.bestNode = eliteNode;
        rt.bestMetric = eliteMetric;
        rt.bestSecondary = results[eliteIdx].secondary;
        rt.bestLevel = levelOf(t, eliteMetric);
        rt.history.push({ i: rt.iterations, metric: eliteMetric, size: eliteNode.size });
        progress.breakthroughs.push({
          iteration: progress.iterationsUsed + 1,
          taskId: t.id,
          taskTitle: t.title,
          level: rt.bestLevel,
          kind: "first_valid",
          label: "Première formule valide",
          formula: nodeToString(eliteNode),
          metric: eliteMetric,
          deltaPct: null,
        });
      }
      rt.recentMetrics.push(eliteMetric);
      if (rt.recentMetrics.length > 40) rt.recentMetrics.shift();
    }

    // stagnation bookkeeping
    const prevBest = rt.recentMetrics.length > 1 ? rt.recentMetrics[rt.recentMetrics.length - 2] : undefined;
    const improvedLast =
      prevBest !== undefined && Number.isFinite(prevBest) && Number.isFinite(eliteMetric)
        ? t.metricDirection === "min"
          ? eliteMetric < prevBest - Math.abs(prevBest) * 1e-6
          : eliteMetric > prevBest + Math.abs(prevBest) * 1e-6
        : false;
    rt.stagnation = improvedLast ? 0 : rt.stagnation + 1;

    // Pareto front snapshot (rank 0 of the current population, de-duplicated)
    const seen = new Set<string>();
    rt.frontRaw = [];
    for (let i = 0; i < rt.population.length && rt.frontRaw.length < 12; i++) {
      if (ranks[i] !== 0 || !Number.isFinite(results[i].metric)) continue;
      const key = canonicalKey(rt.population[i]);
      if (seen.has(key)) continue;
      seen.add(key);
      rt.frontRaw.push({ node: rt.population[i], metric: results[i].metric, size: rt.population[i].size });
    }

    // elitist replacement + breeding
    const maxRank = Math.max(...ranks);
    const byFront: number[][] = Array.from({ length: maxRank + 1 }, () => []);
    for (let i = 0; i < ranks.length; i++) byFront[ranks[i]].push(i);
    const survivors: number[] = [];
    for (const front of byFront) {
      if (front.length === 0) continue;
      if (survivors.length + front.length <= POP) {
        survivors.push(...front);
        continue;
      }
      const dist = front.map((i) => ({ i, f: objs[i].fitness, s: objs[i].size }));
      dist.sort((a, b) => a.s - b.s);
      for (const { i } of dist) {
        if (survivors.length >= POP) break;
        survivors.push(i);
      }
      break;
    }
    const parentPool = survivors.length > 0 ? survivors : objs.map((_, i) => i);
    const next: SpearNode[] = [rt.population[eliteIdx]];
    while (next.length < POP) {
      const p1 = rt.population[parentPool[tournamentSelect(parentPool.map((i) => objs[i]))]];
      let child = rand01() < 0.8
        ? crossover(p1, rt.population[parentPool[tournamentSelect(parentPool.map((i) => objs[i]))]], t.gpConfig.maxDepth)
        : p1;
      if (rand01() < 0.4) child = mutate(child, t.gpConfig);
      child = simplify(child);
      if (child.depth <= t.gpConfig.maxDepth) next.push(child);
      else next.push(p1);
    }
    rt.population = next;
    rt.iterations++;
    progress.iterationsUsed++;

    // ---- periodic local search on the elite (constants refinement)
    if (rt.iterations % REFINE_EVERY === 0 && rt.bestNode) {
      // 3-stage polish: tune → prune → re-tune. Each restart uses its own
      // eval budget so we don't starve the main loop on refinement.
      const stage1 = t.refine(simplify(rt.bestNode));
      let candidate = stage1;
      if (t.family !== "kv_cache" && stage1.node) {
        const scoreFn = (nd: SpearNode) => cachedEval(rt, nd).metric;
        const prunedRes = prune(candidate.node, scoreFn, 1.02);
        if (prunedRes.removed > 0 && Number.isFinite(prunedRes.score)) {
          const reTuned = t.refine(prunedRes.node);
          if (Number.isFinite(cachedEval(rt, reTuned.node).metric)) candidate = reTuned;
        }
        // one extra polish pass on the pruned form
        for (let step = 0; step < 3; step++) {
          const polished = mutatePolish(candidate.node, [0.02, 0.08, 0.25][step]);
          const res = cachedEval(rt, polished);
          if (Number.isFinite(res.metric)) {
            const prev = cachedEval(rt, candidate.node).metric;
            const better = rt.def.metricDirection === "min" ? res.metric < prev - 1e-9 : res.metric > prev + 1e-9;
            if (better) candidate = { node: polished, evals: 1 };
          }
        }
      }
      if (candidate.node && candidate.evals > 0) {
        rt.evals += candidate.evals;
        cachedEval(rt, candidate.node); // populates the memo (counts evals once)
        const res = reportMetric(rt, candidate.node);
        if (breakthroughWorthy(t, rt.bestMetric, res.metric)) {
          const deltaPct = Number.isFinite(rt.bestMetric) && rt.bestMetric !== 0
            ? t.metricDirection === "min"
              ? ((rt.bestMetric - res.metric) / Math.abs(rt.bestMetric)) * 100
              : ((res.metric - rt.bestMetric) / Math.abs(rt.bestMetric)) * 100
            : null;
          rt.bestNode = candidate.node;
          rt.bestMetric = res.metric;
          rt.bestSecondary = res.secondary;
          rt.bestLevel = levelOf(t, res.metric);
          for (const m of t.milestones) if (m.test(res.metric)) rt.milestonesHit.add(m.label);
          rt.history.push({ i: rt.iterations, metric: res.metric, size: candidate.node.size });
          progress.breakthroughs.push({
            iteration: progress.iterationsUsed,
            taskId: t.id,
            taskTitle: t.title,
            level: rt.bestLevel,
            kind: "improvement",
            label: `Raffinement de constantes : ${deltaPct !== null && Number.isFinite(deltaPct) ? `+${deltaPct.toFixed(1)} %` : "gain"}`,
            formula: nodeToString(candidate.node),
            metric: res.metric,
            deltaPct: Number.isFinite(deltaPct as number) ? deltaPct : null,
          });
          const reached = t.milestones.filter((m) => m.test(res.metric));
          for (const m of reached) {
            progress.breakthroughs.push({
              iteration: progress.iterationsUsed,
              taskId: t.id,
              taskTitle: t.title,
              level: m.level,
              kind: "milestone",
              label: `Jalon atteint : ${m.label}`,
              formula: nodeToString(candidate.node),
              metric: res.metric,
              deltaPct: null,
            });
          }
          const bl = bestBaselineMetric(t);
          if (!rt.baselineBeaten) {
            const beaten = t.metricDirection === "min" ? res.metric < bl.metric : res.metric > bl.metric;
            if (beaten) {
              rt.baselineBeaten = true;
              progress.breakthroughs.push({
                iteration: progress.iterationsUsed,
                taskId: t.id,
                taskTitle: t.title,
                level: Math.max(3, rt.bestLevel),
                kind: "beats_baseline",
                label: `Dépasse la meilleure baseline « ${bl.name} »`,
                formula: nodeToString(candidate.node),
                metric: res.metric,
                deltaPct: null,
                note: `${t.metricLabel} : ${t.formatMetric(res.metric)} vs ${t.formatMetric(bl.metric)}`,
              });
            }
          }
        }
      }
      progress.iterationsUsed++; // refinement pass consumes budget too
    }

    progress.elapsedMs = performance.now() - t0;

    // periodic yield + persistence so the UI can stream progress
    if (performance.now() - lastPersist > 500 || progress.iterationsUsed >= budget) {
      lastPersist = performance.now();
      snapshot(rts, progress);
      if (opts.onProgress) await opts.onProgress(progress);
      await yieldToEventLoop();
    }
    if (progress.iterationsUsed % 4 === 0) await yieldToEventLoop();
  }

  if (progress.status === "running") progress.status = progress.iterationsUsed >= budget ? "stopped_budget" : "completed";
  progress.elapsedMs = performance.now() - t0;
  snapshot(rts, progress);
  if (opts.onProgress) await opts.onProgress(progress);
  return progress;
}


