import { performance } from "node:perf_hooks";
import {
  evaluateNode,
  evolve,
  nodeToString,
  type EvolveConfig,
  type FitnessFn,
  type GenerationRecord,
  type SpearNode,
} from "./engine";
import {
  clamp,
  erf,
  gaussianRandom,
  gelu,
  linfError,
  linspace,
  mapArray,
  mse,
  r2Score,
  sigmoid,
  silu,
} from "./math-utils";

export interface ChartPoint {
  x: number;
  target?: number;
  predicted?: number;
}

export interface SpearRunResult {
  formulaText: string;
  formulaTree: SpearNode;
  fitness: number;
  mse: number | null;
  linfError: number | null;
  treeSize: number;
  durationMs: number;
  history: GenerationRecord[];
  metrics: Record<string, unknown>;
  chartData: ChartPoint[];
}

// ---------------------------------------------------------------------------
// Preset 1 — Fast activation function synthesis (GELU / SiLU / tanh / sigmoid)
// Goal: discover a transcendental-free algebraic approximation used in LLM
// feed-forward layers (SwiGLU / GEGLU) and benchmark the speedup.
// ---------------------------------------------------------------------------

export type ActivationTarget = "silu" | "gelu" | "tanh" | "sigmoid" | "gaussian_cdf" | "rl_distillation";

const TARGET_FNS: Record<ActivationTarget, (x: number) => number> = {
  silu,
  gelu,
  tanh: Math.tanh,
  sigmoid,
  gaussian_cdf: (x) => 0.5 * (1 + erf(x / Math.SQRT2)),
  rl_distillation: (x) => Math.tanh(2 * x),
};

export interface ActivationConfig {
  target: ActivationTarget;
  populationSize: number;
  generations: number;
  maxDepth: number;
  domainMin: number;
  domainMax: number;
  points: number;
}

export function runActivationPreset(cfg: ActivationConfig): SpearRunResult {
  const n = clamp(cfg.points, 64, 800);
  const domainMin = clamp(cfg.domainMin, -20, 0);
  const domainMax = clamp(cfg.domainMax, 0, 20);
  const xArr = linspace(domainMin, domainMax, n);
  const fn = TARGET_FNS[cfg.target] ?? silu;
  const yTarget = mapArray(xArr, fn);
  const vars = { x: xArr };

  const gpCfg: EvolveConfig = {
    variables: ["x"],
    constRange: [-3, 3],
    ops: ["add", "sub", "mul", "pdiv", "relu", "abs", "neg", "sq", "max", "min"],
    maxDepth: clamp(cfg.maxDepth, 2, 6),
    populationSize: clamp(cfg.populationSize, 20, 400),
    generations: clamp(cfg.generations, 5, 100),
  };

  const fitnessFn: FitnessFn = (node) => {
    let pred: Float64Array;
    try {
      pred = evaluateNode(node, vars, n);
    } catch {
      return { fitness: -1e9, size: node.size };
    }
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(pred[i])) return { fitness: -1e9, size: node.size };
    }
    const m = mse(pred, yTarget);
    const li = linfError(pred, yTarget);
    if (!Number.isFinite(m)) return { fitness: -1e9, size: node.size };
    return { fitness: -(m + 0.1 * li), size: node.size, extra: li };
  };

  const result = evolve(gpCfg, fitnessFn);
  const predFinal = evaluateNode(result.best, vars, n);
  const finalMse = mse(predFinal, yTarget);
  const finalLinf = linfError(predFinal, yTarget);

  // Speed benchmark: exact transcendental function vs. discovered algebraic tree
  const benchN = 400_000;
  const benchX = new Float64Array(benchN);
  for (let i = 0; i < benchN; i++) benchX[i] = domainMin + Math.random() * (domainMax - domainMin);
  const benchVars = { x: benchX };

  const t0 = performance.now();
  let sinkExact = 0;
  for (let i = 0; i < benchN; i++) sinkExact += fn(benchX[i]);
  const exactTimeMs = performance.now() - t0;

  const t1 = performance.now();
  const spearBench = evaluateNode(result.best, benchVars, benchN);
  const spearTimeMs = performance.now() - t1;
  let sinkSpear = 0;
  for (let i = 0; i < benchN; i++) sinkSpear += spearBench[i];

  const chartN = 60;
  const chartX = linspace(domainMin, domainMax, chartN);
  const chartTarget = mapArray(chartX, fn);
  const chartPred = evaluateNode(result.best, { x: chartX }, chartN);
  const chartData: ChartPoint[] = Array.from({ length: chartN }, (_, i) => ({
    x: chartX[i],
    target: chartTarget[i],
    predicted: chartPred[i],
  }));

  return {
    formulaText: nodeToString(result.best),
    formulaTree: result.best,
    fitness: result.bestFitness,
    mse: finalMse,
    linfError: finalLinf,
    treeSize: result.best.size,
    durationMs: result.durationMs,
    history: result.history,
    metrics: {
      target: cfg.target,
      exactTimeMs,
      spearTimeMs,
      speedup: exactTimeMs / Math.max(spearTimeMs, 1e-6),
      benchElements: benchN,
      // prevent dead-code elimination from being a concern for readers of the numbers
      checksum: Number((sinkExact - sinkSpear).toFixed(6)),
    },
    chartData,
  };
}

// ---------------------------------------------------------------------------
// Preset 2 — KV-cache token eviction rule discovery (long-context inference)
// Goal: evolve a cheap scoring formula Score(A, P, S, R) that decides which
// tokens to keep in the KV-cache while maximising retained attention mass.
// ---------------------------------------------------------------------------

export interface KvCacheConfig {
  populationSize: number;
  generations: number;
  maxDepth: number;
  seqLen: number;
  keepBudget: number;
  numSamples: number;
}

type KvSample = Record<"A" | "P" | "S" | "R", Float64Array> & Record<string, Float64Array>;

function generateKvDataset(seqLen: number, numSamples: number): KvSample[] {
  const samples: KvSample[] = [];
  const sinkCount = Math.min(4, seqLen);
  const recencyCount = Math.min(64, Math.max(0, seqLen - sinkCount));

  for (let s = 0; s < numSamples; s++) {
    const logits = new Float64Array(seqLen);
    for (let i = 0; i < seqLen; i++) logits[i] = gaussianRandom() * 0.5;
    for (let i = 0; i < sinkCount; i++) logits[i] += 3 + Math.random() * 2;
    for (let i = 0; i < recencyCount; i++) {
      const idx = seqLen - recencyCount + i;
      logits[idx] += 1 + (recencyCount > 1 ? (i / (recencyCount - 1)) * 3 : 3);
    }
    const hhCount = Math.min(15, Math.max(0, seqLen - sinkCount - recencyCount));
    const used = new Set<number>();
    for (let k = 0; k < hhCount; k++) {
      let idx = 0;
      let guard = 0;
      do {
        idx = sinkCount + Math.floor(Math.random() * Math.max(1, seqLen - sinkCount - recencyCount));
        guard++;
      } while (used.has(idx) && guard < 50);
      used.add(idx);
      logits[idx] += 2.5 + Math.random() * 2;
    }

    let maxL = -Infinity;
    for (let i = 0; i < seqLen; i++) if (logits[i] > maxL) maxL = logits[i];
    const expL = new Float64Array(seqLen);
    let sumExp = 0;
    for (let i = 0; i < seqLen; i++) {
      expL[i] = Math.exp(logits[i] - maxL);
      sumExp += expL[i];
    }
    const A = new Float64Array(seqLen);
    for (let i = 0; i < seqLen; i++) A[i] = expL[i] / sumExp;

    const P = new Float64Array(seqLen);
    for (let i = 0; i < seqLen; i++) P[i] = seqLen > 1 ? i / (seqLen - 1) : 0;

    const S = new Float64Array(seqLen);
    for (let i = 0; i < sinkCount; i++) S[i] = 1;

    const R = new Float64Array(seqLen);
    for (let i = seqLen - recencyCount; i < seqLen; i++) R[i] = 1;

    samples.push({ A, P, S, R });
  }
  return samples;
}

// Selects the K largest values without a full O(n log n) sort (min-heap based).
function topKSum(score: Float64Array, weight: Float64Array, k: number): number {
  const n = score.length;
  if (k >= n) {
    let total = 0;
    for (let i = 0; i < n; i++) total += weight[i];
    return total;
  }
  const heapScore = new Float64Array(k);
  const heapWeight = new Float64Array(k);
  let size = 0;

  const siftUp = (pos: number) => {
    while (pos > 0) {
      const parent = (pos - 1) >> 1;
      if (heapScore[parent] <= heapScore[pos]) break;
      [heapScore[parent], heapScore[pos]] = [heapScore[pos], heapScore[parent]];
      [heapWeight[parent], heapWeight[pos]] = [heapWeight[pos], heapWeight[parent]];
      pos = parent;
    }
  };
  const siftDown = (pos: number) => {
    for (;;) {
      const l = pos * 2 + 1;
      const r = pos * 2 + 2;
      let smallest = pos;
      if (l < size && heapScore[l] < heapScore[smallest]) smallest = l;
      if (r < size && heapScore[r] < heapScore[smallest]) smallest = r;
      if (smallest === pos) break;
      [heapScore[smallest], heapScore[pos]] = [heapScore[pos], heapScore[smallest]];
      [heapWeight[smallest], heapWeight[pos]] = [heapWeight[pos], heapWeight[smallest]];
      pos = smallest;
    }
  };

  for (let i = 0; i < n; i++) {
    if (size < k) {
      heapScore[size] = score[i];
      heapWeight[size] = weight[i];
      size++;
      siftUp(size - 1);
    } else if (score[i] > heapScore[0]) {
      heapScore[0] = score[i];
      heapWeight[0] = weight[i];
      siftDown(0);
    }
  }
  let total = 0;
  for (let i = 0; i < size; i++) total += heapWeight[i];
  return total;
}

export function runKvCachePreset(cfg: KvCacheConfig): SpearRunResult {
  const seqLen = clamp(cfg.seqLen, 128, 768);
  const keepBudget = clamp(cfg.keepBudget, 8, Math.floor(seqLen / 2));
  const numSamples = clamp(cfg.numSamples, 4, 12);
  const dataset = generateKvDataset(seqLen, numSamples);

  const gpCfg: EvolveConfig = {
    variables: ["A", "P", "S", "R"],
    constRange: [0.1, 2.0],
    ops: ["add", "sub", "mul", "pdiv", "max", "min"],
    maxDepth: clamp(cfg.maxDepth, 2, 5),
    populationSize: clamp(cfg.populationSize, 20, 90),
    generations: clamp(cfg.generations, 5, 25),
  };

  const evalMass = (node: SpearNode): number[] | null => {
    const masses: number[] = [];
    for (const sample of dataset) {
      let score: Float64Array;
      try {
        score = evaluateNode(node, sample, seqLen);
      } catch {
        return null;
      }
      for (let i = 0; i < seqLen; i++) {
        if (!Number.isFinite(score[i])) return null;
      }
      masses.push(topKSum(score, sample.A, keepBudget));
    }
    return masses;
  };

  const fitnessFn: FitnessFn = (node) => {
    const masses = evalMass(node);
    if (!masses) return { fitness: -1e9, size: node.size };
    const meanMass = masses.reduce((a, b) => a + b, 0) / masses.length;
    const variance = masses.reduce((a, b) => a + (b - meanMass) ** 2, 0) / masses.length;
    const std = Math.sqrt(variance);
    return { fitness: meanMass - 0.1 * std, size: node.size, extra: meanMass };
  };

  const result = evolve(gpCfg, fitnessFn);
  const finalMasses = evalMass(result.best) ?? [0];
  const meanMass = finalMasses.reduce((a, b) => a + b, 0) / finalMasses.length;

  return {
    formulaText: nodeToString(result.best),
    formulaTree: result.best,
    fitness: result.bestFitness,
    mse: null,
    linfError: null,
    treeSize: result.best.size,
    durationMs: result.durationMs,
    history: result.history,
    metrics: {
      attentionMassCapturedPct: meanMass * 100,
      seqLen,
      keepBudget,
      numSamples,
      compressionRatio: seqLen / keepBudget,
      memoryReductionPct: (1 - keepBudget / seqLen) * 100,
    },
    chartData: [],
  };
}

// ---------------------------------------------------------------------------
// Preset 3 — Generic symbolic regression on a user-supplied CSV dataset.
// ---------------------------------------------------------------------------

export interface ParsedDataset {
  variables: string[];
  vars: Record<string, Float64Array>;
  target: Float64Array;
  n: number;
}

export function parseCsvDataset(text: string): ParsedDataset {
  const lines = text
    .trim()
    .split(/\r?\n/)
    .filter((l) => l.trim().length > 0);
  if (lines.length < 3) {
    throw new Error("Le jeu de données doit contenir un en-tête et au moins 2 lignes de données.");
  }
  const header = lines[0].split(",").map((h) => h.trim());
  const yIdx = header.findIndex((h) => h.toLowerCase() === "y");
  if (yIdx === -1) {
    throw new Error("L'en-tête doit contenir une colonne cible nommée 'y'.");
  }
  const varNames = header.filter((_, i) => i !== yIdx);
  if (varNames.length === 0) {
    throw new Error("Le jeu de données doit contenir au moins une variable en plus de 'y'.");
  }
  const n = lines.length - 1;
  if (n > 4000) {
    throw new Error("Jeu de données trop volumineux (4000 lignes maximum).");
  }
  const vars: Record<string, Float64Array> = {};
  for (const v of varNames) vars[v] = new Float64Array(n);
  const target = new Float64Array(n);

  for (let r = 0; r < n; r++) {
    const row = lines[r + 1].split(",").map((c) => c.trim());
    if (row.length !== header.length) {
      throw new Error(`La ligne ${r + 2} contient ${row.length} valeurs, ${header.length} attendues.`);
    }
    let vi = 0;
    for (let c = 0; c < row.length; c++) {
      const val = Number(row[c]);
      if (!Number.isFinite(val)) {
        throw new Error(`La ligne ${r + 2} contient une valeur non numérique: "${row[c]}".`);
      }
      if (c === yIdx) target[r] = val;
      else {
        vars[varNames[vi]][r] = val;
        vi++;
      }
    }
  }
  return { variables: varNames, vars, target, n };
}

export interface CustomRegressionConfig {
  csv: string;
  populationSize: number;
  generations: number;
  maxDepth: number;
}

export function runCustomRegressionPreset(cfg: CustomRegressionConfig): SpearRunResult {
  const { variables, vars, target, n } = parseCsvDataset(cfg.csv);

  const gpCfg: EvolveConfig = {
    variables,
    constRange: [-5, 5],
    ops: ["add", "sub", "mul", "pdiv", "relu", "abs", "neg", "sq", "max", "min"],
    maxDepth: clamp(cfg.maxDepth, 2, 6),
    populationSize: clamp(cfg.populationSize, 20, 300),
    generations: clamp(cfg.generations, 5, 80),
  };

  const fitnessFn: FitnessFn = (node) => {
    let pred: Float64Array;
    try {
      pred = evaluateNode(node, vars, n);
    } catch {
      return { fitness: -1e9, size: node.size };
    }
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(pred[i])) return { fitness: -1e9, size: node.size };
    }
    const m = mse(pred, target);
    const li = linfError(pred, target);
    if (!Number.isFinite(m)) return { fitness: -1e9, size: node.size };
    return { fitness: -(m + 0.05 * li), size: node.size, extra: li };
  };

  const result = evolve(gpCfg, fitnessFn);
  const predFinal = evaluateNode(result.best, vars, n);
  const finalMse = mse(predFinal, target);
  const finalLinf = linfError(predFinal, target);
  const r2 = r2Score(predFinal, target);

  const chartCount = Math.min(n, 80);
  const step = Math.max(1, Math.floor(n / chartCount));
  const chartData: ChartPoint[] = [];
  for (let i = 0; i < n; i += step) {
    chartData.push({ x: i, target: target[i], predicted: predFinal[i] });
  }

  return {
    formulaText: nodeToString(result.best),
    formulaTree: result.best,
    fitness: result.bestFitness,
    mse: finalMse,
    linfError: finalLinf,
    treeSize: result.best.size,
    durationMs: result.durationMs,
    history: result.history,
    metrics: { r2, variables, rows: n },
    chartData,
  };
}
