// Numeric helpers shared by SPEAR presets.

export function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

export function linspace(start: number, end: number, n: number): Float64Array {
  const out = new Float64Array(n);
  if (n === 1) {
    out[0] = start;
    return out;
  }
  const step = (end - start) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = start + step * i;
  return out;
}

// Abramowitz & Stegun 7.1.26 approximation, max error ~1.5e-7
export function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * ax);
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

export function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

export function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function mapArray(arr: Float64Array, fn: (v: number) => number): Float64Array {
  const out = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = fn(arr[i]);
  return out;
}

export function mse(pred: Float64Array, target: Float64Array): number {
  let s = 0;
  for (let i = 0; i < pred.length; i++) {
    const d = pred[i] - target[i];
    s += d * d;
  }
  return s / pred.length;
}

export function linfError(pred: Float64Array, target: Float64Array): number {
  let m = 0;
  for (let i = 0; i < pred.length; i++) {
    const d = Math.abs(pred[i] - target[i]);
    if (d > m) m = d;
  }
  return m;
}

export function r2Score(pred: Float64Array, target: Float64Array): number {
  const n = target.length;
  let meanY = 0;
  for (let i = 0; i < n; i++) meanY += target[i];
  meanY /= n;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const dRes = target[i] - pred[i];
    ssRes += dRes * dRes;
    const dTot = target[i] - meanY;
    ssTot += dTot * dTot;
  }
  if (ssTot === 0) return 1;
  return 1 - ssRes / ssTot;
}

/**
 * Dataset noise source.
 *
 * DESIGN RULE (learned the hard way): the noise that builds a task's dataset
 * MUST NOT come from the search PRNG. When it did, every seed scored its
 * champion against a *different* realisation of the data, so:
 *   • "best across all seeds" compared numbers that were never comparable —
 *     it systematically crowned whichever seed drew the friendliest noise;
 *   • a stored record could not be reproduced, because re-evaluating the same
 *     AST after a different number of PRNG draws regenerated different data
 *     (free_fall wandered over 2.9e-4 … 5.1e-4 for one fixed formula).
 *
 * So the dataset stream is its own fixed-seed generator: identical in every
 * run, on every seed, in every script. Metrics become a property of the
 * formula alone — which is the only way a hall of fame means anything.
 *
 * `setUniformSource` remains for callers that deliberately want a different
 * realisation (noise-sensitivity studies); it does not affect the default.
 */
const DATASET_SEED = 0x5f3a_c91d;

function makeLcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    // xorshift32 — cheap, well-distributed, fully deterministic
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 0x1_0000_0000;
  };
}

let uniformSource: () => number = makeLcg(DATASET_SEED);

export function setUniformSource(fn: () => number): void {
  uniformSource = fn;
}

/**
 * Rewind the dataset stream to its canonical start. Dataset builders call this
 * so that task construction order never leaks into the data either.
 */
export function resetDatasetStream(seed: number = DATASET_SEED): void {
  uniformSource = makeLcg(seed);
}

/** Stable 32-bit hash of a dataset tag (FNV-1a). */
function hashTag(tag: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < tag.length; i++) {
    h ^= tag.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/**
 * Run a dataset builder on its OWN deterministic noise stream, keyed by a
 * stable tag. Two consequences that make the ledger trustworthy:
 *   • the data a task sees no longer depends on how many tasks were built
 *     before it, nor on the search seed;
 *   • re-evaluating a stored AST — in CI, in a script, a year later — hits
 *     exactly the dataset the record was set on.
 * The previous source is restored afterwards.
 */
export function withDataset<T>(tag: string, build: () => T): T {
  const prev = uniformSource;
  uniformSource = makeLcg((hashTag(tag) ^ DATASET_SEED) >>> 0);
  try { return build(); } finally { uniformSource = prev; }
}

/** A uniform draw from the dataset stream (never the search PRNG). */
export function datasetUniform(): number {
  return uniformSource();
}

export function gaussianRandom(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = uniformSource();
  while (v === 0) v = uniformSource();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
