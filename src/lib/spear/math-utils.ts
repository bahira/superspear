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
 * Box-Muller on an injectable uniform source. The SPEAR loop wires this to its
 * seeded PRNG so that datasets — and therefore every reported metric — are
 * bit-for-bit reproducible from the seed alone.
 */
let uniformSource: () => number = Math.random;

export function setUniformSource(fn: () => number): void {
  uniformSource = fn;
}

export function gaussianRandom(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = uniformSource();
  while (v === 0) v = uniformSource();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
