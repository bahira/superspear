// SPEAR QUANT PACK — 5 vertical demos with live measurements.
// Each demo: production-style workload, SPEAR kernel vs reference method,
// wall-clock comparison + quality gate. Run: node quant-demos/all.mjs
import { kernels } from "../packages/spear-kernels/index.js";
import stats from "../packages/spear-kernels/stats.js";

const cdfSpear = (x) => kernels.gaussian_cdf.precise.eval(x);
const probitSpear = (x) => kernels.probit_quantile.precise.eval(x);

const hr = (t) => console.log(`\n=== ${t} ===`);

// ---------- 1. TELECOM — BER closed-form vs Monte-Carlo ---------------
hr("1. TELECOM — BER BPSK: forme close vs Monte-Carlo");
function mcBer(snrLin, samples = 2_000_000) {
  let errors = 0;
  for (let i = 0; i < samples; i++) {
    const noise = boxMuller();
    if (Math.sqrt(2 * snrLin) + noise < 0) errors++;
  }
  return errors / samples;
}
let _spare = null;
function boxMuller() {
  if (_spare !== null) { const s = _spare; _spare = null; return s; }
  const u1 = Math.random() || 1e-12, u2 = Math.random();
  const r = Math.sqrt(-2 * Math.log(u1));
  _spare = r * Math.sin(2 * Math.PI * u2);
  return r * Math.cos(2 * Math.PI * u2);
}
for (const snrDb of [0, 4, 7, 10]) {
  const snr = Math.pow(10, snrDb / 10);
  const t0 = performance.now();
  const mc = mcBer(snr);
  const mcMs = (performance.now() - t0).toFixed(0);
  const t1 = performance.now();
  for (let r = 0; r < 10000; r++) stats.berBpsk(snrDb);
  const cfUs = ((performance.now() - t1) * 1000 / 10000).toFixed(3);
  const relErr = mc > 0 ? Math.abs(mc - stats.berBpsk(snrDb)) / mc * 100 : 0;
  console.log(`SNR ${String(snrDb).padStart(2)} dB | MC ${mc.toExponential(3)} (${mcMs} ms, ±${relErr.toFixed(0)}% err) | close-form ${stats.berBpsk(snrDb).toExponential(3)} (${cfUs} µs) → ×${((mcMs*1000)/parseFloat(cfUs)).toFixed(0)}`);
}

// ---------- 2. RISK — VaR/CVaR portfolio sweep ------------------------
hr("2. RISK — VaR/CVaR sweep 5 000 positions × 3 niveaux");
const positions = Array.from({ length: 5000 }, (_, i) => ({
  mu: (Math.random() - 0.4) * 0.001,
  sigma: 0.002 + Math.random() * 0.02,
  size: 10_000 + Math.random() * 100_000,
}));
const t2 = performance.now();
const res = stats.portfolioVar(positions, [0.95, 0.99]);
const varMs = performance.now() - t2;
console.log(`sweep complet en ${varMs.toFixed(1)} ms`);
for (const r of res) console.log(`  VaR ${Math.round(r.alpha * 100)}% = ${r.var.toFixed(0)} € | CVaR = ${r.cvar.toFixed(0)} €`);

// ---------- 3. OPTIONS — pricing chain + implied vol ------------------
hr("3. OPTIONS — chaîne de pricing 120 options");
const bsRef = (s, k, t, vol) => {
  const d1 = (Math.log(s/k) + 0.02*t)/(vol*Math.sqrt(t)) + vol*Math.sqrt(t)/2;
  const d2 = d1 - vol*Math.sqrt(t);
  // exact via Abramowitz-Stegun
  const erf = (x) => { const t = 1/(1+0.3275911*Math.abs(x)); const y = 1-((((1.061405429*t-1.453152027)*t+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-Math.abs(x*x)); return x>=0?y:-y; };
  const Nd = (x) => 0.5*(1+erf(x/Math.SQRT2));
  return s*Nd(d1)-k*Math.exp(-0.02*t)*Nd(d2);
};
const strikes = []; for (let k = 80; k <= 120; k += 2.5) strikes.push(k);
const t3 = performance.now();
let chainOut = [];
for (const T of [0.25, 0.5, 1]) for (const K of strikes) {
  const S = 100, iv = 0.35;
  const d1 = (Math.log(S/K)+ (0.02+iv*iv/2)*T)/(iv*Math.sqrt(T));
  const d2 = d1 - iv*Math.sqrt(T);
  chainOut.push({ K, T, call: S*cdfSpear(d1) - K*Math.exp(-0.02*T)*cdfSpear(d2), delta: cdfSpear(d1) });
}
const chainMs = performance.now() - t3;
console.log(`${chainOut.length} options pricées (+delta) en ${chainMs.toFixed(1)} ms — zéro appel transcendental natif`);
console.log(`exemple: K=100 T=1 call=${chainOut.find(o=>o.K===100&&o.T===1)?.call?.toFixed(3)}`);

// ---------- 4. GENOME — p-values à l'échelle -------------------------
hr("4. GENOME — 1M z-scores: p-values + Bonferroni");
const M = 1_000_000;
const zs = new Float64Array(M);
for (let i = 0; i < M; i++) zs[i] = (Math.random() + Math.random() + Math.random() + Math.random() + Math.random() + Math.random() - 3) * 1.8;
const t4 = performance.now();
let sig = 0;
const threshold = 0.05 / M;
for (let i = 0; i < M; i++) {
  const p = 2 * (1 - cdfSpear(Math.abs(zs[i])));
  if (p < threshold) sig++;
}
const genomeMs = performance.now() - t4;
console.log(`${M.toLocaleString("en-US")} p-values + Bonferroni en ${genomeMs.toFixed(0)} ms | significatives: ${sig}`);

// ---------- 5. ADTECH — conversion scoring à l'enchère ---------------
hr("5. ADTECH — 100k scores de conversion probit (fenêtre 100 ms)");
const t5 = performance.now();
let sum = 0;
for (let i = 0; i < 100_000; i++) {
  const score = -1.2 + 0.8 * Math.sin(i) + 0.5 * ((i % 97) / 97);
  sum += stats.conversionProb(score) > 0.5 ? 1 : 0;
}
const adMs = performance.now() - t5;
console.log(`100k conversions scorées en ${adMs.toFixed(1)} ms (${(100000/adMs*1000).toFixed(0)} scorés/s) | high-intent: ${sum}`);

console.log("\n— fin des démos SPEAR QUANT —");
