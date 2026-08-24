// spear-kernels/stats — vertical pack: telecom BER, risk (VaR/CVaR),
// genome-scale p-values, ad-tech conversion scoring.
// Built on the parity-audited SPEAR kernels (cdf/probit/pdf).
// Self-contained ESM. MIT license.

import { kernels } from "./index.js";

const cdfNode = kernels.gaussian_cdf.precise.eval; // takes a number
const probitNode = kernels.probit_quantile.precise.eval;

/** Normal PDF φ(x) */
export const pdf = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

/** Normal CDF Φ(x) — SPEAR algebraic form (no native erf needed) */
export const cdf = (x) => cdfNode(x);

/** Probit Φ⁻¹(p) — inverse, p ∈ (0,1) */
export const probit = (p) => probitNode(p);

/** Q-function = 1 − Φ(x) (upper tail) */
export const qfunc = (x) => 1 - cdf(x);

// ---------- Telecom: bit error rates ---------------------------------
/** BPSK/QPSK BER over AWGN from SNR in dB. Exact closed form via Q(). */
export const berBpsk = (snrDb) => qfunc(Math.sqrt(2 * Math.pow(10, snrDb / 10)));

/** M-QAM approximate BER (square M-QAM, Gray coding). */
export const berQam = (m, snrDb) => {
  const k = Math.log2(m);
  const snr = Math.pow(10, snrDb / 10);
  return (4 / k) * (1 - 1 / Math.sqrt(m)) * qfunc(Math.sqrt((3 * k * snr) / (m - 1)));
};

// ---------- Risk: parametric VaR / CVaR ------------------------------
/** Parametric VaR at confidence α (e.g. 0.95, 0.99) for N(μ,σ) P&L. */
export const varNormal = (mu, sigma, alpha = 0.95) =>
  -(mu - sigma * probit(alpha));

/**
 * Portfolio sweep: positions [{mu, sigma, size}] × scenarios.
 * Returns portfolio-level parametric VaR/CVaR assuming independence.
 * Aggregated μ,σ via moments; uses SPEAR probit/cdf internally.
 */
export function portfolioVar(positions, scenarios = [0.95, 0.99]) {
  let mu = 0, varSum = 0;
  for (const p of positions) {
    mu += p.mu * p.size;
    varSum += p.sigma * p.sigma * p.size * p.size;
  }
  const sigma = Math.sqrt(varSum);
  return scenarios.map((alpha) => {
    const z = probit(alpha);
    const varAlpha = -(mu - sigma * z);
    // Normal CVaR closed form: μ − σ·φ(z)/(1−α)
    const cvarAlpha = -(mu - (sigma * pdf(z)) / (1 - alpha));
    return { alpha, var: varAlpha, cvar: cvarAlpha };
  });
}

// ---------- Bioinformatics: genome-scale p-values --------------------
/** Two-sided p-value from a z-score. */
export const pValue = (z) => 2 * qfunc(Math.abs(z));

/** Bonferroni correction over n tests: returns significance flags. */
export function bonferroni(zScores, familyAlpha = 0.05) {
  const threshold = familyAlpha / zScores.length;
  return zScores.map((z) => ({ z, p: pValue(z), significant: pValue(z) < threshold }));
}

// ---------- Ad-tech: conversion probability scoring ------------------
/** Probit-link conversion probability from a linear score. */
export const conversionProb = (score) => cdf(score);

export default { pdf, cdf, probit, qfunc, berBpsk, berQam, varNormal, portfolioVar, pValue, bonferroni, conversionProb };
