# 3DSPEAR integration kit — SPEAR-discovered kernels

Copy-paste kernels discovered and verified by the SPEAR engine, ready for a
Three.js city scene. Each file is self-contained JavaScript; `glsl/` holds the
shader translations.

## Contents

| File | Kernel | Replaces | Numbers |
|---|---|---|---|
| `js/blackbody.js` | red channel vs color temperature | Tanner Helland piecewise fit | L4, MSE 3.6e-6, ×1.79 faster |
| `js/kdvWave.js` | KdV soliton crest (water) | `2·sech²(...)` per vertex | 7 ALU units, ×1.96 wall-clock measured |
| `js/ikElbow.js` | 2-link elbow angle (radians) | Newton-DLS iterative chains | EXACT 1.1e-13, ×20 vs 8-iteration solve |
| `js/gaussianBloom.js` | gaussian weight e^(-x²/2)-grade | exp-based kernel tap | 9 ALU units, MSE 6.4e-4 |
| `glsl/effects.frag.glsl` | GLSL versions of all four + exact smoothstep | shader equivalents | — |

## Integration notes

- **Gaussian**: for *static* blurs precompute your taps once — the win here is
  **per-pixel variable-σ** evaluation (depth of field, local bloom radius).
  On backends with fast native `exp`, measure first: our form trades
  transcendental cost for ALU ops.
- **KdV crest / blackbody**: designed for per-frame per-vertex/per-light
  evaluation — that is where the closed-form beats everything.
- **IK elbow**: valid across the whole reachable domain; clamp the ratio
  input if you feed it beyond-spec reaches.

## Provenance

Every formula was evolved by the SPEAR engine against reference data,
validated through its parity-audited export pipeline (JS/WASM/C agree to
float precision), and recorded in `spear-hall-of-fame.json` with cost
(ALU/SFU units), accuracy, and speedup metadata.
