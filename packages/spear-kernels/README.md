# spear-kernels

Every closed-form kernel discovered by the [SPEAR](https://github.com/) symbolic
regression engine, published as a zero-dependency, fully self-contained package.
Each kernel ships four interchangeable backends plus metadata:

| Field | What it is |
|---|---|
| `precise.js` / `precise.eval` | standalone JS arrow-function (source string / compiled) |
| `precise.c` | CUDA C `__device__` source |
| `precise.py` | PyTorch source |
| `precise.wasmBase64` | compiled f64 WebAssembly module |
| `fast` | algebraic-only variant when one was discovered |
| `meta` | `metric`, `level`, `formulaCost`, `exactCost`, `speedupVsExact`, `vsIterative` |

## Usage

```js
import { kernels, kernelIds } from "spear-kernels";

kernelIds;                       // list of all kernel ids

kernels.silu.precise.eval(2);    // ~1.762 (SiLU approximant)

if (kernels.sigmoid.fast) {
  kernels.sigmoid.fast.eval(3);  // algebraic fast path
}

kernels.gaussian_cdf.meta.speedupVsExact; // metadata access
```

### WebAssembly backend

```js
const bytes = Uint8Array.from(atob(kernels.silu.precise.wasmBase64), (c) => c.charCodeAt(0));
const env = {
  exp: Math.exp, sin: Math.sin, cos: Math.cos, atan: Math.atan,
  log: (v) => Math.log(v > 1e-30 ? v : 1e-30),
};
const { instance } = await WebAssembly.instantiate(bytes, { env });
instance.exports.spear(2); // same protected-division semantics as the JS backend
```

Division is **protected** exactly like the engine: denominators floored at
±1e-4 (sign-preserving), results clamped to ±1e4. Discovered formulas rely on
these rails — do not swap in bare `/`.

## Support

If SPEAR kernels save you compute time, consider tipping the lab:

₿ **bc1q54n9x2894rr43f7nkaqywtegn8aufxwlnnkh5r**

`npm fund` also surfaces this address (declared in the `funding` field).
Every satoshi goes back into GPU time for the evolution farm.
