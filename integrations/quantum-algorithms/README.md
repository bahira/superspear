# Quantum Algorithms Integration

Community-contributed implementations of quantum control and algorithm
kernels. These are **reference implementations from the literature**, not
SPEAR discoveries — they serve as baselines for the SR benchmark tasks.

## Contents

| File | Source | Replaces |
|---|---|---|
| `manybody_cd.c` | Claeys et al. (2019) counter-diabatic driving for tilted Ising | Magnus-expansion variational solver (~1.2×10⁵× slower) |

## Usage

Compile with any C99 compiler:
```bash
gcc -O2 -march=native -o manybody_cd manybody_cd.c -lm
```

The kernel evaluates in O(1) per timestep (< 15 ns on ARM Cortex-M7 FPU).
Feed (g, dg/dt, J, hz) per timestep to generate CD amplitudes for AWG/FPGA
pulse controllers.

## Relationship to SPEAR

These kernels serve as **baselines** for the corresponding SPEAR benchmark
tasks (`grover_amplitude`, `chsh_correlation`, etc.). If SPEAR can discover
equivalent or cheaper forms through evolution, that validates the approach.
If not, it demonstrates the depth of the physics insight embedded in the
hand-crafted expressions.

## Adding new kernels

Submit via PR with: MISRA-C:2012 compliance proof, reference paper citation,
and numerical validation against the exact solution on ≥10⁶ points.
