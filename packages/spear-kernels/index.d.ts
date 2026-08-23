/** One executable bundle for a discovered formula. */
export interface KernelVariant {
  /** Standalone arrow-function source, e.g. `"((x) => ...)"`. */
  js: string;
  /** The same arrow, precompiled and ready to call. */
  eval: (...args: number[]) => number;
  /** CUDA C (`__device__`) source. */
  c: string;
  /** PyTorch (`torch.*`) source. */
  py: string;
  /** Compiled WebAssembly binary (f64 params -> f64 result), base64-encoded. */
  wasmBase64: string;
}

export interface KernelMeta {
  /** Fitness achieved by the discovered formula (lower is better). */
  metric: number | null;
  level: number | null;
  formulaCost: number | null;
  exactCost: number | null;
  speedupVsExact: number | null;
  vsIterative: number | null;
}

export interface Kernel {
  id: string;
  precise: KernelVariant;
  fast?: KernelVariant;
  meta: KernelMeta;
}

export declare const kernels: Record<string, Kernel>;
export declare const kernelIds: string[];
declare const _default: Record<string, Kernel>;
export default _default;
