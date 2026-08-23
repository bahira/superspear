// SPEAR â€” Symbolic Pareto Evolutionary Algorithm for Research
// Engine v2: seeded RNG, algebraic simplification, affine wrapping, constant
// refinement (coordinate descent), full NSGA-II (non-dominated sort +
// crowding distance), Pareto archive and code generation.

export type NodeOp =
  | "var" | "const"
  | "add" | "sub" | "mul" | "pdiv"
  | "relu" | "abs" | "neg" | "sq" | "sqrt" | "cube"
  | "max" | "min"
  | "exp" | "sin" | "cos" | "log" | "atan";

export interface SpearNode {
  op: NodeOp;
  value: number;
  name: string;
  children: SpearNode[];
  size: number;
  depth: number;
}

const BINARY = new Set<NodeOp>(["add", "sub", "mul", "pdiv", "max", "min"]);
const UNARY = new Set<NodeOp>(["relu", "abs", "neg", "sq", "sqrt", "cube", "exp", "atan", "sin", "cos", "log"]);

export const ALL_OPS: NodeOp[] = [
  "add", "sub", "mul", "pdiv", "relu", "abs", "neg", "sq", "sqrt", "cube", "max", "atan", "min", "exp", "sin", "cos", "log",
];

/** Ops that a GPU/TPU executes as plain algebra (no transcendental unit). */
export const ALGEBRAIC_OPS = new Set<NodeOp>([
  "add", "sub", "mul", "pdiv", "relu", "abs", "neg", "sq", "cube", "max", "min",
]);

export function arity(op: NodeOp): number {
  if (op === "var" || op === "const") return 0;
  if (BINARY.has(op)) return 2;
  if (UNARY.has(op)) return 1;
  return 0;
}

// ---------------------------------------------------------------- seeded RNG
let rngState = 0x2f6e2b1;

export function setSeed(seed: number): void {
  rngState = (seed >>> 0) || 1;
}

export function rand(): number {
  rngState = (rngState + 0x6d2b79f5) >>> 0;
  let t = rngState;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function randRange(min: number, max: number): number {
  return min + rand() * (max - min);
}

export function randInt(maxExclusive: number): number {
  return Math.floor(rand() * maxExclusive);
}

function pick<T>(arr: readonly T[]): T {
  return arr[randInt(arr.length)];
}

/** Relative cost of each operator in ALU/SFU units on a modern GPU. */
export const OP_COST: Record<NodeOp, number> = {
  var: 0, const: 0,
  add: 1, sub: 1, mul: 1,
  pdiv: 4, relu: 1, abs: 1, neg: 1, sq: 1, cube: 2, sqrt: 2, max: 1, min: 1,
  exp: 20, sin: 20, cos: 20, atan: 20, log: 20,
};

export function estimateCost(node: SpearNode): number {
  let total = 0;
  const walk = (nd: SpearNode) => {
    total += OP_COST[nd.op] ?? 1;
    nd.children.forEach(walk);
  };
  walk(node);
  return total;
}

/** Backward-elimination pruning: drop subtrees that do not hurt the score. */
export function prune(
  node: SpearNode,
  score: (n: SpearNode) => number,
  tolerance = 1.002,
): { node: SpearNode; score: number; removed: number } {
  let current = cloneNode(node);
  let currentScore = score(current);
  let removed = 0;
  for (let pass = 0; pass < 3; pass++) {
    const positions: SpearNode[] = [];
    const collect = (nd: SpearNode) => {
      if (nd.children.length > 0) positions.push(nd);
      nd.children.forEach(collect);
    };
    collect(current);
    let improvedThisPass = false;
    for (const target of positions) {
      const candidates: SpearNode[] = [];
      if (target.children[0]) candidates.push(target.children[0]);
      if (target.children[1]) candidates.push(target.children[1]);
      for (const rep of candidates) {
        const trial = replaceForPrune(current, target, cloneNode(rep));
        const s = score(trial);
        if (Number.isFinite(s) && s <= currentScore * tolerance) {
          current = trial;
          currentScore = s;
          removed++;
          improvedThisPass = true;
          break;
        }
      }
    }
    if (!improvedThisPass) break;
  }
  const cleaned = simplify(current);
  const cleanedScore = score(cleaned);
  if (Number.isFinite(cleanedScore) && cleanedScore <= currentScore * tolerance) {
    return { node: cleaned, score: cleanedScore, removed };
  }
  return { node: current, score: currentScore, removed };
}

function replaceForPrune(root: SpearNode, target: SpearNode, rep: SpearNode): SpearNode {
  if (root === target) return rep;
  if (root.children.length === 0) return root;
  return makeNode(root.op, {
    value: root.value, name: root.name,
    children: root.children.map((c) => replaceForPrune(c, target, rep)),
  });
}

// ---------------------------------------------------------------- node utils
export function makeNode(
  op: NodeOp,
  opts: { value?: number; name?: string; children?: SpearNode[] } = {},
): SpearNode {
  const children = opts.children ?? [];
  const size = 1 + children.reduce((s, c) => s + c.size, 0);
  const depth = 1 + children.reduce((m, c) => Math.max(m, c.depth), 0);
  return { op, value: opts.value ?? 0, name: opts.name ?? "", children, size, depth };
}

// ------------------------------------------------------- (de)serialization
export interface SerializedNode {
  o: NodeOp;
  v?: number;
  n?: string;
  c?: SerializedNode[];
}

export function serializeNode(node: SpearNode): SerializedNode {
  return {
    o: node.op,
    ...(node.op === "const" ? { v: node.value } : {}),
    ...(node.op === "var" ? { n: node.name } : {}),
    ...(node.children.length > 0 ? { c: node.children.map(serializeNode) } : {}),
  };
}

export function parseNode(s: SerializedNode): SpearNode {
  return makeNode(s.o, {
    value: s.v ?? 0,
    name: s.n ?? "",
    children: (s.c ?? []).map(parseNode),
  });
}

export function cloneNode(n: SpearNode): SpearNode {
  return makeNode(n.op, {
    value: n.value,
    name: n.name,
    children: n.children.map(cloneNode),
  });
}

export function roundConst(v: number): number {
  return Math.abs(v) < 1e-6 ? 0 : Number(v.toFixed(6));
}

// ---------------------------------------------------------------- evaluation
export function evaluateNode(
  node: SpearNode,
  vars: Record<string, Float64Array>,
  n: number,
): Float64Array {
  switch (node.op) {
    case "var": {
      const v = vars[node.name];
      if (!v) throw new Error(`Unknown variable "${node.name}"`);
      return v;
    }
    case "const": {
      const out = new Float64Array(n);
      out.fill(node.value);
      return out;
    }
    default: break;
  }
  if (BINARY.has(node.op)) {
    const a = evaluateNode(node.children[0], vars, n);
    const b = evaluateNode(node.children[1], vars, n);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const av = a[i];
      const bv = b[i];
      let r = 0;
      switch (node.op) {
        case "add": r = av + bv; break;
        case "sub": r = av - bv; break;
        case "mul": r = av * bv; break;
        case "pdiv": {
          let d = bv;
          if (d > -1e-4 && d < 1e-4) d = d >= 0 ? 1e-4 : -1e-4;
          r = av / d;
          if (r > 1e4) r = 1e4;
          else if (r < -1e4) r = -1e4;
          break;
        }
        case "max": r = av > bv ? av : bv; break;
        case "min": r = av < bv ? av : bv; break;
        default: break;
      }
      out[i] = r;
    }
    return out;
  }
  const a = evaluateNode(node.children[0], vars, n);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const av = a[i];
    let r = 0;
    switch (node.op) {
      case "relu": r = av > 0 ? av : 0; break;
      case "abs": r = av < 0 ? -av : av; break;
      case "neg": r = -av; break;
      case "sq": r = av * av; break;
      case "cube": r = av * av * av; break;
      case "sqrt": r = Math.sqrt(av < 0 ? -av : av); break;
      case "exp": r = Math.exp(Math.max(-50, Math.min(50, av))); break;
      case "sin": r = Math.sin(av); break;
      case "cos": r = Math.cos(av); break;
        case "atan": r = Math.atan(av); break;
      case "log": r = Math.log(av > 1e-30 ? av : 1e-30); break;
      default: break;
    }
    out[i] = r;
  }
  return out;
}

export function evaluateScalar(node: SpearNode, scope: Record<string, number>): number {
  switch (node.op) {
    case "var": return scope[node.name] ?? 0;
    case "const": return node.value;
    default: break;
  }
  const a = node.children.map((c) => evaluateScalar(c, scope));
  switch (node.op) {
    case "add": return a[0] + a[1];
    case "sub": return a[0] - a[1];
    case "mul": return a[0] * a[1];
    case "pdiv": {
      let d = a[1];
      if (d > -1e-4 && d < 1e-4) d = d >= 0 ? 1e-4 : -1e-4;
      const r = a[0] / d;
      return Math.max(-1e4, Math.min(1e4, r));
    }
    case "max": return Math.max(a[0], a[1]);
    case "min": return Math.min(a[0], a[1]);
    case "relu": return a[0] > 0 ? a[0] : 0;
    case "abs": return Math.abs(a[0]);
    case "neg": return -a[0];
    case "sq": return a[0] * a[0];
    case "cube": return a[0] * a[0] * a[0];
    case "sqrt": return Math.sqrt(Math.abs(a[0]));
    case "exp": return Math.exp(Math.max(-50, Math.min(50, a[0])));
    case "sin": return Math.sin(a[0]);
      case "cos": return Math.cos(a[0]);
      case "atan": return Math.atan(a[0]);
      case "log": return Math.log(a[0] > 1e-30 ? a[0] : 1e-30);
    default: return 0;
  }
}

// ---------------------------------------------------------------- printing
function fmt(v: number): string {
  const r = roundConst(v);
  return Number.isInteger(r) ? r.toString() : r.toString();
}

export function nodeToString(node: SpearNode): string {
  switch (node.op) {
    case "var": return node.name ?? "?";
    case "const": return fmt(node.value);
    case "relu": return `relu(${nodeToString(node.children[0])})`;
    case "abs": return `|${nodeToString(node.children[0])}|`;
    case "neg": return `(-${nodeToString(node.children[0])})`;
    case "sq": return `(${nodeToString(node.children[0])})Â²`;
    case "cube": return `(${nodeToString(node.children[0])})Â³`;
    case "sqrt": return `sqrt(|${nodeToString(node.children[0])}|)`;
    case "exp": return `exp(${nodeToString(node.children[0])})`;
    case "sin": return `sin(${nodeToString(node.children[0])})`;
    case "cos": return `cos(${nodeToString(node.children[0])})`;
    case "atan": return `atan(${nodeToString(node.children[0])})`;
    case "log": return `log(max(${nodeToString(node.children[0])}, 1e-30))`;
    case "max": return `max(${nodeToString(node.children[0])}, ${nodeToString(node.children[1])})`;
    case "min": return `min(${nodeToString(node.children[0])}, ${nodeToString(node.children[1])})`;
    case "pdiv": return `(${nodeToString(node.children[0])} / ${nodeToString(node.children[1])})`;
    case "add": return `(${nodeToString(node.children[0])} + ${nodeToString(node.children[1])})`;
    case "sub": return `(${nodeToString(node.children[0])} - ${nodeToString(node.children[1])})`;
    case "mul": return `(${nodeToString(node.children[0])} * ${nodeToString(node.children[1])})`;
    default: return "?";
  }
}

// ------------------------------------------------------------ formula parser
// Inverse of nodeToString: reconstructs an AST from a printed formula.
// Display-rounding caveat: constants come back at their printed (6-decimal)
// precision, so parsed trees may differ from originals in far decimals.

export function parseFormula(src: string): SpearNode {
  // repair double-encoded superscripts (Â² -> ²) left by old Windows writes
  src = src.replace(/\u00C2\u00B2/g, "\u00B2").replace(/\u00C2\u00B3/g, "\u00B3");
  const toks: string[] = [];
  {
    const NUMRE = /^[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?/;
    let i = 0;
    let prev: "#" | "v" | "op" | null = null;
    while (i < src.length) {
      const c = src[i];
      if (c === " ") { i++; continue; }
      // '-' merged into a numeric literal only in operand-start position
      if (c === "-" && NUMRE.test(src.slice(i + 1)) && prev !== "v" && prev !== "#") {
        const m = src.slice(i + 1).match(NUMRE) as RegExpMatchArray;
        toks.push("-" + m[0]); prev = "#"; i += 1 + m[0].length; continue;
      }
      if (/[0-9.]/.test(c)) {
        const m = src.slice(i).match(NUMRE) as RegExpMatchArray;
        toks.push(m[0]); prev = "#"; i += m[0].length; continue;
      }
      if (/[a-z_]/i.test(c)) {
        let j = i + 1;
        while (j < src.length && /[a-z0-9_]/i.test(src[j])) j++;
        toks.push(src.slice(i, j)); prev = "v"; i = j; continue;
      }
      toks.push(c); prev = "op"; i++;
    }
  }
  let p = 0;
  const peek = () => toks[p];
  const eat = (t?: string): string => {
    const tok = toks[p];
    if (t !== undefined && tok !== t) throw new Error(`parseFormula: attendu "${t}", trouvé "${tok}" @${p} dans ${src}`);
    p++;
    return tok;
  };

  const BIN: Record<string, NodeOp> = { "+": "add", "-": "sub", "*": "mul", "/": "pdiv" };
  const FUNCS = new Set(["relu", "sqrt", "exp", "sin", "cos", "log", "atan", "max", "min"]);

  const atom = (): SpearNode => {
    const t = peek();
    if (t === "(") {
      eat("(");
      if (peek() === "-") { // (-E) negation form
        eat("-");
        const e = expr();
        eat(")");
        return makeNode("neg", { children: [e] });
      }
      const e1 = expr();
      if (peek() !== undefined && peek() in BIN) {
        const op = BIN[eat()];
        const e2 = expr();
        eat(")");
        return makeNode(op, { children: [e1, e2] });
      }
      eat(")");
      if (peek() === "\u00B2") { eat(); return makeNode("sq", { children: [e1] }); }
      if (peek() === "\u00B3") { eat(); return makeNode("cube", { children: [e1] }); }
      return e1;
    }
    if (t === "|") { eat("|"); const e = expr(); eat("|"); return makeNode("abs", { children: [e] }); }
    if (t === "-") { eat("-"); return makeNode("neg", { children: [atom()] }); }
    if (/[a-z_]/.test(t[0])) {
      eat();
      if (peek() === "(" && FUNCS.has(t)) {
        eat("(");
        const args: SpearNode[] = [expr()];
        while (peek() === ",") { eat(","); args.push(expr()); }
        eat(")");
        if (t === "log" && args.length === 1 && args[0].op === "max" &&
            args[0].children[1].op === "const" && Math.abs(args[0].children[1].value - 1e-30) < 1e-24) {
          return makeNode("log", { children: [args[0].children[0]] }); // unwrap printed guard
        }
        if (t === "sqrt" && args.length === 1 && args[0].op === "abs") {
          return makeNode("sqrt", { children: [args[0].children[0]] }); // printer adds its own |·|
        }
        return makeNode(t as NodeOp, { children: args });
      }
      return makeNode("var", { name: t });
    }
    return makeNode("const", { value: Number(eat()) });
  };

  const expr = (): SpearNode => atom();

  const out = expr();
  if (p < toks.length) throw new Error(`parseFormula: ${toks.length - p} tokens restants dans ${src}`);
  return out;
}

const PY: Record<NodeOp, string> = {
  var: "", const: "",
  add: "({a} + {b})", sub: "({a} - {b})", mul: "({a} * {b})",
  pdiv: "({a} / (abs({b}) < 1e-4 ? 1e-4 : {b}))",
  relu: "torch.relu({a})", abs: "torch.abs({a})", neg: "(-{a})",
  sq: "({a} * {a})", cube: "({a} * {a} * {a})", sqrt: "torch.sqrt(torch.abs({a}))",
  max: "torch.maximum({a}, {b})", min: "torch.minimum({a}, {b})",
  exp: "torch.exp(torch.clamp({a}, -50.0, 50.0))",
  sin: "torch.sin({a})", cos: "torch.cos({a})",
  log: "torch.log(torch.clamp({a}, min=1e-30))",
    atan: "torch.atan({a})",
};

export function toPython(node: SpearNode, fnName = "spear_fn"): string {
  const walk = (nd: SpearNode): string => {
    if (!nd) return "0";
    if (nd.op === "var") return nd.name;
    if (nd.op === "const") return fmt(nd.value);
    const tpl = PY[nd.op];
    if (!tpl) {
      // eslint-disable-next-line no-console
      console.error(`[toPython] op inconnu "${String(nd.op)}" â€” arbre: ${JSON.stringify(node).slice(0, 300)}`);
      return "0";
    }
    if (nd.op === "cube") return `(${walk(nd.children[0])} ** 3)`;
    if (UNARY.has(nd.op)) return tpl.replace(/\{a\}/g, walk(nd.children[0]));
    return tpl.replace(/\{a\}/g, walk(nd.children[0])).replace(/\{b\}/g, walk(nd.children[1]));
  };
  const args = [...new Set(collectVarNames(node))].join(", ");
  return `import torch\n\ndef ${fnName}(${args}):\n    # Evolved by SPEAR â€” zero transcendental ops (exp/erf/tanh free)\n    return ${walk(node)}`;
}

export function toC(node: SpearNode, fnName = "spear_fn", varDecl = "const float x"): string {
  const walk = (nd: SpearNode): string => {
    if (nd.op === "var") return nd.name;
    if (nd.op === "const") return `${fmt(nd.value)}f`;
    switch (nd.op) {
      case "relu": return `fmaxf(0.0f, ${walk(nd.children[0])})`;
      case "abs": return `fabsf(${walk(nd.children[0])})`;
      case "neg": return `(-${walk(nd.children[0])})`;
      case "sq": return `powf(${walk(nd.children[0])}, 2.0f)`;
      case "cube": return `powf(${walk(nd.children[0])}, 3.0f)`;
      case "sqrt": return `sqrtf(fabsf(${walk(nd.children[0])}))`;
      case "exp": return `expf(fmaxf(-50.0f, fminf(50.0f, ${walk(nd.children[0])})))`;
      case "sin": return `sinf(${walk(nd.children[0])})`;
      case "cos": return `cosf(${walk(nd.children[0])})`;
      case "atan": return `atanf(${walk(nd.children[0])})`;
      case "log": return `logf(fmaxf(1.0e-30f, ${walk(nd.children[0])}))`;
      case "max": return `fmaxf(${walk(nd.children[0])}, ${walk(nd.children[1])})`;
      case "min": return `fminf(${walk(nd.children[0])}, ${walk(nd.children[1])})`;
      // protected division — matches evaluateNode rails (±1e-4 floor, ±1e4 clamp)
      case "pdiv": return `fminf(fmaxf((${walk(nd.children[0])} / copysignf(fmaxf(fabsf(${walk(nd.children[1])}), 1.0e-4f), ${walk(nd.children[1])})), -1.0e4f), 1.0e4f)`;
      default: break;
    }
    return `(${walk(nd.children[0])} ${nd.op === "add" ? "+" : nd.op === "sub" ? "-" : "*"} ${walk(nd.children[1])})`;
  };
  return `// Evolved by SPEAR â€” algebraic only, FP16 safe\n__device__ inline float ${fnName}(${varDecl}) {\n    return ${walk(node)};\n}`;
}

/**
 * MISRA-C:2012 strict emission. Differences from toC():
 * - fixed-width stdint types (float32_t), no implicit narrowing
 * - branchless: relu/max/min/exp/log via fminf/fmaxf only, zero if/for/while/goto
 * - FMA-friendly Horner shapes: sq/cube emitted as repeated multiplies (no powf)
 * - zero heap: pure scalar function, no pointers written, no dynamic allocation
 */
export function toMisraC(node: SpearNode, fnName = "spear_fn", params = "const float32_t x"): string {
  const walk = (nd: SpearNode): string => {
    if (nd.op === "var") return nd.name;
    if (nd.op === "const") {
      // integers need a decimal point before the F suffix ("1F" is illegal C)
      const r = roundConst(nd.value);
      return `${Number.isInteger(r) ? r.toFixed(1) : r}F`;
    }
    switch (nd.op) {
      case "relu": return `fmaxf(0.0F, ${walk(nd.children[0])})`;
      case "abs": return `fabsf(${walk(nd.children[0])})`;
      case "neg": return `(-${walk(nd.children[0])})`;
      // FMA-friendly: x*x and x*x*x compile to fma chains, unlike powf calls
      case "sq": return `(${walk(nd.children[0])} * ${walk(nd.children[0])})`;
      case "cube": return `(${walk(nd.children[0])} * ${walk(nd.children[0])} * ${walk(nd.children[0])})`;
      case "sqrt": return `sqrtf(fabsf(${walk(nd.children[0])}))`;
      case "exp": return `expf(fminf(fmaxf(${walk(nd.children[0])}, -50.0F), 50.0F))`;
      case "sin": return `sinf(${walk(nd.children[0])})`;
      case "cos": return `cosf(${walk(nd.children[0])})`;
      case "atan": return `atanf(${walk(nd.children[0])})`;
      case "log": return `logf(fmaxf(1.0e-30F, ${walk(nd.children[0])}))`;
      case "max": return `fmaxf(${walk(nd.children[0])}, ${walk(nd.children[1])})`;
      case "min": return `fminf(${walk(nd.children[0])}, ${walk(nd.children[1])})`;
      // protected division — bit-faithful to evaluateNode (denominator floored
      // at ±1e-4, result clamped ±1e4): discovered forms lean on these rails
      case "pdiv":
        return `fminf(fmaxf((${walk(nd.children[0])} / copysignf(fmaxf(fabsf(${walk(nd.children[1])}), 1.0e-4F), ${walk(nd.children[1])})), -1.0e4F), 1.0e4F)`;
      default: break;
    }
    const opSym = nd.op === "add" ? "+" : nd.op === "sub" ? "-" : "*";
    return `(${walk(nd.children[0])} ${opSym} ${walk(nd.children[1])})`;
  };
  return [
    "/* Generated by SPEAR - strict C99, branchless, FMA-friendly, zero heap */",
    "#include <stdint.h>",
    "#include <math.h>",
    "",
    "/* Platform fixed-width float type (MISRA-C:2012 dir 4.5 - defined per target) */",
    "typedef float float32_t;",
    "",
    `float32_t ${fnName}(${params})`,
    "{",
    `    return ${walk(node)};`,
    "}",
  ].join("\n");
}

/** Static MISRA audit of generated code: returns every violation found. */
export function lintMisraC(code: string): { ok: boolean; violations: string[] } {
  const violations: string[] = [];
  // the platform typedef itself is the one sanctioned bare-float occurrence
  const body = code.replace(/\/\*[\s\S]*?\*\//g, "").replace(/typedef float float32_t;/g, "");
  const checks: { token: RegExp; label: string }[] = [
    { token: /\bmalloc\b|\bcalloc\b|\brealloc\b|\bfree\b|\bnew\b/, label: "allocation dynamique" },
    { token: /\bfor\s*\(|\bwhile\s*\(|\bdo\b/, label: "boucle" },
    { token: /\bif\s*\(|\bswitch\s*\(|\bgoto\b|\bcase\b/, label: "branche" },
    { token: /\bfloat\b(?!32_t)|\bdouble\b/, label: "type flottant non fixe" },
    { token: /\bpowf\s*\(/, label: "powf (non FMA-friendly)" },
  ];
  for (const { token, label } of checks) {
    if (token.test(body)) violations.push(label);
  }
  return { ok: violations.length === 0, violations };
}

export function collectVarNames(node: SpearNode): string[] {
  const out: string[] = [];
  const walk = (nd: SpearNode) => {
    if (nd.op === "var") out.push(nd.name);
    nd.children.forEach(walk);
  };
  walk(node);
  return out;
}

export function countOps(node: SpearNode): { total: number; transcendental: number } {
  let total = 0;
  let transcendental = 0;
  const walk = (nd: SpearNode) => {
    if (nd.op === "var" || nd.op === "const") return;
    total++;
    if (!ALGEBRAIC_OPS.has(nd.op)) transcendental++;
    nd.children.forEach(walk);
  };
  walk(node);
  return { total, transcendental };
}

// ------------------------------------------------------- canonical cache key
export function canonicalKey(node: SpearNode): string {
  const walk = (nd: SpearNode): string => {
    if (nd.op === "var") return `v:${nd.name}`;
    if (nd.op === "const") return `c:${roundConst(nd.value)}`;
    return `${nd.op}(${nd.children.map(walk).join(",")})`;
  };
  return walk(node);
}

// ---------------------------------------------------------------- simplify
function isConst(n: SpearNode, v?: number): boolean {
  return n.op === "const" && (v === undefined || roundConst(n.value) === roundConst(v));
}

export function simplify(node: SpearNode): SpearNode {
  const kids = node.children.map(simplify);
  // constant folding
  if (node.op !== "const" && node.op !== "var" && kids.every((k) => k.op === "const")) {
    const scope: Record<string, number> = {};
    const folded = evaluateScalar(makeNode(node.op, { value: node.value, children: kids }), scope);
    if (Number.isFinite(folded)) return makeNode("const", { value: roundConst(folded) });
  }
  switch (node.op) {
    case "add":
      if (isConst(kids[0], 0)) return kids[1];
      if (isConst(kids[1], 0)) return kids[0];
      if (kids[0].op === "const" && kids[1].op === "const") return makeNode("const", { value: roundConst(kids[0].value + kids[1].value) });
      // collapse nested constants: (x + a) + b -> x + (a+b) and a + (x + b) -> x + (a+b)
      if (kids[0].op === "add" && kids[0].children[1].op === "const" && kids[1].op === "const") {
        return simplify(makeNode("add", { children: [kids[0].children[0], makeNode("const", { value: roundConst(kids[0].children[1].value + kids[1].value) })] }));
      }
      if (kids[1].op === "add" && kids[1].children[1].op === "const" && kids[0].op === "const") {
        return simplify(makeNode("add", { children: [kids[1].children[0], makeNode("const", { value: roundConst(kids[0].value + kids[1].children[1].value) })] }));
      }
      break;
    case "sub":
      if (isConst(kids[1], 0)) return kids[0];
      if (isConst(kids[0], 0)) return simplify(makeNode("neg", { children: [kids[1]] }));
      // (x + a) - a -> x
      if (kids[0].op === "add" && kids[1].op === "const" && kids[0].children[1].op === "const" &&
          roundConst(kids[0].children[1].value) === roundConst(kids[1].value)) {
        return kids[0].children[0];
      }
      break;
    case "mul":
      if (isConst(kids[0], 0) || isConst(kids[1], 0)) return makeNode("const", { value: 0 });
      if (isConst(kids[0], 1)) return kids[1];
      if (isConst(kids[1], 1)) return kids[0];
      if (isConst(kids[0], -1)) return simplify(makeNode("neg", { children: [kids[1]] }));
      if (isConst(kids[1], -1)) return simplify(makeNode("neg", { children: [kids[0]] }));
      if (kids[0].op === "sq" && kids[0].children[0] === kids[1]) return makeNode("cube", { children: [kids[1]] });
      // collapse nested constants: c1 * (c2 * x) -> (c1*c2) * x, kills degenerate
      // large-coefficient forms like 563.38 * (0.0018 * u) that evaluate to ~1*u
      if (kids[0].op === "mul" && kids[0].children[0].op === "const") {
        return simplify(makeNode("mul", { children: [makeNode("const", { value: roundConst(kids[0].children[0].value * kids[1].value) }), kids[0].children[1]] }));
      }
      if (kids[1].op === "mul" && kids[1].children[0].op === "const") {
        return simplify(makeNode("mul", { children: [makeNode("const", { value: roundConst(kids[0].value * kids[1].children[0].value) }), kids[1].children[1]] }));
      }
      break;
    case "pdiv":
      if (isConst(kids[1], 1)) return kids[0];
      if (isConst(kids[0], 0)) return makeNode("const", { value: 0 });
      // x / c -> x * (1/c): div costs 4 units, mul costs 1. ONLY legal when
      // |c| >= 1e-4 (the evaluator's protection floor): below it, pdiv clamps
      // both denominator and output, and discovered formulas may lean on those
      // rails as free saturation (gaussian_cdf champion does exactly that).
      // Result-clamp cases above the floor (huge numerator) are left alone
      // here and caught by the parity gate in scripts/optimize-ledger.ts.
      if (kids[1].op === "const" && Math.abs(kids[1].value) >= 1e-4) {
        return simplify(makeNode("mul", { children: [kids[0], makeNode("const", { value: 1 / kids[1].value })] }));
      }
      break;
    case "neg":
      if (kids[0].op === "neg") return kids[0].children[0];
      if (kids[0].op === "const") return makeNode("const", { value: roundConst(-kids[0].value) });
      break;
    case "abs":
      if (kids[0].op === "abs" || kids[0].op === "sq") return kids[0];
      break;
    case "relu":
      if (kids[0].op === "relu") return kids[0];
      break;
    case "sq":
      if (kids[0].op === "abs") return makeNode("sq", { children: [kids[0].children[0]] });
      if (kids[0].op === "sqrt") return kids[0].children[0];
      break;
    case "max":
    case "min":
      if (canonicalKey(kids[0]) === canonicalKey(kids[1])) return kids[0];
      break;
    default: break;
  }
  if (
    (node.op === "add" || node.op === "mul" || node.op === "max" || node.op === "min") &&
    canonicalKey(kids[0]) === canonicalKey(kids[1])
  ) {
    if (node.op === "add") return makeNode("mul", { children: [makeNode("const", { value: 2 }), kids[0]] });
    if (node.op === "mul") return makeNode("sq", { children: [kids[0]] });
    return kids[0];
  }
  return makeNode(node.op, { value: node.value, name: node.name, children: kids });
}

// ------------------------------------------------------------ affine wrapper
export function wrapAffine(node: SpearNode, a: number, b: number): SpearNode {
  return simplify(
    makeNode("add", {
      children: [makeNode("mul", { children: [makeNode("const", { value: a }), node] }), makeNode("const", { value: b })],
    }),
  );
}

/** Least-squares fit of aÂ·p + b onto target (Keijzer linear scaling). */
export function fitLinearScaling(pred: Float64Array, target: Float64Array): { a: number; b: number } {
  const n = pred.length;
  let sp = 0, st = 0, spp = 0, spt = 0;
  for (let i = 0; i < n; i++) { sp += pred[i]; st += target[i]; spp += pred[i] * pred[i]; spt += pred[i] * target[i]; }
  const denom = n * spp - sp * sp;
  if (Math.abs(denom) < 1e-12) return { a: 0, b: st / n };
  const a = (n * spt - sp * st) / denom;
  const b = (st - a * sp) / n;
  return { a, b };
}

export function countConstants(node: SpearNode): number {
  let c = node.op === "const" ? 1 : 0;
  for (const k of node.children) c += countConstants(k);
  return c;
}

/** Coordinate-descent refinement of every constant in the tree. */
export function refineConstants(
  node: SpearNode,
  score: (candidate: SpearNode) => number,
  budget = 40,
): { node: SpearNode; score: number; evals: number } {
  let best = cloneNode(node);
  let bestScore = score(best);
  let evals = 1;
  const positions: SpearNode[] = [];
  const collect = (nd: SpearNode) => {
    if (nd.op === "const") positions.push(nd);
    nd.children.forEach(collect);
  };
  collect(best);
  if (positions.length === 0 || !Number.isFinite(bestScore)) {
    return { node: best, score: bestScore, evals };
  }

  const writeAt = (idx: number, value: number) => { positions[idx].value = roundConst(value); };

  for (let round = 0; round < 6 && evals < budget; round++) {
    let improved = false;
    for (let i = 0; i < positions.length && evals < budget; i++) {
      const original = positions[i].value;
      const scale = Math.max(0.05, Math.abs(original) * 0.5);
      const steps = [scale, scale * 0.3, scale * 0.08, 0.02, 0.005, 0.001];
      for (const step of steps) {
        if (evals >= budget) break;
        for (const dir of [1, -1]) {
          if (evals >= budget) break;
          writeAt(i, original + dir * step);
          const s = score(best);
          evals++;
          if (Number.isFinite(s) && s < bestScore - 1e-12) {
            bestScore = s;
            improved = true;
            break;
          }
          writeAt(i, original);
        }
        if (improved) break;
      }
    }
    if (!improved) break;
  }
  // final cleanup: fold trivial constants
  const cleaned = simplify(best);
  const cleanedScore = score(cleaned);
  evals++;
  return {
    node: Number.isFinite(cleanedScore) && cleanedScore <= bestScore ? cleaned : best,
    score: Math.min(cleanedScore, bestScore),
    evals,
  };
}

// ---------------------------------------------------------------- GP config
export interface GpConfig {
  variables: string[];
  constRange: [number, number];
  ops: NodeOp[];
  maxDepth: number;
  terminalVarProb?: number;
}

export function randomTree(cfg: GpConfig, maxDepth: number): SpearNode {
  const termProb = cfg.terminalVarProb ?? 0.65;
  if (maxDepth <= 1 || (maxDepth < cfg.maxDepth && rand() < 0.25)) {
    if (cfg.variables.length > 0 && rand() < termProb) {
      return makeNode("var", { name: pick(cfg.variables) });
    }
    return makeNode("const", { value: roundConst(randRange(cfg.constRange[0], cfg.constRange[1])) });
  }
  const op = pick(cfg.ops);
  const ar = arity(op);
  const children = Array.from({ length: ar }, () => randomTree(cfg, maxDepth - 1));
  return makeNode(op, { children });
}

function collectNodes(node: SpearNode): SpearNode[] {
  const res: SpearNode[] = [node];
  for (const c of node.children) res.push(...collectNodes(c));
  return res;
}

function replaceNode(root: SpearNode, target: SpearNode, replacement: SpearNode): SpearNode {
  if (root === target) return replacement;
  if (root.children.length === 0) return root;
  return makeNode(root.op, {
    value: root.value,
    name: root.name,
    children: root.children.map((c) => replaceNode(c, target, replacement)),
  });
}

export function crossover(p1: SpearNode, p2: SpearNode, maxDepth: number): SpearNode {
  const n1 = collectNodes(p1);
  const n2 = collectNodes(p2);
  const child = replaceNode(p1, pick(n1), pick(n2));
  return child.depth <= maxDepth ? child : p1;
}

export function mutate(p: SpearNode, cfg: GpConfig): SpearNode {
  const nodes = collectNodes(p);
  const target = pick(nodes);
  let replacement: SpearNode;
  if (target.op === "const") {
    replacement = makeNode("const", { value: roundConst(target.value + (rand() * 2 - 1) * Math.max(0.05, Math.abs(target.value) * 0.4)) });
  } else if (target.op === "var") {
    if (cfg.variables.length > 1 && rand() < 0.5) {
      const others = cfg.variables.filter((v) => v !== target.name);
      replacement = makeNode("var", { name: pick(others) });
    } else {
      replacement = randomTree(cfg, 2);
    }
  } else {
    const ar = arity(target.op);
    const same = cfg.ops.filter((o) => arity(o) === ar);
    if (same.length > 0 && rand() < 0.6) {
      replacement = makeNode(pick(same), { children: target.children });
    } else {
      replacement = randomTree(cfg, Math.max(1, cfg.maxDepth - 1));
    }
  }
  const mutated = simplify(replaceNode(p, target, replacement));
  return mutated.depth <= cfg.maxDepth ? mutated : p;
}

/** Fine-tune a single constant by a small delta. Preserves structure. */
export function mutatePolish(p: SpearNode, step = 0.08): SpearNode {
  const consts: SpearNode[] = [];
  const collect = (nd: SpearNode) => {
    if (nd.op === "const") consts.push(nd);
    nd.children.forEach(collect);
  };
  collect(p);
  if (consts.length === 0) return p;
  const target = pick(consts);
  const delta = (rand() < 0.5 ? -1 : 1) * step * (1 + rand());
  return simplify(replaceNode(p, target, makeNode("const", { value: roundConst(target.value + delta) })));
}

/** Swap a constant with a small algebraic expression, or vice versa. */
export function mutateStructure(p: SpearNode, cfg: GpConfig): SpearNode {
  const nodes = collectNodes(p);
  const target = pick(nodes);
  if (target.op === "const") {
    // promote constant â†’ tiny expression
    const tiny = pick(cfg.variables);
    const rep = rand() < 0.5
      ? makeNode("mul", { children: [makeNode("const", { value: roundConst(target.value) }), makeNode("var", { name: tiny })] })
      : makeNode("add", { children: [makeNode("var", { name: tiny }), makeNode("const", { value: roundConst(target.value) })] });
    const m = simplify(replaceNode(p, target, rep));
    return m.depth <= cfg.maxDepth ? m : p;
  }
  if (target.op !== "var" && target.children.length > 0 && target.children.every((c) => c.op === "const" || c.op === "var")) {
    // demote a small subexpression â†’ constant equal to its evaluation at 0
    const scope: Record<string, number> = {};
    const folded = evaluateScalar(target, scope);
    if (Number.isFinite(folded)) {
      const m = simplify(replaceNode(p, target, makeNode("const", { value: roundConst(folded) })));
      return m.depth <= cfg.maxDepth ? m : p;
    }
  }
  return p;
}

// ---------------------------------------------------------------- NSGA-II
export interface ObjectiveVector {
  /** primary objective, higher is better after internal normalisation */
  fitness: number;
  /** parsimony: smaller is better */
  size: number;
}

export function dominates(a: ObjectiveVector, b: ObjectiveVector): boolean {
  const aFit = Number.isFinite(a.fitness);
  const bFit = Number.isFinite(b.fitness);
  if (!aFit) return false;
  if (!bFit) return true;
  return (a.fitness >= b.fitness && a.size <= b.size) && (a.fitness > b.fitness || a.size < b.size);
}

export function nonDominatedSort(pop: ObjectiveVector[]): number[] {
  const n = pop.length;
  const domCount = new Array(n).fill(0);
  const ranks = new Array(n).fill(0);
  const fronts: number[][] = [[]];
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      if (p === q) continue;
      if (dominates(pop[p], pop[q])) domCount[q]++;
    }
  }
  for (let i = 0; i < n; i++) if (domCount[i] === 0) { ranks[i] = 0; fronts[0].push(i); }
  let f = 0;
  while (fronts[f] && fronts[f].length > 0) {
    const next: number[] = [];
    for (const i of fronts[f]) {
      for (let q = 0; q < n; q++) {
        if (dominates(pop[i], pop[q])) {
          domCount[q]--;
          if (domCount[q] === 0) { ranks[q] = f + 1; next.push(q); }
        }
      }
    }
    f++;
    fronts[f] = next;
  }
  return ranks;
}

export function crowdingDistance(front: ObjectiveVector[]): number[] {
  const n = front.length;
  const dist = new Array(n).fill(0);
  if (n <= 2) return front.map(() => Infinity);
  for (const key of ["fitness", "size"] as const) {
    const order = front.map((v, i) => ({ i, v: v[key] })).sort((a, b) => a.v - b.v);
    dist[order[0].i] = Infinity;
    dist[order[n - 1].i] = Infinity;
    const span = order[n - 1].v - order[0].v || 1;
    for (let k = 1; k < n - 1; k++) dist[order[k].i] += (order[k + 1].v - order[k - 1].v) / span;
  }
  return dist;
}

export function tournamentSelect(pop: ObjectiveVector[]): number {
  const a = randInt(pop.length);
  const b = randInt(pop.length);
  const rA = pop[a];
  const rB = pop[b];
  if (rA.fitness !== rB.fitness) {
    if (!Number.isFinite(rB.fitness)) return a;
    if (!Number.isFinite(rA.fitness)) return b;
    return rA.fitness > rB.fitness ? a : b;
  }
  if (rA.size !== rB.size) return rA.size < rB.size ? a : b;
  return rand() < 0.5 ? a : b;
}

// ---------------------------------------------------------------- evolution
export interface FitnessResult extends ObjectiveVector {
  extra?: number;
}

export type FitnessFn = (node: SpearNode) => FitnessResult;

export interface EvolveConfig extends GpConfig {
  populationSize: number;
  generations: number;
  crossoverRate?: number;
  mutationRate?: number;
}

export interface GenerationRecord {
  generation: number;
  bestFitness: number;
  bestSize: number;
  bestExtra?: number;
}

export interface EvolveResult {
  best: SpearNode;
  bestFitness: number;
  bestExtra?: number;
  front: SpearNode[];
  history: GenerationRecord[];
  durationMs: number;
  evals: number;
}

export function evolve(cfg: EvolveConfig, fitnessFn: FitnessFn): EvolveResult {
  const t0 = Date.now();
  let population = Array.from({ length: cfg.populationSize }, () => simplify(randomTree(cfg, cfg.maxDepth)));
  const archive = new Map<string, { node: SpearNode; obj: FitnessResult }>();
  let best: SpearNode = population[0];
  let bestFitness = -Infinity;
  let bestExtra: number | undefined;
  const history: GenerationRecord[] = [];
  let evals = 0;
  const crossoverRate = cfg.crossoverRate ?? 0.8;
  const mutationRate = cfg.mutationRate ?? 0.35;

  for (let gen = 0; gen < cfg.generations; gen++) {
    const results = population.map((ind) => {
      evals++;
      return fitnessFn(ind);
    });
    const objs = results.map((r) => ({ fitness: r.fitness, size: r.size }));

    for (let i = 0; i < population.length; i++) {
      const key = canonicalKey(population[i]);
      const prev = archive.get(key);
      if (!prev || results[i].fitness > prev.obj.fitness) {
        archive.set(key, { node: population[i], obj: results[i] });
      }
    }

    let genBest = 0;
    for (let i = 1; i < population.length; i++) if (results[i].fitness > results[genBest].fitness) genBest = i;
    if (results[genBest].fitness > bestFitness) {
      bestFitness = results[genBest].fitness;
      best = population[genBest];
      bestExtra = results[genBest].extra;
    }
    history.push({
      generation: gen,
      bestFitness: Number.isFinite(bestFitness) ? bestFitness : -1e9,
      bestSize: best.size,
      bestExtra,
    });

    const ranks = nonDominatedSort(objs);
    const maxRank = Math.max(...ranks);
    const byFront: number[][] = Array.from({ length: maxRank + 1 }, () => []);
    for (let i = 0; i < population.length; i++) byFront[ranks[i]].push(i);

    const selected: number[] = [];
    for (const front of byFront) {
      if (front.length === 0) continue;
      if (selected.length + front.length <= cfg.populationSize) {
        selected.push(...front);
        continue;
      }
      const frontObjs = front.map((i) => objs[i]);
      const dist = crowdingDistance(frontObjs);
      const order = front.map((i, k) => ({ i, d: dist[k] })).sort((a, b) => b.d - a.d);
      for (const { i } of order) {
        if (selected.length >= cfg.populationSize) break;
        selected.push(i);
      }
      break;
    }

    const newPop: SpearNode[] = [best];
    while (newPop.length < cfg.populationSize) {
      const p1 = population[tournamentSelect(objs)];
      let child: SpearNode;
      if (rand() < crossoverRate) {
        const p2 = population[tournamentSelect(objs)];
        child = crossover(p1, p2, cfg.maxDepth);
      } else {
        child = p1;
      }
      if (rand() < mutationRate) child = mutate(child, cfg);
      child = simplify(child);
      if (child.depth <= cfg.maxDepth) newPop.push(child);
    }
    population = newPop;
  }

  const frontObjs = population.map((p) => fitnessFn(p));
  const ranksFinal = nonDominatedSort(frontObjs.map((r) => ({ fitness: r.fitness, size: r.size })));
  const frontIdx = population.filter((_, i) => ranksFinal[i] === 0);
  evals += population.length;

  const archiveFront = [...archive.values()]
    .filter((e) => Number.isFinite(e.obj.fitness))
    .sort((a, b) => b.obj.fitness - a.obj.fitness)
    .slice(0, 8)
    .map((e) => e.node);

  return {
    best,
    bestFitness,
    bestExtra,
    front: frontIdx.length > 0 ? frontIdx : archiveFront,
    history,
    durationMs: Date.now() - t0,
    evals,
  };
}
