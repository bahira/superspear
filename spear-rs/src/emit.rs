//! Émission de kernels Rust drop-in à partir d'un arbre SPEAR.
//!
//! Le code produit est **branchless-compatible** et sans allocation : chaque
//! bornage du moteur TS devient un petit helper `#[inline(always)]`, donc
//! l'expression émise reste une expression Rust valide (Rust n'a pas
//! d'expression-statement comme GCC).
//!
//! Deux sorties par kernel :
//! * `spear_<id>(…) -> f64` — une valeur ;
//! * `spear_<id>_into(&[f64], …, &mut [f64])` — passe fusionnée, une seule
//!   écriture par élément. C'est celle-là qui remplace utilement un graphe
//!   `evaluateNode` qui alloue un buffer par nœud.
//!
//! `--f32` bascule tout le kernel (helpers compris) en f32 : c'est le format
//! des backends GPU/Metal, donc le format à viser pour TensorFold/MLX.

use crate::ir::{var_names, Node, Op};

/// Type de flottant du kernel émis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Float {
    F32,
    F64,
}

impl Float {
    fn ty(self) -> &'static str {
        match self {
            Float::F32 => "f32",
            Float::F64 => "f64",
        }
    }

    fn suffix(self) -> &'static str {
        self.ty()
    }

    /// Précision des littéraux : 9 décimales suffisent en f32, on garde la
    /// précision f64 sinon (arrondir une constante f64 décalerait le kernel).
    fn prec(self) -> usize {
        match self {
            Float::F32 => 9,
            Float::F64 => 17,
        }
    }
}

/// Bornages SPEAR, copiés de `evaluateNode`/`erf` du moteur TS — émis dans le
/// même type de flottant que le kernel, sinon un kernel f32 qui appelle
/// `spear_pdiv` ne compilerait pas.
pub fn preamble(fl: Float) -> String {
    let ty = fl.ty();
    let lit = |v: f64| fmt_const(v, fl);
    format!(
        r##"// ── rails SPEAR (fidèles à src/lib/spear/engine.ts) ─────────────────────
#[inline(always)]
#[allow(dead_code)]
fn spear_pdiv(a: {ty}, b: {ty}) -> {ty} {{
    let d = if b > {neps4} && b < {eps4} {{ if b >= {zero} {{ {eps4} }} else {{ {neps4} }} }} else {{ b }};
    (a / d).clamp({n1e4}, {p1e4})
}}
#[inline(always)]
#[allow(dead_code)]
fn spear_exp(x: {ty}) -> {ty} {{ x.clamp({n50}, {p50}).exp() }}
#[inline(always)]
#[allow(dead_code)]
fn spear_log(x: {ty}) -> {ty} {{ x.max({e30}).ln() }}
#[inline(always)]
#[allow(dead_code)]
fn spear_sqrt(x: {ty}) -> {ty} {{ x.abs().sqrt() }}
#[inline(always)]
#[allow(dead_code)]
fn spear_asin(x: {ty}) -> {ty} {{ x.clamp({none}, {one}).asin() }}
#[inline(always)]
#[allow(dead_code)]
fn spear_acos(x: {ty}) -> {ty} {{ x.clamp({none}, {one}).acos() }}
// Abramowitz & Stegun 7.1.26 — identique à math-utils.ts
#[inline(always)]
#[allow(dead_code)]
fn spear_erf(x: {ty}) -> {ty} {{
    let sign = if x < {zero} {{ {none} }} else {{ {one} }};
    let ax = x.abs();
    let t = {one} / ({one} + {p} * ax);
    let poly = ((((({a5}) * t + ({a4})) * t + ({a3})) * t + ({a2})) * t + ({a1})) * t;
    sign * ({one} - poly * (-ax * ax).exp())
}}
"##,
        ty = ty,
        eps4 = lit(1e-4),
        neps4 = lit(-1e-4),
        zero = lit(0.0),
        one = lit(1.0),
        none = lit(-1.0),
        n1e4 = lit(-1e4),
        p1e4 = lit(1e4),
        n50 = lit(-50.0),
        p50 = lit(50.0),
        e30 = lit(1e-30),
        p = lit(0.3275911),
        a1 = lit(0.254829592),
        a2 = lit(-0.284496736),
        a3 = lit(1.421413741),
        a4 = lit(-1.453152027),
        a5 = lit(1.061405429),
    )
}

/// `swiglu_fused` → `spear_swiglu_fused`. Tout ce qui n'est pas alnum devient `_`.
pub fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn fmt_const(v: f64, fl: Float) -> String {
    if !v.is_finite() {
        // Une constante non finie ne peut pas venir d'un champion validé ; on
        // l'émet quand même, sans crasher la compilation du kernel.
        return format!("0.0{}", fl.suffix());
    }
    // La notation fixe à 9 décimales imprime 1e-30 comme 0.000000000 : le rail
    // `log` devenait `max(x, 0.0)` et ln(0) partait à -inf, en f64 comme en
    // f32 (17 décimales perdent tout ce qui est plus petit que ~1e-17). La
    // notation exponentielle ne perd rien, quelle que soit la magnitude.
    if v != 0.0 && (v.abs() < 1e-4 || v.abs() >= 1e7) {
        return format!("{:e}{}", v, fl.suffix());
    }
    format!("{:.*}{}", fl.prec(), v, fl.suffix())
}

fn walk(n: &Node, fl: Float) -> String {
    use Op::*;
    let a = |k: usize| walk(&n.children[k], fl);
    match n.op {
        Var => n.name.clone(),
        Const => fmt_const(n.value, fl),
        Add => format!("({} + {})", a(0), a(1)),
        Sub => format!("({} - {})", a(0), a(1)),
        Mul => format!("({} * {})", a(0), a(1)),
        Pdiv => format!("spear_pdiv({}, {})", a(0), a(1)),
        Max => format!("{}.max({})", a(0), a(1)),
        Min => format!("{}.min({})", a(0), a(1)),
        Relu => format!("{}.max(0.0{})", a(0), fl.suffix()),
        Abs => format!("{}.abs()", a(0)),
        Neg => format!("(-{})", a(0)),
        Sq => {
            let w = a(0);
            format!("({} * {})", w, w)
        }
        Cube => {
            let w = a(0);
            format!("({} * {} * {})", w, w, w)
        }
        Sqrt => format!("spear_sqrt({})", a(0)),
        Exp => format!("spear_exp({})", a(0)),
        Sin => format!("{}.sin()", a(0)),
        Cos => format!("{}.cos()", a(0)),
        Log => format!("spear_log({})", a(0)),
        Atan => format!("{}.atan()", a(0)),
        Asin => format!("spear_asin({})", a(0)),
        Tanh => format!("{}.tanh()", a(0)),
        Acos => format!("spear_acos({})", a(0)),
        Erf => format!("spear_erf({})", a(0)),
    }
}

/// Kernel complet en f64 : scalaire + variante fusionnée, avec les bornages.
pub fn emit_kernel(node: &Node, id: &str, formula: &str) -> String {
    emit_kernel_flavor(node, id, formula, Float::F64)
}

/// Idem dans le type de flottant demandé. La structure de l'arbre est
/// inchangée ; seules les constantes sont arrondies à la précision du type.
pub fn emit_kernel_flavor(node: &Node, id: &str, formula: &str, fl: Float) -> String {
    let ty = fl.ty();
    let vars = var_names(node);
    let fname = format!("spear_{}", sanitize_id(id));
    let params: Vec<String> = vars.iter().map(|v| format!("{}: {}", v, ty)).collect();
    let slice_params: Vec<String> = vars.iter().map(|v| format!("{}: &[{}]", v, ty)).collect();
    // un `*/` dans un commentaire Rust fermerait le commentaire
    let safe_formula = formula.replace("*/", "* /");
    let body_args: Vec<String> = vars.iter().map(|v| format!("{}[row]", v)).collect();
    let first_slice = vars
        .first()
        .map(|v| format!("{}.len()", v))
        .unwrap_or_else(|| "0".to_string());
    let flag = if fl == Float::F32 { " --f32" } else { "" };
    let prec_note = if fl == Float::F32 {
        "// constantes arrondies à 9 décimales (précision f32)\n"
    } else {
        ""
    };

    format!(
        "// {fname} — kernel SPEAR « {id} » [{ty}]\n\
         // formule : {safe_formula}\n\
         // généré par spear-rs : `spear emit {id}{flag}` (passe `--fast` pour le slot rapide)\n\
         // bornages identiques au moteur TS : voir spear_pdiv/spear_exp/spear_log/spear_erf\n\
         {prec_note}\n\
         {pre}\n\
         /// Champion SPEAR — `{safe_formula}`\n\
         #[inline(always)]\n\
         pub fn {fname}({}) -> {ty} {{\n    {}\n}}\n\
         \n\
         /// Passe fusionnée : une écriture par élément, zéro allocation par nœud.\n\
         #[inline]\n\
         pub fn {fname}_into({}out: &mut [{ty}]) {{\n    \
         debug_assert_eq!({}, out.len());\n    \
         for (row, o) in out.iter_mut().enumerate() {{\n        *o = {fname}({});\n    }}\n}}\n",
        params.join(", "),
        walk(node, fl),
        if slice_params.is_empty() { String::new() } else { format!("{}, ", slice_params.join(", ")) },
        first_slice,
        body_args.join(", "),
        pre = preamble(fl),
    )
}

// ─── Triton (Python, GPU NVIDIA) ────────────────────────────────────────────

/// Littéral Python : `{:?}` donne la forme la plus courte qui round-trip en
/// f64, donc rien ne se perd en passant par la source générée (1e-30 reste
/// 1e-30, une constante `log` ne devient pas 0).
fn fmt_py(v: f64) -> String {
    if !v.is_finite() {
        return "0.0".to_string();
    }
    format!("{:?}", v)
}

/// Rails SPEAR transcrits en Triton, mêmes bornages que le moteur TS et que
/// la VM Rust (voir `eval.rs`) : le rail `pdiv` garde la condition ORIGINALE
/// `b > -1e-4 && b < 1e-4`, pas la version inversée qui a vécu un moment dans
/// le préambule Rust émis.
const TRITON_RAILS: &str = r#"import triton
import triton.language as tl

try:  # triton >= 3.2
    from triton.language.extra import libdevice
except ImportError:  # triton plus ancien
    from triton.language.extra.cuda import libdevice


@triton.jit
def _spear_pdiv(a, b):
    d = tl.where((b > -1e-4) & (b < 1e-4), tl.where(b >= 0.0, 1e-4, -1e-4), b)
    return tl.minimum(tl.maximum(a / d, -1e4), 1e4)


@triton.jit
def _spear_exp(x):
    return tl.exp(tl.minimum(tl.maximum(x, -50.0), 50.0))


@triton.jit
def _spear_log(x):
    return tl.log(tl.maximum(x, 1e-30))


@triton.jit
def _spear_sqrt(x):
    return tl.sqrt(tl.abs(x))


@triton.jit
def _spear_asin(x):
    return libdevice.asin(tl.minimum(tl.maximum(x, -1.0), 1.0))


@triton.jit
def _spear_acos(x):
    return libdevice.acos(tl.minimum(tl.maximum(x, -1.0), 1.0))


@triton.jit
def _spear_erf(x):
    # Abramowitz & Stegun 7.1.26 — mêmes constantes que math-utils.ts
    sign = tl.where(x < 0.0, -1.0, 1.0)
    ax = tl.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = ((((1.061405429 * t + -1.453152027) * t + 1.421413741) * t + -0.284496736) * t + 0.254829592) * t
    return sign * (1.0 - poly * tl.exp(-ax * ax))

"#;

fn walk_triton(n: &Node) -> String {
    use Op::*;
    let a = |k: usize| walk_triton(&n.children[k]);
    match n.op {
        Var => n.name.clone(),
        Const => fmt_py(n.value),
        Add => format!("({} + {})", a(0), a(1)),
        Sub => format!("({} - {})", a(0), a(1)),
        Mul => format!("({} * {})", a(0), a(1)),
        Pdiv => format!("_spear_pdiv({}, {})", a(0), a(1)),
        Max => format!("tl.maximum({}, {})", a(0), a(1)),
        Min => format!("tl.minimum({}, {})", a(0), a(1)),
        Relu => format!("tl.maximum({}, 0.0)", a(0)),
        Abs => format!("tl.abs({})", a(0)),
        Neg => format!("(-{})", a(0)),
        Sq => {
            let w = a(0);
            format!("({w} * {w})")
        }
        Cube => {
            let w = a(0);
            format!("({w} * {w} * {w})")
        }
        Sqrt => format!("_spear_sqrt({})", a(0)),
        Exp => format!("_spear_exp({})", a(0)),
        Sin => format!("tl.sin({})", a(0)),
        Cos => format!("tl.cos({})", a(0)),
        Log => format!("_spear_log({})", a(0)),
        Atan => format!("libdevice.atan({})", a(0)),
        Asin => format!("_spear_asin({})", a(0)),
        Tanh => format!("libdevice.tanh({})", a(0)),
        Acos => format!("_spear_acos({})", a(0)),
        Erf => format!("_spear_erf({})", a(0)),
    }
}

/// Kernel Triton (Python) — même arbre, mêmes bornages que le moteur TS.
///
/// Deux objets sont émis : `spear_<id>`, la fonction élémentaire sur tenseurs
/// (utilisable dans n'importe quel `@triton.jit`), et `spear_<id>_kernel`, un
/// kernel de lancement complet (chargement masqué, calcul, stockage masqué).
/// Cible f32 — c'est le type par défaut des tenseurs Triton.
///
/// La sortie est du Python valide par construction ; la compilation/le run
/// GPU ne sont pas vérifiés ici (pas de triton dans l'environnement de test,
/// et pas de GPU) — le contrat testé est la génération, pas l'exécution.
pub fn emit_triton(node: &Node, id: &str, formula: &str) -> String {
    let vars = var_names(node);
    let fname = format!("spear_{}", sanitize_id(id));
    let safe_formula = formula.replace("*/", "* /");
    let scalar_params = vars.join(", ");
    let mut kernel_params: Vec<String> = vars.iter().map(|v| format!("{}_ptr", v)).collect();
    kernel_params.push("out_ptr".to_string());
    let loads: Vec<String> = vars
        .iter()
        .map(|v| format!("    {} = tl.load({}_ptr + offs, mask=m)", v, v))
        .collect();
    let call = format!("{}({})", fname, scalar_params);

    let mut out = String::new();
    out.push_str(&format!("# {fname} — kernel Triton SPEAR « {id} » [f32]\n"));
    out.push_str(&format!("# formule : {safe_formula}\n"));
    out.push_str(&format!(
        "# généré par spear-rs : `spear emit {id} --triton` (passe `--fast` pour le slot rapide)\n"
    ));
    out.push_str("# bornages identiques au moteur TS : voir _spear_pdiv/_spear_exp/_spear_log/_spear_erf\n");
    out.push_str(TRITON_RAILS);
    out.push_str(&format!(
        "@triton.jit\ndef {fname}({scalar_params}):\n    return {}\n\n\n",
        walk_triton(node)
    ));
    out.push_str(&format!(
        "@triton.jit\ndef {fname}_kernel({}, n, BLOCK: tl.constexpr):\n    \
         pid = tl.program_id(0)\n    \
         offs = pid * BLOCK + tl.arange(0, BLOCK)\n    \
         m = offs < n\n{}\n    \
         tl.store(out_ptr + offs, {call}, mask=m)\n",
        kernel_params.join(", "),
        loads.join("\n"),
    ));
    out
}
