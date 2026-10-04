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
    let lit = |v: f64| format!("{:.*}{}", fl.prec(), v, fl.suffix());
    format!(
        r##"// ── rails SPEAR (fidèles à src/lib/spear/engine.ts) ─────────────────────
#[inline(always)]
#[allow(dead_code)]
fn spear_pdiv(a: {ty}, b: {ty}) -> {ty} {{
    let d = if b > {eps4} && b < {neps4} {{ if b >= {zero} {{ {eps4} }} else {{ {neps4} }} }} else {{ b }};
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
