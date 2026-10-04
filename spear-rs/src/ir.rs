//! IR SPEAR : opps, coûts ALU/SFU, désérialisation du format ledger.
//!
//! Les noms de variants sont exactement les opps JSON de `engine.ts` — d'où
//! `rename_all = "lowercase"`, qui rend le désérialiseur direct.

use serde::Deserialize;

/// Les 22 opps du moteur TypeScript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    Var,
    Const,
    Add,
    Sub,
    Mul,
    Pdiv,
    Relu,
    Abs,
    Neg,
    Sq,
    Sqrt,
    Cube,
    Max,
    Min,
    Exp,
    Sin,
    Cos,
    Log,
    Atan,
    Asin,
    Tanh,
    Acos,
    Erf,
}

/// Opps servies par l'unité SFU (coût 20 par défaut, 1 en mode GPU).
pub const TRANSCENDENTAL: [Op; 9] = [
    Op::Exp, Op::Sin, Op::Cos, Op::Log, Op::Atan, Op::Asin, Op::Tanh, Op::Acos, Op::Erf,
];

/// Un nœud sérialisé tel que stocké dans le ledger / les lois exportées.
#[derive(Debug, Clone, Deserialize)]
pub struct Node {
    #[serde(rename = "o")]
    pub op: Op,
    #[serde(rename = "v", default)]
    pub value: f64,
    #[serde(rename = "n", default)]
    pub name: String,
    #[serde(rename = "c", default)]
    pub children: Vec<Node>,
}

impl Node {
    pub fn var(name: &str) -> Self {
        Node { op: Op::Var, value: 0.0, name: name.to_string(), children: Vec::new() }
    }

    pub fn constant(value: f64) -> Self {
        Node { op: Op::Const, value, name: String::new(), children: Vec::new() }
    }

    /// Construit un nœud à partir d'op et d'enfants.
    pub fn new(op: Op, children: Vec<Node>) -> Self {
        Node { op, value: 0.0, name: String::new(), children }
    }
}

/// Poids d'un op en unités ALU/SFU — identique à `OP_COST` de `engine.ts`.
/// Ce ne sont PAS des FLOPs : c'est le modèle de sélection du Pareto.
pub fn op_cost(op: Op) -> i64 {
    use Op::*;
    match op {
        Var | Const => 0,
        Add | Sub | Mul | Relu | Abs | Neg | Sq | Max | Min => 1,
        Pdiv => 4,
        Cube | Sqrt => 2,
        Exp | Sin | Cos | Log | Atan | Asin | Tanh | Acos | Erf => 20,
    }
}

pub fn is_transcendental(op: Op) -> bool {
    TRANSCENDENTAL.contains(&op)
}

pub fn arity(op: Op) -> usize {
    use Op::*;
    match op {
        Var | Const => 0,
        Add | Sub | Mul | Pdiv | Max | Min => 2,
        _ => 1,
    }
}

/// Coût total d'un arbre, `estimateCost` de `engine.ts`.
pub fn cost(node: &Node) -> i64 {
    op_cost(node.op) + node.children.iter().map(cost).sum::<i64>()
}

/// Variables utilisées, sans doublon, dans l'ordre de première apparition
/// (même ordre que `collectVarNames` côté TS).
pub fn var_names(node: &Node) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    collect_vars(node, &mut out);
    out
}

fn collect_vars(node: &Node, out: &mut Vec<String>) {
    if node.op == Op::Var && !out.contains(&node.name) {
        out.push(node.name.clone());
    }
    for c in &node.children {
        collect_vars(c, out);
    }
}

/// Histogramme des opps (trié par fréquence décroissante) — pour `spear show`.
pub fn op_histogram(node: &Node) -> Vec<(Op, usize)> {
    let mut counts: Vec<(Op, usize)> = Vec::new();
    fn walk(n: &Node, acc: &mut Vec<(Op, usize)>) {
        match acc.iter_mut().find(|(op, _)| *op == n.op) {
            Some((_, c)) => *c += 1,
            None => acc.push((n.op, 1)),
        }
        for c in &n.children {
            walk(c, acc);
        }
    }
    walk(node, &mut counts);
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(format!("{:?}", a.0).cmp(&format!("{:?}", b.0))));
    counts
}
