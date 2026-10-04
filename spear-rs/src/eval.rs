//! Évaluation fidèle aux rails de `engine.ts`.
//!
//! Deux chemins, volontairement identiques sémantiquement :
//! * [`eval_tree`] — récursion directe sur l'arbre (référence, lisible) ;
//! * [`Program`] — compilation postfix + VM, scalaire (`eval`) et vectorielle
//!   (`run_into`) avec un scratch réutilisé, donc zéro allocation par élément.
//!
//! Les bornages ne sont pas décoratifs : les formules découvertes s'appuient
//! dessus (le champion de `gaussian_cdf` exploite le plancher de `pdiv`). Les
//! changer changerait les métriques.

use crate::ir::{arity, Node, Op};

/// Abramowitz & Stegun 7.1.26 — **mêmes constantes et même ordre** que
/// `erf` de `math-utils.ts`, donc même résultat que le moteur TS.
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = ((((1.061405429 * t + -1.453152027) * t + 1.421413741) * t + -0.284496736) * t
        + 0.254829592)
        * t;
    sign * (1.0 - poly * (-ax * ax).exp())
}

/// Division protégée : dénominateur flooré à ±1e-4 (signe conservé), résultat
/// borné à ±1e4.
#[inline]
pub fn pdiv(a: f64, b: f64) -> f64 {
    let d = if b > -1e-4 && b < 1e-4 {
        if b >= 0.0 {
            1e-4
        } else {
            -1e-4
        }
    } else {
        b
    };
    (a / d).clamp(-1e4, 1e4)
}

/// Application d'un op unaire/binaire. `b` est ignoré pour les unaires.
#[inline]
pub fn apply(op: Op, a: f64, b: f64) -> f64 {
    use Op::*;
    match op {
        Add => a + b,
        Sub => a - b,
        Mul => a * b,
        Pdiv => pdiv(a, b),
        Max => {
            if a > b {
                a
            } else {
                b
            }
        }
        Min => {
            if a < b {
                a
            } else {
                b
            }
        }
        Relu => {
            if a > 0.0 {
                a
            } else {
                0.0
            }
        }
        Abs => a.abs(),
        Neg => -a,
        Sq => a * a,
        Cube => a * a * a,
        Sqrt => a.abs().sqrt(),
        Exp => a.clamp(-50.0, 50.0).exp(),
        Sin => a.sin(),
        Cos => a.cos(),
        Log => a.max(1e-30).ln(),
        Atan => a.atan(),
        Asin => a.clamp(-1.0, 1.0).asin(),
        Tanh => a.tanh(),
        Acos => a.clamp(-1.0, 1.0).acos(),
        Erf => erf(a),
        Var | Const => f64::NAN,
    }
}

/// Évaluation récursive — sert de référence au test de parité VM.
pub fn eval_tree(node: &Node, get: &dyn Fn(&str) -> f64) -> f64 {
    match node.op {
        Op::Var => get(&node.name),
        Op::Const => node.value,
        op => {
            let mut args = [0.0f64; 2];
            for (k, c) in node.children.iter().enumerate().take(2) {
                args[k] = eval_tree(c, get);
            }
            apply(op, args[0], args[1])
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Instr {
    Push(f64),
    Load(usize),
    Op(Op),
}

/// Un kernel compilé : postfix + table de variables.
#[derive(Debug, Clone)]
pub struct Program {
    code: Vec<Instr>,
    /// Variables dans l'ordre de première apparition ; l'index est le slot.
    pub vars: Vec<String>,
    stack: usize,
}

impl Program {
    pub fn compile(node: &Node) -> Self {
        let mut p = Program { code: Vec::new(), vars: Vec::new(), stack: 0 };
        let mut depth = 0usize;
        p.emit(node, &mut depth);
        p
    }

    fn emit(&mut self, n: &Node, depth: &mut usize) {
        for c in &n.children {
            self.emit(c, depth);
        }
        match n.op {
            Op::Const => self.code.push(Instr::Push(n.value)),
            Op::Var => {
                let slot = match self.vars.iter().position(|v| v == &n.name) {
                    Some(i) => i,
                    None => {
                        self.vars.push(n.name.clone());
                        self.vars.len() - 1
                    }
                };
                self.code.push(Instr::Load(slot));
            }
            op => self.code.push(Instr::Op(op)),
        }
        *depth += 1;
        if *depth > self.stack {
            self.stack = *depth;
        }
    }

    /// Nombre d'instructions.
    pub fn len(&self) -> usize {
        self.code.len()
    }

    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    /// Profondeur de pile max — borne la taille du scratch.
    pub fn stack_depth(&self) -> usize {
        self.stack
    }

    /// Slot d'une variable, pour associer `--set nom=valeur` aux entrées.
    pub fn slot_of(&self, name: &str) -> Option<usize> {
        self.vars.iter().position(|v| v == name)
    }

    /// Évaluation scalaire. `scope` est indexé par slot.
    #[inline]
    pub fn eval(&self, scope: &[f64]) -> f64 {
        debug_assert!(self.stack <= 64, "arbre plus profond que le scratch scalaire");
        let mut st = [0.0f64; 64];
        let mut top = 0usize;
        for ins in &self.code {
            match *ins {
                Instr::Push(v) => {
                    st[top] = v;
                    top += 1;
                }
                Instr::Load(i) => {
                    st[top] = scope.get(i).copied().unwrap_or(0.0);
                    top += 1;
                }
                Instr::Op(op) => {
                    let ar = arity(op);
                    let a = st[top - ar];
                    let b = if ar == 2 { st[top - 1] } else { a };
                    top -= ar;
                    st[top] = apply(op, a, b);
                    top += 1;
                }
            }
        }
        st[0]
    }

    /// Évaluation vectorielle : `inputs[slot]` doit avoir `out.len()` éléments.
    pub fn run_into(&self, inputs: &[&[f64]], out: &mut [f64]) -> Result<(), String> {
        let n = out.len();
        if inputs.len() != self.vars.len() {
            return Err(format!(
                "kernel attend {} variable(s) [{}], reçu {}",
                self.vars.len(),
                self.vars.join(", "),
                inputs.len()
            ));
        }
        for (i, slice) in inputs.iter().enumerate() {
            if slice.len() != n {
                return Err(format!(
                    "variable #{} ({}) : {} échantillons, attendu {}",
                    i,
                    self.vars.get(i).cloned().unwrap_or_else(|| "?".into()),
                    slice.len(),
                    n
                ));
            }
        }
        let mut st = vec![0.0f64; self.stack.max(1)];
        for row in 0..n {
            let mut top = 0usize;
            for ins in &self.code {
                match *ins {
                    Instr::Push(v) => {
                        st[top] = v;
                        top += 1;
                    }
                    Instr::Load(i) => {
                        st[top] = inputs[i][row];
                        top += 1;
                    }
                    Instr::Op(op) => {
                        let ar = arity(op);
                        let a = st[top - ar];
                        let b = if ar == 2 { st[top - 1] } else { a };
                        top -= ar;
                        st[top] = apply(op, a, b);
                        top += 1;
                    }
                }
            }
            out[row] = st[0];
        }
        Ok(())
    }

    /// Variante pratique : entrées en slices indexées par slot.
    pub fn run_vec(&self, inputs: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        let n = inputs.first().map(|v| v.len()).unwrap_or(0);
        let mut out = vec![0.0f64; n];
        let borrowed: Vec<&[f64]> = inputs.iter().map(|v| v.as_slice()).collect();
        self.run_into(&borrowed, &mut out)?;
        Ok(out)
    }
}
