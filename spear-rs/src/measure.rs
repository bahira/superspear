//! Mesure : grille de sonde, erreur contre la loi, timing ns/élément, audit.
//!
//! ## Méthodologie de timing
//!
//! * une passe = `rows` éléments × toutes les variables du kernel ;
//! * 1 passe de chauffe, puis `samples` passes mesurées, chacune répétée
//!   `repeat` fois ; on garde **la médiane** des passes, pas la moyenne ;
//! * le résultat est lu dans un `black_box` (checksum accumulé) pour que le
//!   compilateur ne supprime pas la boucle ;
//! * `ns/élément` = durée d'une passe / (rows × repeat).
//!
//! Ce n'est **pas** criterion : on veut un tableau sur 90 tâches × 2 variantes
//! en une commande, pas 180 rapports HTML. Les chiffres sont indicatifs et
//! comparables *entre kernels sur cette machine*, pas des performances
//! absolues (pas d'épinglage de cœur, pas de fréquence fixe).
//!
//! ## Deux métriques d'erreur qui ne se mélangent pas
//!
//! * `metric` du ledger : MSE sur la grille d'entraînement TypeScript ;
//! * `rmse`/`max_abs` d'ici : erreur du champion **contre la loi génératrice**
//!   sur la grille de sonde. Mesure différente, souvent plus grande, et c'est
//!   normal. Ne jamais les comparer comme un « drift ».

use std::time::Instant;

use crate::eval::Program;
use crate::ir::{cost, Node};
use crate::ledger::{Entry, LawTask};

/// Grille de sonde : une colonne par variable du kernel, + cible optionnelle.
pub struct Grid {
    pub names: Vec<String>,
    pub cols: Vec<Vec<f64>>,
    pub target: Option<Vec<f64>>,
}

impl Grid {
    pub fn rows(&self) -> usize {
        self.cols.first().map(|c| c.len()).unwrap_or(0)
    }

    pub fn inputs(&self) -> Vec<&[f64]> {
        self.cols.iter().map(|c| c.as_slice()).collect()
    }
}

/// Domaines par défaut, **heuristiques de timing uniquement**. Quand la tâche
/// déclare un `domain` (40 tâches sur 90), c'est lui qui gagne. Ces bornes
/// n'ont aucune valeur de vérité mathématique : elles servent à fournir des
/// entrées représentatives, et la borne utilisée est rappelée par `--domains`
/// dans le tableau de sortie.
pub fn default_span(name: &str) -> (f64, f64) {
    match name {
        "t" | "tau" => (0.0, 4.0),
        "th" => (-3.0, 3.0),
        "c" => (0.05, 5.0),
        "e" => (0.0, 0.9),
        "M" => (0.0, std::f64::consts::TAU),
        "k" => (0.0, 8.0),
        "n" => (1.0, 8.0),
        "r" => (0.0, 0.2),
        "p" => (0.01, 0.99),
        "v" => (0.0, 30.0),
        "l" => (0.5, 4.0),
        "m" => (1.0, 8.0),
        _ => (-4.0, 4.0),
    }
}

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![lo];
    }
    let step = (hi - lo) / (n - 1) as f64;
    (0..n).map(|i| lo + step * i as f64).collect()
}

/// Construit la grille pour les variables d'un kernel, dans l'ordre de ses slots.
pub fn build_grid(law: Option<&LawTask>, prog: &Program, rows: usize) -> Grid {
    let names = prog.vars.clone();
    let cols: Vec<Vec<f64>> = names
        .iter()
        .map(|name| {
            let span = match law {
                // domaine déclaré, uniquement pertinent pour les tâches 1-variable
                Some(t) if t.domain.is_some() && names.len() == 1 => {
                    let s = t.domain.as_ref().unwrap();
                    (s.lo, s.hi)
                }
                _ => default_span(name),
            };
            linspace(span.0, span.1, rows)
        })
        .collect();
    let target = law.and_then(|t| t.law.as_ref()).and_then(|l| eval_law(l, &names, &cols));
    Grid { names, cols, target }
}

fn eval_law(law: &Node, names: &[String], cols: &[Vec<f64>]) -> Option<Vec<f64>> {
    let prog = Program::compile(law);
    let mut inputs: Vec<Vec<f64>> = vec![Vec::new(); prog.vars.len()];
    for (slot, name) in prog.vars.iter().enumerate() {
        let col = cols
            .iter()
            .zip(names.iter())
            .find(|(_, n)| *n == name)
            .map(|(c, _)| c.clone());
        inputs[slot] = col?;
    }
    let out = prog.run_vec(&inputs).ok()?;
    if out.iter().all(|v| v.is_finite()) {
        Some(out)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ErrStat {
    pub rmse: f64,
    pub max_abs: f64,
}

/// RMSE et erreur max contre la cible sur la même grille.
pub fn errors(pred: &[f64], target: &[f64]) -> ErrStat {
    let mut sse = 0.0f64;
    let mut max_abs = 0.0f64;
    for (p, t) in pred.iter().zip(target) {
        let d = p - t;
        sse += d * d;
        max_abs = max_abs.max(d.abs());
    }
    let n = pred.len().max(1) as f64;
    ErrStat { rmse: (sse / n).sqrt(), max_abs }
}

pub fn median(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

/// ns par élément (médiane des passes).
pub fn time_ns_per_elem(prog: &Program, grid: &Grid, samples: usize, repeat: usize) -> f64 {
    let n = grid.rows();
    let inputs = grid.inputs();
    let mut out = vec![0.0f64; n];
    if prog.run_into(&inputs, &mut out).is_err() {
        return f64::NAN;
    }
    // chauffe
    let mut acc = 0.0f64;
    for _ in 0..repeat {
        let _ = prog.run_into(&inputs, &mut out);
        for v in &out {
            acc += *v;
        }
    }
    let per_pass = samples.max(1);
    let mut timings = Vec::with_capacity(per_pass);
    for _ in 0..per_pass {
        let t0 = Instant::now();
        for _ in 0..repeat.max(1) {
            let _ = prog.run_into(&inputs, &mut out);
            for v in &out {
                acc += *v;
            }
        }
        let dt = t0.elapsed();
        timings.push(dt.as_secs_f64() * 1e9 / (n as f64 * repeat.max(1) as f64));
    }
    std::hint::black_box(acc);
    median(timings)
}

// ------------------------------------------------------------------ lignes

#[derive(Debug, Clone)]
pub struct Row {
    pub id: String,
    pub vars: usize,
    pub champ_cost: i64,
    pub fast_cost: Option<i64>,
    pub champ_ns: f64,
    pub fast_ns: Option<f64>,
    pub law_ns: Option<f64>,
    pub champ_err: Option<ErrStat>,
    pub fast_err: Option<ErrStat>,
}

/// Benchmark complet d'une tâche : champion, slot rapide, et la loi si elle existe.
pub fn bench_task(
    id: &str,
    entry: &Entry,
    law: Option<&LawTask>,
    rows: usize,
    samples: usize,
    repeat: usize,
) -> Result<Row, String> {
    let tree = entry.tree.as_ref().ok_or_else(|| format!("{}: pas d'arbre champion", id))?;
    let prog = Program::compile(tree);
    let grid = build_grid(law, &prog, rows);
    // Une formule sans variable (constante pure) n'a rien à chronométrer.
    let champ_ns = if prog.vars.is_empty() { f64::NAN } else { time_ns_per_elem(&prog, &grid, samples, repeat) };
    let champ_pred = prog.run_vec(&grid.cols).unwrap_or_default();
    let champ_err = grid.target.as_ref().map(|t| errors(&champ_pred, t));

    let (fast_ns, fast_err, fast_cost) = match entry.fast_tree.as_ref() {
        Some(fast_tree) => {
            let fp = Program::compile(fast_tree);
            let fgrid = build_grid(law, &fp, rows);
            let ns = if fp.vars.is_empty() { f64::NAN } else { time_ns_per_elem(&fp, &fgrid, samples, repeat) };
            let pred = fp.run_vec(&fgrid.cols).unwrap_or_default();
            let err = fgrid.target.as_ref().map(|t| errors(&pred, t));
            (if ns.is_finite() { Some(ns) } else { None }, err, Some(cost(fast_tree)))
        }
        None => (None, None, None),
    };

    let law_ns = match law {
        Some(t) => t.law.as_ref().and_then(|l| {
            let lp = Program::compile(l);
            if lp.vars.is_empty() {
                return None;
            }
            let lgrid = build_grid(Some(t), &lp, rows);
            let v = time_ns_per_elem(&lp, &lgrid, samples, repeat);
            if v.is_finite() {
                Some(v)
            } else {
                None
            }
        }),
        None => None,
    };

    Ok(Row {
        id: id.to_string(),
        vars: prog.vars.len(),
        champ_cost: cost(tree),
        fast_cost,
        champ_ns,
        fast_ns,
        law_ns,
        champ_err,
        fast_err,
    })
}

// ------------------------------------------------------------------ audit

#[derive(Debug, Clone)]
pub struct Flag {
    /// `false` = avertissement, `true` = échec (comme les `✗` du script TS).
    pub fail: bool,
    pub msg: String,
}

/// Miroir Rust des contrôles de `scripts/audit-champions.ts` qui n'ont pas
/// besoin des données TS : recalcul du coût depuis l'arbre, cohérence du slot
/// rapide, variables déclarées, constantes finies.
pub fn audit_entry(id: &str, entry: &Entry, law: Option<&LawTask>) -> Vec<Flag> {
    let mut flags = Vec::new();
    let Some(tree) = entry.tree.as_ref() else {
        flags.push(Flag { fail: true, msg: format!("{}: pas d'arbre champion", id) });
        return flags;
    };
    let champ_cost = cost(tree);

    if let Some(stored) = entry.speed.as_ref().and_then(|s| s.formula_cost) {
        if stored != champ_cost {
            flags.push(Flag {
                fail: true,
                msg: format!("{}: coût champion ledger={} recalculé={}", id, stored, champ_cost),
            });
        }
    }

    if let Some(fast_tree) = entry.fast_tree.as_ref() {
        let fast_cost = cost(fast_tree);
        if fast_cost >= champ_cost {
            flags.push(Flag {
                fail: true,
                msg: format!("{}: fast slot coûte {} >= champion {} — ce n'est pas un fast slot", id, fast_cost, champ_cost),
            });
        }
        if let Some(stored) = entry.fast.as_ref().and_then(|f| f.formula_cost) {
            if stored != fast_cost {
                flags.push(Flag {
                    fail: true,
                    msg: format!("{}: coût fast ledger={} recalculé={}", id, stored, fast_cost),
                });
            }
        }
    }

    // Variables : une formule ne doit utiliser que des variables déclarées.
    if let Some(t) = law {
        let used: Vec<String> = crate::ir::var_names(tree);
        for v in used {
            if !t.variables.contains(&v) {
                flags.push(Flag {
                    fail: true,
                    msg: format!("{}: variable inconnue « {} » (déclarées : {})", id, v, t.variables.join(", ")),
                });
            }
        }
    }

    if let Some(ft) = entry.fast_tree.as_ref() {
        if let Some(t) = law {
            for v in crate::ir::var_names(ft) {
                if !t.variables.contains(&v) {
                    flags.push(Flag {
                        fail: false,
                        msg: format!("{}: fast slot utilise une variable non déclarée « {} »", id, v),
                    });
                }
            }
        }
    }

    flags
}

impl Row {
    /// Ligne markdown pour BENCH.md : 10 colonnes (voir [`BENCH_HEADER`]).
    pub fn to_markdown(&self) -> String {
        let gain = match (self.champ_ns, self.fast_ns) {
            (c, Some(f)) if f > 0.0 => format!("×{:.2}", c / f),
            _ => "—".into(),
        };
        let opt = |v: Option<f64>| v.map(|x| format!("{:.2}", x)).unwrap_or_else(|| "—".into());
        let err = |e: Option<ErrStat>| e.map(|x| format!("{:.2e}", x.rmse)).unwrap_or_else(|| "—".into());
        format!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            self.id,
            self.vars,
            self.champ_cost,
            self.fast_cost.map(|c| c.to_string()).unwrap_or_else(|| "—".into()),
            opt(Some(self.champ_ns)),
            self.fast_ns.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "—".into()),
            gain,
            self.law_ns.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "—".into()),
            err(self.champ_err),
            err(self.fast_err),
        )
    }

    /// Rangée alignée pour le terminal.
    pub fn to_text(&self) -> String {
        let gain = match (self.champ_ns, self.fast_ns) {
            (c, Some(f)) if f > 0.0 => format!("x{:.2}", c / f),
            _ => "-".into(),
        };
        let err = |e: Option<ErrStat>| e.map(|x| format!("{:.1e}", x.rmse)).unwrap_or_else(|| "-".into());
        format!(
            "{:<22} {:>2} {:>5} {:>5} {:>8} {:>8} {:>7} {:>8} {:>9} {:>9}",
            self.id,
            self.vars,
            self.champ_cost,
            self.fast_cost.map(|c| c.to_string()).unwrap_or_else(|| "-".into()),
            format!("{:.2}", self.champ_ns),
            self.fast_ns.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".into()),
            gain,
            self.law_ns.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".into()),
            err(self.champ_err),
            err(self.fast_err),
        )
    }
}

/// En-têtes des tableaux de benchmark, partagés par le terminal et le markdown.
pub const BENCH_HEADER: &str = "task                   va  cost  fcost   champ_ns   fast_ns   gain   law_ns  rmse_champ  rmse_fast";
/// En-tête markdown correspondant à [`Row::to_markdown`].
pub const BENCH_MD_HEADER: &str = "| tâche | vars | coût champion | coût fast | champion ns/él | fast ns/él | gain mesuré | loi ns/él | RMSE champion | RMSE fast |";
