//! Parité et invariants du port Rust.
//!
//! Ces tests sont la seule preuve que le port est fidèle : VM == arbre
//! récursif, vecteur == scalaire, coût recalculé == coût du ledger, et le
//! **Rust émis compile et rend la même valeur** (rustc est appelé pour de vrai).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use spear::emit::emit_kernel;
use spear::eval::{eval_tree, erf, Program};
use spear::ir::{cost, var_names};
use spear::ledger::{self, Entry, Laws};
use spear::measure;

fn root() -> PathBuf {
    ledger::repo_root()
}

fn load() -> (BTreeMap<String, Entry>, Laws) {
    let r = root();
    let led = ledger::load(&r.join("spear-hall-of-fame.json")).expect("ledger");
    let laws = Laws::load(&ledger::laws_path(&r)).expect("laws.json");
    (led, laws)
}

/// LCG déterministe : les tests ne doivent pas dépendre de `rand`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next() * (hi - lo)
    }
}

fn scope_for(prog: &Program, rng: &mut Lcg) -> Vec<f64> {
    prog.vars.iter().map(|v| rng.range(-4.0, 4.0)).collect()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
}

/// VM postfix == récursion sur l'arbre, sur **tout** le ledger.
#[test]
fn vm_matches_recursive_tree_eval() {
    let (led, _) = load();
    let mut rng = Lcg(0xC0FFEE);
    let mut checked = 0;
    for (id, e) in &led {
        let tree = match e.tree.as_ref() {
            Some(t) => t,
            None => continue,
        };
        let prog = Program::compile(tree);
        let scope = scope_for(&prog, &mut rng);
        let via_vm = prog.eval(&scope);
        let via_tree = eval_tree(tree, &|name| {
            prog.slot_of(name).map(|s| scope[s]).unwrap_or(0.0)
        });
        assert!(
            close(via_vm, via_tree),
            "{}: VM={} arbre={} (formule {})",
            id,
            via_vm,
            via_tree,
            e.formula.as_deref().unwrap_or("?")
        );
        checked += 1;
    }
    assert!(checked >= 85, "seulement {} arbres testés, le ledger en a plus", checked);
}

/// Le chemin vectoriel rend exactement ce que rend le chemin scalaire.
#[test]
fn vector_path_matches_scalar_path() {
    let (led, _) = load();
    let mut rng = Lcg(4242);
    for (id, e) in &led {
        for tree in [e.tree.as_ref(), e.fast_tree.as_ref()].into_iter().flatten() {
            let prog = Program::compile(tree);
            if prog.vars.is_empty() {
                // champion constant (aucune variable) : rien à échantillonner
                assert_eq!(prog.eval(&[]), prog.eval(&[]));
                continue;
            }
            let cols: Vec<Vec<f64>> = (0..prog.vars.len())
                .map(|_| (0..17).map(|_| rng.range(-3.0, 3.0)).collect())
                .collect();
            let vec_out = prog.run_vec(&cols).expect("run_vec");
            for i in 0..17 {
                let scope: Vec<f64> = cols.iter().map(|c| c[i]).collect();
                assert!(
                    close(vec_out[i], prog.eval(&scope)),
                    "{}: vector={} scalaire={}",
                    id,
                    vec_out[i],
                    prog.eval(&scope)
                );
            }
        }
    }
}

/// Le modèle de coût Rust redonne exactement les coûts stockés dans le ledger.
#[test]
fn champion_cost_matches_ledger_cost() {
    let (led, _) = load();
    let mut checked = 0;
    let mut drift: Vec<String> = Vec::new();
    for (id, e) in &led {
        if let (Some(tree), Some(stored)) = (e.tree.as_ref(), e.speed.as_ref().and_then(|s| s.formula_cost)) {
            let got = cost(tree);
            if got != stored {
                drift.push(format!("{}: ledger={} recalculé={}", id, stored, got));
            }
            checked += 1;
        }
    }
    assert!(checked >= 80, "seulement {} coûts de champion vérifiés", checked);
    assert!(drift.is_empty(), "dérive de coût champion:\n{}", drift.join("\n"));
}

/// Chaque champion reste fini sur sa propre grille de sonde.
#[test]
fn every_champion_stays_finite_on_probe_grid() {
    let (led, laws) = load();
    for (id, e) in &led {
        let tree = match e.tree.as_ref() {
            Some(t) => t,
            None => continue,
        };
        let prog = Program::compile(tree);
        let grid = measure::build_grid(laws.get(id), &prog, 128);
        let out = prog.run_vec(&grid.cols).expect("run_vec");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{}: champion non fini sur la grille de sonde",
            id
        );
    }
}

/// erf : A&S 7.1.26, donc ~1e-7 près des tables — et symétrique.
#[test]
fn erf_matches_abscissa_stegun_reference() {
    assert!((erf(0.0)).abs() < 2e-7, "erf(0) = {}", erf(0.0));
    assert!((erf(1.0) - 0.8427007929).abs() < 2e-7, "erf(1) = {}", erf(1.0));
    assert!((erf(2.0) - 0.9953222650).abs() < 2e-7, "erf(2) = {}", erf(2.0));
    assert!((erf(-1.5) + erf(1.5)).abs() < 1e-12, "erf doit être impaire");
}

/// La tâche 2-variable `swiglu_fused` : la loi exportée rend silu(g)·u.
#[test]
fn swiglu_law_is_two_variable_and_correct() {
    let (_, laws) = load();
    let t = laws.get("swiglu_fused").expect("swiglu_fused dans laws.json");
    assert_eq!(t.variables, vec!["g".to_string(), "u".to_string()]);
    let law = t.law.as_ref().expect("loi swiglu_fused");
    let prog = Program::compile(law);
    assert_eq!(prog.vars.len(), 2);
    // u·silu(g) avec g=1.5, u=2 → 2·1.5/(1+e^-1.5) = 2.670459...
    let v = {
        let g = prog.slot_of("g").unwrap();
        let u = prog.slot_of("u").unwrap();
        let mut scope = vec![0.0; prog.vars.len()];
        scope[g] = 1.5;
        scope[u] = 2.0;
        prog.eval(&scope)
    };
    let expected = 2.0 * 1.5 / (1.0 + (-1.5f64).exp());
    assert!((v - expected).abs() < 1e-12, "swiglu(1.5,2) = {} attendu {}", v, expected);
}

/// Le Rust **émis** compile avec rustc et rend la même valeur que la VM.
#[test]
fn emitted_rust_compiles_and_matches_vm() {
    let (led, _) = load();
    let cases = [
        ("silu", false),   // champion : pdiv + exp
        ("silu", true),    // fast slot : relu
        ("gelu", false),   // champion : erf
        ("chsh_correlation", false), // multi-variable
        ("swiglu_fused", false),    // 2 variables, loi exacte
    ];
    let mut rng = Lcg(7);
    for (id, fast) in cases {
        let e = led.get(id).unwrap_or_else(|| panic!("{} dans le ledger", id));
        let tree = if fast { e.fast_tree.as_ref() } else { e.tree.as_ref() }
            .unwrap_or_else(|| panic!("{}: arbre manquant (fast={})", id, fast));
        let formula = if fast {
            e.fast.as_ref().and_then(|f| f.formula.clone()).unwrap_or_default()
        } else {
            e.formula.clone().unwrap_or_default()
        };
        let code = emit_kernel(tree, id, &formula);

        let dir = std::env::temp_dir().join(format!("spear-rs-parity-{}-{}", id, fast as u8));
        std::fs::create_dir_all(&dir).expect("répertoire temporaire");
        let main = dir.join("main.rs");
        let prog = Program::compile(tree);
        // appelle le kernel scalaire sur une entrée fixed par variable
        let scope: Vec<f64> = prog.vars.iter().map(|v| rng.range(0.2, 1.4)).collect();
        let args: Vec<String> = scope.iter().map(|v| format!("{:?}", v)).collect();
        let fname = format!("spear_{}", id.replace('-', "_"));
        std::fs::write(
            &main,
            format!(
                "#![allow(dead_code, unused_imports, unused_variables, unused_mut)]\n{}\nfn main() {{ println!(\"{{:.17}}\", {}({})); }}\n",
                code,
                fname,
                args.join(", ")
            ),
        )
        .expect("écriture main.rs");

        let exe = dir.join("parity.exe");
        let out = Command::new("rustc")
            .args(["--edition", "2021", "-O", "--crate-name", "parity"])
            .arg(&main)
            .arg("-o")
            .arg(&exe)
            .output()
            .expect("rustc disponible");
        assert!(
            out.status.success(),
            "rustc a échoué pour {}{} :\n{}",
            id,
            if fast { " --fast" } else { "" },
            String::from_utf8_lossy(&out.stderr)
        );

        let run = Command::new(&exe).output().expect("exécution");
        let got: f64 = String::from_utf8_lossy(&run.stdout).trim().parse().expect("sortie numérique");
        let want = prog.eval(&scope);
        assert!(
            close(got, want),
            "{}: Rust émis = {} mais VM = {} (args {:?})",
            id,
            got,
            want,
            scope
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Le kernel f32 (format GPU/Metal) compile lui aussi, bornages compris, et
/// reste fidèle à la VM à la précision f32 près. C'est le chemin qu'il faut
/// pour TensorFold/MLX, donc il est testé, pas supposé.
#[test]
fn emitted_f32_rust_compiles_and_tracks_vm() {
    use spear::emit::{emit_kernel_flavor, Float};
    let (led, _) = load();
    for id in ["silu", "gelu", "swiglu_fused"] {
        let e = led.get(id).unwrap_or_else(|| panic!("{} dans le ledger", id));
        let tree = e.tree.as_ref().expect("champion");
        let code = emit_kernel_flavor(tree, id, e.formula.as_deref().unwrap_or(""), Float::F32);
        assert!(code.contains("-> f32"), "{} : le kernel f32 doit typer en f32", id);
        let dir = std::env::temp_dir().join(format!("spear-rs-f32-{}", id));
        std::fs::create_dir_all(&dir).expect("répertoire temporaire");
        let main = dir.join("main.rs");
        let prog = Program::compile(tree);
        let mut rng = Lcg(31);
        let scope: Vec<f64> = prog.vars.iter().map(|_| rng.range(0.3, 1.2)).collect();
        let args: Vec<String> = scope.iter().map(|v| format!("{:?}f32", v)).collect();
        let fname = format!("spear_{}", id.replace('-', "_"));
        std::fs::write(
            &main,
            format!(
                "#![allow(dead_code, unused_imports, unused_variables, unused_mut)]\n{}\nfn main() {{ println!(\"{{:.9}}\", {}({})); }}\n",
                code,
                fname,
                args.join(", ")
            ),
        )
        .expect("écriture main.rs");
        let exe = dir.join("f32.exe");
        let build = Command::new("rustc")
            .args(["--edition", "2021", "-O", "--crate-name", "k32"])
            .arg(&main)
            .arg("-o")
            .arg(&exe)
            .output()
            .expect("rustc");
        assert!(
            build.status.success(),
            "rustc a échoué pour {} en f32 :\n{}",
            id,
            String::from_utf8_lossy(&build.stderr)
        );
        let run = Command::new(&exe).output().expect("exécution");
        let got: f64 = String::from_utf8_lossy(&run.stdout).trim().parse().expect("sortie numérique");
        let want = prog.eval(&scope);
        // tolérance f32 sur la magnitude du kernel (relatif 1e-6, absolue 1e-5)
        assert!(
            (got - want).abs() <= 1e-5 + 1e-6 * want.abs().max(1.0),
            "{}: f32 émis = {} mais VM = {} (écart hors tolérance f32)",
            id,
            got,
            want
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Les variables exportées correspondent à celles réellement utilisées.
#[test]
fn exported_laws_cover_most_tasks() {
    let (led, laws) = load();
    assert_eq!(laws.task_count, led.len(), "laws.json doit couvrir tout le registre");
    assert!(
        laws.with_law >= 80,
        "seulement {} lois exportées sur {} tâches — l'export a régressé",
        laws.with_law,
        led.len()
    );
    // et les variables déclarées couvrent bien celles du champion
    for (id, e) in &led {
        if let (Some(tree), Some(t)) = (e.tree.as_ref(), laws.get(id)) {
            for v in var_names(tree) {
                assert!(
                    t.variables.contains(&v),
                    "{}: champion utilise « {} », variables exportées {:?}",
                    id,
                    v,
                    t.variables
                );
            }
        }
    }
}
