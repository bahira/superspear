//! CLI `spear` : lecture du ledger, émission de kernels Rust, vérification,
//! audit et benchmark des champions SPEAR.
//!
//! ```
//! spear list                      # vue d'ensemble du ledger
//! spear show silu [--fast]        # formule, coût détaillé, mesure enregistrée
//! spear emit swiglu_fused         # kernel Rust drop-in sur stdout
//! spear eval swiglu_fused --set g=1.5 --set u=2
//! spear verify                    # erreur de chaque champion contre sa loi
//! spear audit --strict            # contrôle de coût (exit 1 si échec)
//! spear bench --all --md BENCH.md # tableau ns/élément pour tout le registre
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use spear::emit::{emit_kernel_flavor, Float};
use spear::eval::Program;
use spear::ir::{cost, op_histogram};
use spear::ledger::{self, Entry, Laws};
use spear::measure::{self, BENCH_HEADER, BENCH_MD_HEADER};

struct Opts {
    positional: Vec<String>,
    flags: BTreeMap<String, String>,
    bools: Vec<String>,
}

impl Opts {
    fn parse(args: &[String]) -> Self {
        let mut o = Opts { positional: Vec::new(), flags: BTreeMap::new(), bools: Vec::new() };
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            if let Some(rest) = a.strip_prefix("--") {
                if let Some((k, v)) = rest.split_once('=') {
                    o.flags.insert(k.to_string(), v.to_string());
                    i += 1;
                    continue;
                }
                // option à valeur si le prochain token n'est pas une option
                if i + 1 < args.len() && !args[i + 1].starts_with("--") && takes_value(rest) {
                    o.flags.insert(rest.to_string(), args[i + 1].clone());
                    i += 2;
                } else {
                    o.bools.push(rest.to_string());
                    i += 1;
                }
            } else {
                o.positional.push(a.clone());
                i += 1;
            }
        }
        o
    }

    fn has(&self, name: &str) -> bool {
        self.bools.iter().any(|b| b == name)
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(|s| s.as_str())
    }

    fn num(&self, name: &str, default: usize) -> usize {
        self.get(name).and_then(|v| v.parse().ok()).unwrap_or(default)
    }
}

/// Options qui mangent la valeur suivante ; les autres sont des drapeaux.
fn takes_value(name: &str) -> bool {
    matches!(
        name,
        "ledger" | "laws" | "rows" | "samples" | "repeat" | "md" | "set" | "fn" | "id" | "noise"
    )
}

struct Ctx {
    ledger: BTreeMap<String, Entry>,
    laws: Option<Laws>,
    root: PathBuf,
}

fn load_ctx(o: &Opts) -> Result<Ctx, String> {
    let root = match o.get("ledger") {
        Some(p) => PathBuf::from(p),
        None => ledger::repo_root(),
    };
    let bundle = if root.is_file() { root.clone() } else { root.join("spear-hall-of-fame.json") };
    let ledger = ledger::load(&bundle)?;
    let laws_path = match o.get("laws") {
        Some(p) => PathBuf::from(p),
        None => ledger::laws_path(&ledger::repo_root()),
    };
    let laws = match Laws::load(&laws_path) {
        Ok(l) => Some(l),
        Err(e) => {
            eprintln!("#laws absents ({}) — verify/bench_news Concordia skippent les cibles", e);
            None
        }
    };
    Ok(Ctx { ledger, laws, root: bundle.parent().unwrap_or(&root).to_path_buf() })
}

fn ids_of(ctx: &Ctx, o: &Opts) -> Vec<String> {
    if !o.positional.is_empty() {
        return o.positional.clone();
    }
    let mut v: Vec<String> = ctx.ledger.keys().cloned().collect();
    if o.has("fast") {
        v.retain(|id| ctx.ledger.get(id).map(|e| e.fast_tree.is_some()).unwrap_or(false));
    }
    v
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "-h" || argv[0] == "--help" || argv[0] == "help" {
        usage();
        return ExitCode::from(2);
    }
    let o = Opts::parse(&argv[1..]);
    let ctx = match load_ctx(&o) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("erreur: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let result: Result<i32, String> = match argv[0].as_str() {
        "list" => cmd_list(&ctx, &o),
        "show" => cmd_show(&ctx, &o),
        "emit" => cmd_emit(&ctx, &o),
        "eval" => cmd_eval(&ctx, &o),
        "verify" => cmd_verify(&ctx, &o),
        "audit" => cmd_audit(&ctx, &o),
        "bench" => cmd_bench(&ctx, &o),
        "emit-bench" => cmd_emit_bench(&ctx, &o),
        other => Err(format!("commande inconnue « {} » (voir --help)", other)),
    };
    match result {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("erreur: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    eprintln!(
        "spear — port Rust du moteur SPEAR

USAGE
  spear list [--fast]                 vue d'ensemble du ledger
  spear show <id> [--fast]            formule, coût par op, mesure enregistrée
  spear emit <id> [--fast] [--f32]   kernel Rust drop-in sur stdout (f64 par défaut)
  spear eval <id> --set x=0.5 [...]   valeur(s) du kernel
  spear verify [<id>...]              erreur du champion contre la loi génératrice
  spear audit [--strict]               recalcul des coûts (exit 1 si échec)
  spear bench [<id>...] [--rows N] [--samples K] [--repeat R] [--md fichier]
                                       ns/élément pour tout le registre

OPTIONS COMMUNS
  --f32                               kernels émis en f32 (GPU/Metal)
  --ledger <chemin>                   bundle JSON (défaut: spear-hall-of-fame.json du dépôt)
  --laws <chemin>                     lois exportées (défaut: spear-rs/data/laws.json)
"
    );
}

/// `--f32` bascule tous les kernels émis en f32 (format GPU/Metal).
fn flavor(o: &Opts) -> Float {
    if o.has("f32") {
        Float::F32
    } else {
        Float::F64
    }
}

fn get_entry<'a>(ctx: &'a Ctx, id: &str) -> Result<&'a Entry, String> {
    ctx.ledger.get(id).ok_or_else(|| format!("tâche inconnue « {} » (spear list pour la liste)", id))
}

fn cmd_list(ctx: &Ctx, o: &Opts) -> Result<i32, String> {
    println!("{:<22} {:>5} {:>6} {:>6} {:>9}  {}", "task", "vars", "cost", "fcost", "gain_modele", "mesure_enregistree");
    let ids = ids_of(ctx, o);
for id in &ids {
        let e = match ctx.ledger.get(id) {
            Some(e) => e,
            None => continue,
        };
        let champ = e.tree.as_ref().map(cost).unwrap_or(0);
        let fast = e.fast_tree.as_ref().map(cost);
        let gain = match (champ, fast) {
            (c, Some(f)) if f > 0 => format!("x{:.2}", c as f64 / f as f64),
            _ => "-".into(),
        };
        let measured = e
            .fast
            .as_ref()
            .and_then(|f| f.measured.as_ref())
            .and_then(|m| m.measured_speedup)
            .map(|v| format!("x{:.2}", v))
            .unwrap_or_else(|| "-".into());
        println!("{:<22} {:>5} {:>6} {:>6} {:>11}  {}", id, e.tree.as_ref().map(spear::ir::var_names).map(|v| v.len()).unwrap_or(0), champ, fast.map(|c| c.to_string()).unwrap_or_else(|| "-".into()), gain, measured);
    }
    println!("\n{} tâches · racine {}", ids.len(), ctx.root.display());
    Ok(0)
}

fn cmd_show(ctx: &Ctx, o: &Opts) -> Result<i32, String> {
    let id = o.positional.first().ok_or("spear show <id>")?;
    let e = get_entry(ctx, id)?;
    let use_fast = o.has("fast");
    let tree = if use_fast { e.fast_tree.as_ref() } else { e.tree.as_ref() };
    let tree = tree.ok_or_else(|| format!("{}: pas d'arbre {}", id, if use_fast { "fast" } else { "champion" }))?;
    let formula = if use_fast {
        e.fast.as_ref().and_then(|f| f.formula.clone()).unwrap_or_default()
    } else {
        e.formula.clone().unwrap_or_default()
    };
    println!("# {} — {}", id, e.display_id(id));
    if let Some(t) = e.title.as_deref() {
        println!("{}", t);
    }
    println!("variante : {}", if use_fast { "slot rapide (fastTree)" } else { "champion (tree)" });
    println!("formule  : {}", formula);
    println!("coût     : {} unités ALU/SFU", cost(tree));
    println!("variables: {}", spear::ir::var_names(tree).join(", "));
    println!("ops      :");
    for (op, n) in op_histogram(tree) {
        println!("  {:>3}× {:<6} ({} u/op)", n, format!("{:?}", op).to_lowercase(), spear::ir::op_cost(op));
    }
    if let Some(m) = e.metric {
        println!("metric ledger (entraînement TS) : {:.6e}", m);
    }
    if let Some(h) = e.holdout {
        println!("holdout                        : {:.6e}", h);
    }
    if let Some(s) = e.speed.as_ref() {
        if let Some(v) = s.speedup {
            println!("speedup modèle                 : x{:.3}", v);
        }
        if let Some(ec) = s.exact_cost {
            println!("coût loi exacte                : {} u", ec);
        }
        if let Some(m) = s.measured.as_ref() {
            println!("mesure GCC                     : x{:.3}", m.measured_speedup.unwrap_or(f64::NAN));
            println!("  harnais : {}", m.harness.as_deref().unwrap_or("-"));
        }
    }
    if let Some(f) = e.fast.as_ref() {
        println!("fast slot : coût {} u · metric {:.3e} · niveau L{}", f.formula_cost.unwrap_or(-1), f.metric.unwrap_or(f64::NAN), f.level.unwrap_or(0));
        if let Some(m) = f.measured.as_ref() {
            println!("  mesure : champion {:.2} ns vs fast {:.2} ns", m.ns_champion.unwrap_or(f64::NAN), m.ns_fast.unwrap_or(f64::NAN));
        }
    }
    println!("\n--- Rust émis (spear emit {}{}) ---", id, if use_fast { " --fast" } else { "" });
    print!("{}", emit_kernel_flavor(tree, id, &formula, flavor(o)));
    Ok(0)
}

fn cmd_emit(ctx: &Ctx, o: &Opts) -> Result<i32, String> {
    let id = o.positional.first().ok_or("spear emit <id>")?;
    let e = get_entry(ctx, id)?;
    let use_fast = o.has("fast");
    let tree = if use_fast { e.fast_tree.as_ref() } else { e.tree.as_ref() }
        .ok_or_else(|| format!("{}: pas d'arbre {}", id, if use_fast { "fast" } else { "champion" }))?;
    let formula = if use_fast {
        e.fast.as_ref().and_then(|f| f.formula.clone()).unwrap_or_default()
    } else {
        e.formula.clone().unwrap_or_default()
    };
    let code = emit_kernel_flavor(tree, id, &formula, flavor(o));
    match o.get("fn") {
        // --fn NAME : n'émet que le corps de la fonction scalaire (usage rapide)
        Some(_) => {
            let start = code.find("#[inline(always)]").unwrap_or(0);
            let rest = &code[start..];
            let end = rest.find("/// Passe fusionnée").unwrap_or(rest.len());
            println!("{}", &rest[..end]);
        }
        None => print!("{}", code),
    }
    Ok(0)
}

fn cmd_eval(ctx: &Ctx, o: &Opts) -> Result<i32, String> {
    let id = o.positional.first().ok_or("spear eval <id> --set nom=valeur")?;
    let e = get_entry(ctx, id)?;
    let use_fast = o.has("fast");
    let tree = if use_fast { e.fast_tree.as_ref() } else { e.tree.as_ref() }
        .ok_or_else(|| format!("{}: pas d'arbre", id))?;
    let prog = Program::compile(tree);
    // Un `--set` par variable non fournie ; sans valeur, on échantillonne la grille.
    let mut scope = vec![0.0f64; prog.vars.len()];
    let mut provided = false;
    let mut sets: Vec<(String, String)> = Vec::new();
    for s in &o.flags {
        if s.0 == "set" {
            if let Some((k, v)) = s.1.split_once('=') {
                sets.push((k.to_string(), v.to_string()));
            }
        }
    }
    for (k, v) in &sets {
        let slot = prog.slot_of(k).ok_or_else(|| format!("variable « {} » absente de la formule (utilise : {})", k, prog.vars.join(", ")))?;
        scope[slot] = v.parse::<f64>().map_err(|_| format!("valeur non numérique : {}={}", k, v))?;
        provided = true;
    }
    if provided {
        println!("{} = {:.17}", prog.vars.join(", "), prog.eval(&scope));
    } else {
        let rows = o.num("rows", 9);
        let law = ctx.laws.as_ref().and_then(|l| l.get(id));
        let grid = measure::build_grid(law, &prog, rows);
        let inputs = grid.inputs();
        let mut out = vec![0.0f64; rows];
        prog.run_into(&inputs, &mut out)?;
        for (i, v) in out.iter().enumerate() {
            println!("{:>18.10} -> {:.17}", grid.cols[0][i], v);
        }
    }
    Ok(0)
}

fn cmd_verify(ctx: &Ctx, o: &Opts) -> Result<i32, String> {
    let rows = o.num("rows", 256);
    let ids = ids_of(ctx, o);
    println!("{:<22} {:>11} {:>11} {:>11}  {}", "task", "rmse_champ", "rmse_fast", "max_abs", "statut");
    let mut with_law = 0usize;
    let mut skipped: Vec<String> = Vec::new();
    for id in &ids {
        let e = match ctx.ledger.get(id) {
            Some(e) => e,
            None => continue,
        };
        let tree = match e.tree.as_ref() {
            Some(t) => t,
            None => {
                skipped.push(format!("{} (pas d'arbre)", id));
                continue;
            }
        };
        let law = ctx.laws.as_ref().and_then(|l| l.get(id));
        let prog = Program::compile(tree);
        let grid = measure::build_grid(law, &prog, rows);
        let target = match grid.target.as_ref() {
            Some(t) => t.clone(),
            None => {
                skipped.push(id.clone());
                println!("{:<22} {:>11} {:>11} {:>11}  {}", id, "-", "-", "-", "pas de loi — saute");
                continue;
            }
        };
        with_law += 1;
        let pred = prog.run_vec(&grid.cols)?;
        let st = measure::errors(&pred, &target);
        let (fast_rmse, fast_max) = match e.fast_tree.as_ref() {
            Some(f) => {
                let fp = Program::compile(f);
                let fg = measure::build_grid(law, &fp, rows);
                match (fp.run_vec(&fg.cols), fg.target.as_ref()) {
                    (Ok(p), Some(t)) => {
                        let e = measure::errors(&p, t);
                        (Some(e.rmse), Some(e.max_abs))
                    }
                    _ => (None, None),
                }
            }
            None => (None, None),
        };
        let max_all = fast_max.unwrap_or(st.max_abs).max(st.max_abs);
        let fmt = |v: Option<f64>| v.map(|x| format!("{:.3e}", x)).unwrap_or_else(|| "-".into());
        println!("{:<22} {:>11} {:>11} {:>11}  {}", id, format!("{:.3e}", st.rmse), fmt(fast_rmse), format!("{:.3e}", max_all), "ok");
    }
    println!("\n{} tâches vérifiées · {} avec loi · {} sautées", ids.len(), with_law, skipped.len());
    if !skipped.is_empty() {
        println!("sautées : {}", skipped.join(", "));
    }
    Ok(0)
}

fn cmd_audit(ctx: &Ctx, o: &Opts) -> Result<i32, String> {
    let ids = ids_of(ctx, o);
    let mut fails = 0usize;
    let mut warns = 0usize;
    let mut clean = 0usize;
    for id in &ids {
        let e = match ctx.ledger.get(id) {
            Some(e) => e,
            None => continue,
        };
        let law = ctx.laws.as_ref().and_then(|l| l.get(id));
        let flags = measure::audit_entry(id, e, law);
        if flags.is_empty() {
            clean += 1;
        }
        for f in &flags {
            if f.fail {
                fails += 1;
                println!("✗ {}", f.msg);
            } else {
                warns += 1;
                println!("! {}", f.msg);
            }
        }
    }
    println!("== audit-cost (Rust) : {} propre(s), {} avertissement(s), {} échec(s) sur {} tâches", clean, warns, fails, ids.len());
    if o.has("strict") && fails > 0 {
        return Ok(1);
    }
    Ok(0)
}

fn cmd_emit_bench(ctx: &Ctx, o: &Opts) -> Result<i32, String> {
    let rows = o.num("rows", 1 << 20);
    let samples = o.num("samples", 9);
    let ids: Vec<String> = if o.positional.is_empty() {
        // liste courte par défaut : compiler 90 kernels prendrait des minutes
        ["silu", "gelu", "sigmoid", "tanh_sat", "erf_prob", "rope_rot", "swiglu_fused", "bilateral_weight"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        o.positional.clone()
    };
    println!("# spear-rs emit-bench · rustc -O · rows={} samples={} (ns PAR ÉLÉMENT)", rows, samples);
    println!("{:<20} {:>7} {:>12} {:>12} {:>8}  {}", "task", "variante", "ns/elem", "vm_ns/elem", "compilé/VM", "checksum (anti-dead-code)");
    for id in &ids {
        let e = match ctx.ledger.get(id) {
            Some(e) => e,
            None => continue,
        };
        let law = ctx.laws.as_ref().and_then(|l| l.get(id));
        let variants: Vec<(bool, &str)> = if e.fast_tree.is_some() {
            vec![(false, "champion"), (true, "fast")]
        } else {
            vec![(false, "champion")]
        };
        for (fast, label) in variants {
            let tree = if fast { e.fast_tree.as_ref() } else { e.tree.as_ref() };
            let tree = match tree {
                Some(t) => t,
                None => continue,
            };
            let formula = if fast {
                e.fast.as_ref().and_then(|f| f.formula.clone()).unwrap_or_default()
            } else {
                e.formula.clone().unwrap_or_default()
            };
            let prog = Program::compile(tree);
            if prog.vars.is_empty() {
                println!("{:<20} {:>7}  (constante pure — rien à chronométrer)", id, label);
                continue;
            }
            let code = emit_kernel_flavor(tree, id, &formula, flavor(o));
            let fname = format!("spear_{}_into", id.replace('-', "_"));
            // domaine : celui de la tâche s'il est déclaré, sinon [lo,hi] par variable
            let cols: Vec<(String, f64, f64)> = prog
                .vars
                .iter()
                .map(|v| match law {
                    Some(t) if t.domain.is_some() && prog.vars.len() == 1 => {
                        (v.clone(), t.domain.as_ref().unwrap().lo, t.domain.as_ref().unwrap().hi)
                    }
                    _ => {
                        let (lo, hi) = measure::default_span(v);
                        (v.clone(), lo, hi)
                    }
                })
                .collect();
            let decls: Vec<String> = cols
                .iter()
                .map(|(v, lo, hi)| format!("let {}: Vec<f64> = (0..n).map(|i| {:?} + {:?} * i as f64 / (n - 1) as f64).collect();", v, lo, hi - lo))
                .collect();
            // le kernel émis attend des slices : on passe `&Vec<f64>` (déréf coercion)
            let call_args: Vec<String> = cols.iter().map(|(v, _, _)| format!("&{}", v)).collect();
            let driver = format!(
                "#![allow(dead_code, unused_imports, unused_variables, unused_mut)]\n{}\n\
                 fn main() {{\n    let n = {}usize;\n    {}\n    let mut out = vec![0.0f64; n];\n    \
                 let mut best = f64::INFINITY;\n    for _ in 0..{} {{\n        let t = std::time::Instant::now();\n        {}({}, &mut out);\n        \
                 let d = t.elapsed().as_secs_f64() * 1e9 / n as f64;\n        if d < best {{ best = d; }}\n    }}\n    \
                 let mut acc = 0.0f64;\n    for v in &out {{ acc += *v; }}\n    println!(\"{{:.6}} {{:.6}}\", best, acc);\n}}\n",
                code,
                rows,
                decls.join("\n    "),
                samples,
                fname,
                call_args.join(", ")
            );
            let dir = std::env::temp_dir().join(format!("spear-rs-ebench-{}", id.replace('-', "_")));
            std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
            let main_rs = dir.join("main.rs");
            let exe = dir.join("ebench.exe");
            std::fs::write(&main_rs, &driver).map_err(|e| format!("écriture driver : {}", e))?;
            let build = std::process::Command::new("rustc")
                .args(["--edition", "2021", "-O", "--crate-name", "ebench"])
                .arg(&main_rs)
                .arg("-o")
                .arg(&exe)
                .output()
                .map_err(|e| format!("rustc : {}", e))?;
            if !build.status.success() {
                println!(
                    "{:<20} {:>7}  rustc a échoué : {}",
                    id,
                    label,
                    String::from_utf8_lossy(&build.stderr)
                        .lines()
                        .find(|l| l.contains("error"))
                        .unwrap_or("voir rustc")
                );
                continue;
            }
            let run = std::process::Command::new(&exe).output().map_err(|e| format!("run : {}", e))?;
            let text = String::from_utf8_lossy(&run.stdout);
            let mut parts = text.split_whitespace();
            let ns: f64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(f64::NAN);
            let acc: f64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(f64::NAN);
            // référence VM sur la même grille, pour montrer l'écart interpréteur/compilé
            let vm_ns = {
                let grid = measure::build_grid(law, &prog, 4096);
                measure::time_ns_per_elem(&prog, &grid, 5, 4)
            };
            println!(
                "{:<20} {:>7} {:>12.4} {:>12.2} {:>8}  checksum {:.6}",
                id,
                label,
                ns,
                vm_ns,
                if ns > 0.0 { format!("x{:.0}", vm_ns / ns) } else { "-".into() },
                acc
            );
        }
    }
    println!("\nns/elem = meilleur de {} passes sur {} éléments, kernel **compilé** (rustc -O).", samples, rows);
    println!("vm_ns/elem = même formule interprétée par la VM postfix : l'écart mesure le coût de l'interpréteur, pas celui du kernel.");
    Ok(0)
}

fn cmd_bench(ctx: &Ctx, o: &Opts) -> Result<i32, String> {
    let rows = o.num("rows", 4096);
    let samples = o.num("samples", 9);
    let repeat = o.num("repeat", 8);
    let ids = ids_of(ctx, o);
    let mut out_rows = Vec::new();
    println!("# spear-rs bench · {} tâches · rows={} samples={} repeat={} · release", ids.len(), rows, samples, repeat);
    println!("{}", BENCH_HEADER);
    for id in &ids {
        let e = match ctx.ledger.get(id) {
            Some(e) => e,
            None => continue,
        };
        if e.tree.is_none() {
            continue;
        }
        let law = ctx.laws.as_ref().and_then(|l| l.get(id));
        let row = measure::bench_task(id, e, law, rows, samples, repeat)?;
        println!("{}", row.to_text());
        out_rows.push(row);
    }
    // gain médian sur les tâches qui ont un fast slot, mesuré sur cette machine
    let mut gains: Vec<f64> = out_rows
        .iter()
        .filter_map(|r| match (r.champ_ns, r.fast_ns) {
            (c, Some(f)) if f > 0.0 && c.is_finite() => Some(c / f),
            _ => None,
        })
        .collect();
    // fast slots que le modèle de coût dit « moins chers » mais que la machine
    // dit plus lents — c'est le symptôme runtime d'un fast slot corrompu.
    // Seuil à 0,90 : en dessous c'est du bruit (pas d'épinglage de cœur, pas
    // de fréquence fixe), et une liste pleine de faux positifs n'aide personne.
    let noise = {
        let v: f64 = o.get("noise").and_then(|s| s.parse().ok()).unwrap_or(0.90);
        v
    };
    let slower: Vec<(String, f64)> = out_rows
        .iter()
        .filter_map(|r| match (r.champ_ns, r.fast_ns) {
            (c, Some(f)) if f > 0.0 && c.is_finite() && f > c / noise => Some((r.id.clone(), c / f)),
            _ => None,
        })
        .collect();
    if !gains.is_empty() {
        gains.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "\nfast slot : {} tâches · gain médian x{:.2} · min x{:.2} · max x{:.2}",
            gains.len(),
            measure::median(gains.clone()),
            gains[0],
            gains[gains.len() - 1]
        );
    }
    if !slower.is_empty() {
        println!("\n⚠ {} fast slot(s) mesurés PLUS LENTS que leur champion :", slower.len());
        for (id, g) in &slower {
            println!("  {} (x{:.2} mesuré)", id, g);
        }
    }
    if let Some(path) = o.get("md") {
        let mut md = String::new();
        md.push_str("# spear-rs — benchmark du registre\n\n");
        md.push_str(&format!(
            "Généré par `spear bench --rows {} --samples {} --repeat {}` (build release).\n",
            rows, samples, repeat
        ));
        md.push_str("\n**Méthodologie.** Une passe = `rows` éléments par variable ; 1 chauffe puis ");
        md.push_str(&format!("{} passes mesurées de {} répétitions, médiane conservée ; sortie lue dans un `black_box`.\n", samples, repeat));
        md.push_str("`ns/él` est donc **ns par élément** (divisé par le nombre de variables quand il y en a plusieurs). ");
        md.push_str("Ce sont des chiffres *relatifs sur cette machine*, pas des performances absolues : pas d'épinglage de cœur, pas de fréquence fixe.\n\n");
        md.push_str("`gain mesuré` = champion ÷ fast slot sur cette machine (colonne `fast ns/él` non nulle). ");
        md.push_str("`loi ns/él` = la loi génératrice (`data/laws.json`), absente pour 6 tâches sans forme close.\n\n");
        md.push_str("`RMSE champion` / `RMSE fast` = erreur contre la loi **sur la grille de sonde de spear-rs**, ");
        md.push_str("qui n'est PAS la grille d'entraînement TypeScript : à comparer aux métriques du ledger (`spear show <id>`), jamais entre elles.\n\n");
        md.push_str(&format!("{}\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n", BENCH_MD_HEADER));
        for r in &out_rows {
            md.push_str(&r.to_markdown());
            md.push('\n');
        }
        if !gains.is_empty() {
            md.push_str(&format!(
                "\n**Synthèse.** {} tâches ont un fast slot ; gain médian mesuré **x{:.2}** (min x{:.2}, max x{:.2}).\n",
                gains.len(),
                measure::median(gains.clone()),
                gains[0],
                gains[gains.len() - 1]
            ));
        }
        if !slower.is_empty() {
            md.push_str(&format!(
                "\n### ⚠ {} fast slot(s) mesurés plus lents que leur champion\n\n",
                slower.len()
            ));
            md.push_str("Le modèle de coût les déclare « moins chers » ; la machine dit le contraire. ");
            md.push_str("Ce sont des fast slots dont l'arbre contient des transcendantes (souvent la loi exacte restaurée par `scripts/backfill-fast.ts`).\n\n");
            md.push_str("| tâche | gain mesuré |\n|---|---:|\n");
            for (id, g) in &slower {
                md.push_str(&format!("| `{}` | x{:.2} |\n", id, g));
            }
        }
        md.push_str("\n---\n\nRégénéré par `spear bench --md`. Kernels émis : `spear emit <id> [--fast]`.\n");
        std::fs::write(path, md).map_err(|e| format!("écriture {} : {}", path, e))?;
        println!("\nmarkdown écrit : {}", path);
    }
    Ok(0)
}
