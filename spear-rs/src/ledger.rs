//! Lecture du ledger SPEAR et du fichier de lois exporté pour Rust.
//!
//! Le ledger reste la source de vérité côté TS ; ici on n'en fait qu'un lecteur.
//! Schéma volontairement permissif (`Option` partout, `default` everywhere) :
//! le format a évolué au fil des versions et un crate de mesure ne doit pas
//! casser sur une entrée ancienne.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::ir::Node;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Measured {
    #[serde(rename = "measuredSpeedup", default)]
    pub measured_speedup: Option<f64>,
    #[serde(rename = "nsFormula", default)]
    pub ns_formula: Option<f64>,
    #[serde(rename = "nsExact", default)]
    pub ns_exact: Option<f64>,
    #[serde(rename = "nsChampion", default)]
    pub ns_champion: Option<f64>,
    #[serde(rename = "nsFast", default)]
    pub ns_fast: Option<f64>,
    #[serde(default)]
    pub resolvable: Option<bool>,
    #[serde(default)]
    pub harness: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Speed {
    #[serde(rename = "formulaCost", default)]
    pub formula_cost: Option<i64>,
    #[serde(rename = "exactCost", default)]
    pub exact_cost: Option<i64>,
    #[serde(default)]
    pub speedup: Option<f64>,
    #[serde(rename = "costProfile", default)]
    pub cost_profile: Option<String>,
    #[serde(default)]
    pub measured: Option<Measured>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Fast {
    #[serde(default)]
    pub formula: Option<String>,
    #[serde(default)]
    pub metric: Option<f64>,
    #[serde(default)]
    pub level: Option<u8>,
    #[serde(rename = "formulaCost", default)]
    pub formula_cost: Option<i64>,
    #[serde(default)]
    pub deploy: Option<bool>,
    #[serde(default)]
    pub measured: Option<Measured>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Entry {
    #[serde(rename = "taskId", default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub formula: Option<String>,
    #[serde(default)]
    pub metric: Option<f64>,
    #[serde(default)]
    pub level: Option<u8>,
    #[serde(default)]
    pub tree: Option<Node>,
    #[serde(default)]
    pub fast: Option<Fast>,
    #[serde(rename = "fastTree", default)]
    pub fast_tree: Option<Node>,
    #[serde(default)]
    pub speed: Option<Speed>,
    #[serde(default)]
    pub holdout: Option<f64>,
}

impl Entry {
    pub fn display_id<'a>(&'a self, fallback: &'a str) -> &'a str {
        self.task_id.as_deref().unwrap_or(fallback)
    }
}

pub type Ledger = BTreeMap<String, Entry>;

/// Charge le bundle `spear-hall-of-fame.json` (ou n'importe quel objet JSON du
/// même shape).
pub fn load(path: &Path) -> Result<Ledger, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("lecture {} : {}", path.display(), e))?;
    serde_json::from_str::<Ledger>(&raw).map_err(|e| format!("JSON {} : {}", path.display(), e))
}

/// Remonte l'arborescence jusqu'à trouver le ledger du dépôt SPEAR.
pub fn repo_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut cur: Option<&Path> = Some(manifest);
    while let Some(dir) = cur {
        if dir.join("spear-hall-of-fame.json").is_file() {
            return dir.to_path_buf();
        }
        cur = dir.parent();
    }
    manifest.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."))
}

// ------------------------------------------------------------------ lois

#[derive(Debug, Clone, Deserialize)]
pub struct Span {
    pub lo: f64,
    pub hi: f64,
}

/// Une tâche du registre, telle qu'exportée par `scripts/export-rust-data.ts`.
#[derive(Debug, Clone, Deserialize)]
pub struct LawTask {
    pub id: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub variables: Vec<String>,
    #[serde(default)]
    pub domain: Option<Span>,
    /// Loi génératrice en AST ; `None` = pas de forme close (solveur, intégrateur,
    /// politique KV). `verify` saute ces tâches et le dit.
    #[serde(default)]
    pub law: Option<Node>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Laws {
    #[serde(rename = "taskCount", default)]
    pub task_count: usize,
    #[serde(rename = "withLaw", default)]
    pub with_law: usize,
    #[serde(rename = "withDomain", default)]
    pub with_domain: usize,
    pub tasks: Vec<LawTask>,
}

impl Laws {
    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("lecture {} : {}", path.display(), e))?;
        serde_json::from_str::<Laws>(&raw).map_err(|e| format!("JSON {} : {}", path.display(), e))
    }

    pub fn get(&self, id: &str) -> Option<&LawTask> {
        self.tasks.iter().find(|t| t.id == id)
    }
}

/// Chemin par défaut du fichier de lois exporté.
pub fn laws_path(root: &Path) -> PathBuf {
    root.join("spear-rs").join("data").join("laws.json")
}
