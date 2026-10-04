//! `spear-rs` — port Rust du moteur SPEAR.
//!
//! ## Ce que fait ce crate
//!
//! * [`ir`] : les 22 opps de l'IR SPEAR, leur coût ALU/SFU et le désérialiseur
//!   du format ledger (`{"o": ..., "c": [...]}`) — aucun format propriétaire.
//! * [`eval`] : ré-exécution **bit-faible** des rails du moteur TS (`pdiv` borné
//!   à ±1e-4/±1e4, `exp` clampé ±50, `log` plancher 1e-30, `asin`/`acos` bornés,
//!   `sqrt(|·|)`, `erf` A&S 7.1.26), en scalaire et en VM postfix vectorielle.
//! * [`emit`] : emission de kernels Rust drop-in (scalaire `#[inline(always)]` +
//!   variante fusionnée `*_into(&[f64], &mut [f64])` en une passe).
//! * [`ledger`] : lecture du bundle `spear-hall-of-fame.json` / `ledger/*.json`
//!   et du fichier de lois exporté `data/laws.json`.
//! * [`measure`] : grille de sonde, erreurs contre la loi, timing ns/élément,
//!   audit de coût (le miroir Rust des drapeaux de `scripts/audit-champions.ts`).
//!
//! ## Ce que ce crate ne fait PAS (et pourquoi)
//!
//! * Il ne rejoue pas la recherche génétique : le GP est en TypeScript, il
//!   reste là. Rust sert à **exécuter** les formules trouvées, pas à les trouver.
//! * Il ne réimplémente pas les 90 constructeurs de datasets : les cibles
//!   viennent des lois exportées (`data/laws.json`), 84 tâches sur 90. Les 6
//!   restantes n'ont pas de forme close (solveurs, intégrateurs, politique KV) et
//!   `spear verify` le dit plutôt que d'inventer une cible.
//! * Les métriques du ledger (entraînées sur les grilles TS) ne sont pas
//!   recalculables ici : `verify` mesure l'erreur contre la loi sur sa propre
//!   grille de sonde, qui est une **mesure différente**. Les deux sont
//!   rapportées côte à côte, jamais confondues.
//!
//! ## Fidélité numérique
//!
//! L'ordre des opérations, les constantes et les bornages sont copiés de
//! `src/lib/spear/engine.ts` (`evaluateNode`) et `src/lib/spear/math-utils.ts`
//! (`erf`). `tests/parity.rs` vérifie VM == arbre récursif sur tout le ledger,
//! et compile/exécute le Rust émis pour prouver l'égalité des deux chemins.

pub mod emit;
pub mod eval;
pub mod ir;
pub mod ledger;
pub mod measure;
