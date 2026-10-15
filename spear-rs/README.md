# spear-rs — port Rust du moteur SPEAR

Le registre SPEAR trouve des approximations algébriques de lois chères (GELU, SiLU,
erf, RoPE, KV-cache…). Ce crate **exécute** ces formules en Rust, les **émet** sous
forme de kernels drop-in, et les **chronomètre**. La recherche, elle, reste en
TypeScript : Rust sert à exécuter et à mesurer, pas à trouver.

```bash
cd spear-rs
cargo build --release
.\target\release\spear.exe list
```

## Ce que ça fait

| commande | rôle |
|---|---|
| `spear list [--fast]` | vue d'ensemble du ledger : variables, coût champion, coût fast, gain modèle, mesure déjà enregistrée |
| `spear show <id> [--fast]` | formule, coût détaillé par op, métriques ledger, mesure GCC, **et** le kernel Rust émis |
| `spear emit <id> [--fast] [--f32]` | kernel Rust drop-in sur stdout : scalaire `#[inline(always)]` + passe fusionnée `*_into` |
| `spear emit <id> [--fast] --triton` | kernel **Triton** (Python, GPU NVIDIA) : fonction élémentaire `@triton.jit` + kernel de lancement masqué |
| `spear eval <id> --set g=1.5 --set u=2` | valeur ponctuelle ; sans `--set`, échantillonne la grille |
| `spear verify [<id>...]` | RMSE / erreur max de chaque champion **contre sa loi génératrice** |
| `spear audit [--strict]` | recalcule les coûts depuis les AST et signale les fast slots qui ne sont pas moins chers (exit 1) |
| `spear bench [<id>...] [--md f]` | ns/élément pour tout le registre (90 tâches), tableau markdown optionnel |
| `spear emit-bench [<id>...] [--f32]` | compile les kernels **émis** avec `rustc -O` et les chronomètre |

`--f32` bascule tout le kernel, helpers de bornage compris : c'est le format des
backends GPU/Metal, donc le format à viser pour TensorFold/MLX. Les constantes
sont arrondies à 9 décimales (précision f32), la structure de l'arbre est
inchangée, et `emitted_f32_rust_compiles_and_tracks_vm` le compile avec `rustc`
pour vérifier qu'il suit la VM à la précision f32 près.

`--triton` émet la même formule pour GPU NVIDIA : `spear_<id>` (fonction
élémentaire réutilisable dans n'importe quel `@triton.jit`) plus
`spear_<id>_kernel`, un kernel de lancement complet (chargements masqués,
calcul, stockage masqué). Les bornages sont transcrits en rails `@triton.jit`
(`_spear_pdiv`, `_spear_exp`, `_spear_log`, `_spear_erf`…), avec `libdevice`
pour `atan/asin/acos/tanh/erf`. Triton n'étant installé nulle part dans
l'environnement de test, le contrat vérifié par `emitted_triton_parses_as_python`
est la **génération** : les 145 kernels du registre (champions + fast slots)
sont émis puis passés à `ast.parse`. L'exécution GPU demande une machine
NVIDIA — c'est testé, pas supposé, et pas revendiqué.

## Le kernel émis

```rust
let kernel = spear emit silu --fast;
```

produit, en plus du préambule de bornages :

```rust
#[inline(always)]
pub fn spear_silu(x: f64) -> f64 { … }

#[inline]
pub fn spear_silu_into(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), out.len());
    for (row, o) in out.iter_mut().enumerate() { *o = spear_silu(x[row]); }
}
```

`silu --fast` = `1.016356·relu(x) − 0.15849`. Zéro allocation, zéro branche (`.max`/
`.min`), une écriture par élément — c'est ça qui remplace utilement un graphe
`evaluateNode` qui alloue un buffer par nœud. Pour une loi multi-variable, la signature
prend une colonne par variable (`spear_swiglu_fused(g: f64, u: f64) -> f64`) et
`*_into` prend un slice par variable.

Les bornages sont **ceux du moteur TypeScript**, pas des simplifications :

| bornage | règle TS | effet |
|---|---|---|
| `spear_pdiv` | dénominateur flooré à ±1e-4 (signe conservé), résultat borné ±1e4 | les formules découvertes s'appuient dessus |
| `spear_exp` | entrée clampée ±50 | évite `inf` |
| `spear_log` | plancher `1e-30` | idem |
| `spear_asin` / `spear_acos` | entrée bornée ±1 | idem |
| `spear_sqrt` | `sqrt(|x|)` | le moteur ne distingue pas le signe |
| `spear_erf` | Abramowitz & Stegun 7.1.26, mêmes constantes **et même ordre** | parité bit-à-bit avec `math-utils.ts` |

## Résultats

Deux mesures, à ne pas confondre.

### 1. `spear bench` — les 90 tâches, VM postfix

Machine : Windows, build `release` (`opt-level=3`, `lto=thin`, `codegen-units=1`).
Méthodologie : une passe = 4096 éléments/variable, 1 chauffe puis 9 passes de 8
répétitions, **médiane** conservée, résultat lu dans un `black_box`.

* 64 tâches ont un fast slot ; les gains médians sont bons mais **6 fast slots
  sont mesurés plus lents** que leur champion → section dédiée dans
  [`BENCH.md`](BENCH.md), avec le seuil de bruit à 0,90 explicite.

**Attention à la lecture.** Ces chiffres mesurent la **VM interpréteuse**, pas le
kernel compilé : le dispatch `match` sur l'op domine. Ils servent à *classer* des
formules entre elles, pas comme performance absolue.

### 2. `spear emit-bench` — le kernel compilé, 2²⁰ éléments

Machine au repos, `rustc -O`, meilleur de 9 passes :

| tâche | champion ns/él | fast ns/él | gain compilé |
|---|---:|---:|---:|
| `silu` | 9,36 | **1,06** | ×8,8 |
| `gelu` | 11,52 | **1,01** | ×11,4 |
| `sigmoid` | 9,43 | **1,27** | ×7,4 |
| `tanh_sat` | 17,98 | **1,06** | ×17,0 |
| `erf_prob` | 10,42 | **1,20** | ×8,7 |
| `swiglu_fused` (2 var) | 9,30 | **1,39** | ×6,7 |
| `rope_rot` (3 var) | 17,49 | 17,57 | ×1,0 |
| `bilateral_weight` | 10,37 | 15,90 | ×0,65 |

Recoupement avec les mesures déjà enregistrées dans le ledger (gcc `-O2`,
scalaire) : SiLU fast **2,73 ns** (TS) contre **1,06 ns** ici, GELU fast **3,1 ns**
contre **1,01 ns**. Deux compilateurs, deux harnais, même ordre de grandeur : les
deux ports sont cohérents. Reproduire avec `spear emit-bench silu gelu sigmoid`.

`rope_rot` et `bilateral_weight` illustrent le piège : leur fast slot est soit la
loi elle-même, soit une forme qui garde des transcendantes. Le modèle de coût les
déclare moins chers, la machine dit le contraire.

## Audit : ce que le port Rust a trouvé

`spear audit --strict` recalcule le coût de chaque champion et de chaque fast slot
depuis l'arbre, puis le compare à ce que le ledger déclare. Il a détecté
indépendamment les 7 `fast-not-faster` de `scripts/audit-champions.ts` **et** le
détail du drift : 8 tâches avaient un `fast.formulaCost` stocké qui ne correspondait
pas à son AST (`erf_prob` : stocké 1, recalculé 20).

Cause racine côté TS : `scripts/backfill-fast.ts:47-51` ressuscitait des entrées
historiques quand `e.speed.formulaCost < cur.fast.formulaCost`, et les anciennes
entrées portant une loi exacte à primitive directe avaient un `formulaCost` erroné
(1) — donc la loi exacte écrasait un vrai fast slot algébrique. Corrigé depuis : le
script recalcule coût, métrique, niveau et texte depuis l'AST, et refuse tout
candidat qui n'est pas strictement moins juste que le champion. Les deux audits
sont à **0 échec**.

Le port Rust n'a pas ce problème par construction : il ne lit jamais le coût
**stocké** pour décider, il le recalcule à chaque fois.

## Fidélité : ce qui est vérifié par les tests

`cargo test --release` — 8 tests, dont :

* `vm_matches_recursive_tree_eval` — la VM postfix rend la même valeur que la
  récursion sur **tout** le ledger.
* `vector_path_matches_scalar_path` — le chemin vectoriel rend exactement le
  scalaire.
* `champion_cost_matches_ledger_cost` — le modèle de coût Rust redonne les coûts
  stockés (aucune dérive).
* `emitted_rust_compiles_and_matches_vm` — `rustc` compile réellement le kernel émis
  (champion, fast, erf, multi-variable, 2-variable) et la valeur affichée est
  identique à celle de la VM. C'est le test qui prouve que l'émission est
  fiable, pas « ça a l'air correct ».
* `emitted_f32_rust_compiles_and_tracks_vm` — même chose en f32, bornages inclus,
  avec une tolérance f32. C'est le chemin GPU/Metal, donc il est testé.
* `erf_matches_abscissa_stegun_reference`, `swiglu_law_is_two_variable_and_correct`,
  `exported_laws_cover_most_tasks`, `every_champion_stays_finite_on_probe_grid`.

## Limites assumées

* **Pas de recodage du GP.** Le moteur de découverte reste en TS ; ce crate ne fait
  pas de recherche.
* **Pas de JIT ni de SIMD explicite.** La VM est scalaire. Le chemin qui compte pour
  le déploiement est le kernel émis, que le compilateur vectorise tout seul ; la SIMD
  explicite (AVX-512 portable, f32) serait un chantier séparé.
* **6 tâches sans cible** : `kv_cache` (policy), `implied_vol` (solveur),
  `blackbody_b`, `blackbody_g`, `damped_pendulum`, `pendulum_hybrid` (intégrateurs)
  n'ont pas de forme close, donc `verify` les saute **et le dit** — aucune cible
  inventée.
* **Les métriques du ledger ne sont pas recalculables ici.** `metric` est un MSE sur
  la grille d'entraînement TS ; le `RMSE` de `verify` est mesuré sur la grille de
  sonde de spear-rs. Ce sont deux mesures différentes, rapportées côte à côte.
* **Le `domain` déclaré n'existe que pour 40 tâches sur 90.** Les autres utilisent
  des bornes par défaut heuristiques (`measure::default_span`), qui ne servent qu'au
  timing. La borne effective est rappelée par la commande (`--rows`, `spear show`).
* **Bruit de mesure.** Pas d'épinglage de cœur ni de fréquence fixe : les écarts
  inférieurs à ~20 % ne sont pas significatifs. Les gains médians (×1,86) le sont ;
  les classements fin, non.

## Régénérer les données

```bash
# 1. rejouer les lois + domaines depuis les définitions vivantes des tâches
npx tsx scripts/export-rust-data.ts

# 2. rejouer le benchmark et le rapport
cd spear-rs && cargo build --release
.\target\release\spear.exe bench --md BENCH.md
.\target\release\spear.exe emit-bench
```

`scripts/export-rust-data.ts` ne fait **rien d'interprétatif** : il sérialise
`TaskDef.exactRefNode` (la loi génératrice en AST) et `TaskDef.domain`. Si une tâche
n'a pas de loi, l'entrée vaut `null` et `verify` la saute.

## Dépendances

`serde` + `serde_json`. Pas de `clap` (parsing manuel, 7 sous-commandes), pas de
`criterion` (un tableau sur 90 tâches en une commande vaut mieux que 180 rapports
HTML), pas de `rand` (LCG déterministe dans les tests). Le reste est `std` : la
fidélité numérique et la Reproductibilité prime sur le confort.
