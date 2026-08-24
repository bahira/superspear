# Journal des correctifs d'ingénierie

> Bugs trouvés par refus d'accepter un mur ou une incohérence. Chacun est
> adossé à un commit et un test/audit de non-régression.

## CRITIQUES (silencieux, corrompaient des résultats)

### 1. Effondrement NaN dans simplify() — `(c·x)·Y`
- **Symptôme** : eigen3_sym bloqué à 3.14e-2 malgré scaffold parfait ; formes GP disparaissant mystérieusement.
- **Root cause** : les règles `c1·(c2·x) → (c1·c2)·x` lisaient `.value` du sibling NON-constante → `undefined` → NaN → sous-arbre réduit à néant.
- **Fix** : gardes `kids[1].op === "const"` obligatoires. Commit `d2240ba`.
- **Découverte via** : bissection empirique après avoir vu simplify réduire 124→1 unités.
- **Portée** : ce bug mangeait potentiellement des formes depuis l'origine.

### 2. Séries Bessel de référence fausses
- **Symptôme** : bessel_j0/j1 « boss fight » impossible, MSE plancher élevé.
- **Root cause** : transcription A&S/NR bâclée — J0 renvoyait 0.457 en x=1 (vrai : 0.765) ; J1 dénominateur faux dès k=2. **Les données cibles étaient empoisonnées** : le GP approximait une courbe mystère.
- **Fix** : séries convergentes vraies, vérifiées à la main (J0(1)=0.7656 ✓). Commit `1b6f57b`.
- **Résultat** : murs effondrés — j0 ×60, j1 ×108,000.
- **Leçon** : *mur persistant sur tâche jeune = suspect n°1 le générateur de données.*

### 3. Types d'imports WASM erronés
- **Symptôme** : tout module multi-variables avec sin/cos/exp refusait de compiler (« not enough arguments on the stack »).
- **Root cause** : imports déclarés avec le type de la fonction principale (n params) au lieu de (f64)→(f64). Mono-variable tombait juste par hasard.
- **Fix** : type dédié index 1. Commit inclus dans `b25ec3f`.

### 4. Division protégée absente du codegen
- **Symptôme** : parité C↔JS fausses jusqu'à ×10 exact (kdv 1.4, kerr_spin 0.9).
- **Root cause** : évaluateur JS protège (plancher ±1e-4 + clamp ±1e4), codegen émettait division nue ; les champions exploitent ces rails (gaussian_cdf plateau).
- **Fix** : copysign/fmaxf en C, séquence clamp en WASM (opcode copysign 0xa6 — premier essai 0x4c = i32.le_s, corrigé). Commits `b25ec3f`, `d2240ba`.

## MAJEURS

### 5. Garde log non unifiée entre backends
- JS plafonnait à 1e-300, C/WASM à 1e-30 → décalage d'une décade exacte (×10 mantisse-préservée) sur les formes log-guard-firing. Fixé à 1e-30 partout (`74774ab`).

### 6. Race condition run-farm
- Deux fermes simultanées = last-writer-wins écrasant les records. Verrou `.farm-lock` single-writer ajouté (`f51cb6d`).

### 7. parseFormula sans précédence
- Le parser n'acceptait que les formes entièrement parenthésées du ledger ; input utilisateur libre cassait. Montée en précédence terme/expr ajoutée (`ddec7b9`).

### 8. Ledger pollué par clés-titres
- Entrées fantômes (« Prime d'un call europée »…) créées par un vieux bug. Nettoyage + garde regex `[a-z0-9_]` + script `clean-ledger.ts` (`b5e6e07`).

## MINEURS
- padEnd JavaScript dans du Python (kv_real_validate) — ljust.
- torch.minimum scalaire/tenseur (gelu généré) — clamp(max=).
- Assertion test-simplify cherchant le mauvais token (pdiv s'imprime `/`).
