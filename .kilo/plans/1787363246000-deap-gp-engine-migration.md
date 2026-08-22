# Migrate trade_discovery GP Engine from gplearn to DEAP

## Context

The project evolves trading formulas with genetic programming. Today it wraps
gplearn `SymbolicRegressor` (`trade_discovery/project_core/src/gp_engine.py`)
and relies on a fragile **on-disk source patch** of installed gplearn
(`src/gplearn_prior_patch.py`) to fake feature-prior terminal sampling, which
stock gplearn lacks. The outer Global Evolutionary Loop (GEL) lives in
`scripts/main_pipeline.py`.

### What depends on gplearn today
| Site | Usage |
|---|---|
| `src/gp_engine.py` | `SymbolicRegressor`, `make_function`, `make_fitness`; warm-start 3-phase schedule; seed injection into `est._programs[-1]`; anti-convergence micro-GP |
| `src/gplearn_prior_patch.py` | Regex-patches installed gplearn source (v4 marker scheme) to honour `_feature_proba` |
| `scripts/main_pipeline.py` | `str(gp._program)` formula strings parsed by regex guards; `len(gp._program.program)` bloat check; `gp.predict()`; winners store `deepcopy(gp._program)` as next-gen seeds |
| `scripts/verify_fixes.py` | Functional test of the prior patch; checks `gplearn==0.4.3` pin |
| `scripts/requirements.txt`, setup notebook | Pin `gplearn==0.4.3` |

Downstream (`Audit_scripts/vectorbt_evaluator.py`, both
`formula_leaderboard.py` copies) consume **only** `.predict()` and formula
strings via word-boundary regexes — they never touch gplearn internals.

## Decisions (agreed with user)

1. **Full refactor** — `main_pipeline.py` interactions move to native DEAP
   idioms (not just a shim behind the old API).
2. **Optional `n_jobs`** — serial by default (each eval is one vectorized
   numpy pass over the fold matrix); joblib-based parallel map available.
3. **Clean cut** — delete `gplearn_prior_patch.py`, drop the `gplearn` pin,
   update requirements + notebook. No dual-engine switch; old gplearn-pickled
   winner programs are NOT migrated (a fresh GEL run seeds from scratch).

### Key technical constraint: formula string compatibility
`main_pipeline.py` guards (`_features_used`, `_find_trivial_cancellation`,
bool-ratio ratio check) and both leaderboards parse formula STRINGS like
`add(mul(feat_a, feat_b), sub(feat_c, const))`. The DEAP engine MUST emit the
same Lisp-style format:
- Register primitives under the exact current names: `add, sub, mul, div,
  max, min, abs, neg, gt, lt, and, or, if_then`.
- `pset.renameArguments(...)` so argument terminals print real feature names.
- Ephemeral constants must stringify as bare numbers (custom `str.format`
  handling), mirroring gplearn's constant rendering.
- Note: `_find_trivial_cancellation` strips spaces before parsing and the
  bool-ratio/cancellation regexes are `\b`-anchored, so minor whitespace
  differences between DEAP's `str()` and gplearn's are harmless. Hash dedup
  (`hash_formula`) is internally consistent either way.

## Task list

### 1. Dependency updates
- [ ] `scripts/requirements.txt`: replace `gplearn==0.4.3` with a pinned
  `deap==1.4.*` (verify latest 1.4.x available); keep other pins.
- [ ] Setup notebook `project_core/gplearn.ipynb`: update the `_have(...)`
  install cell (`gplearn==0.4.3` → `deap`). Do NOT uninstall gplearn from the
  environment (notebook policy says libraries are never removed).

### 2. Rewrite `src/gp_engine.py` on DEAP
- [ ] **Primitive set**: `gp.PrimitiveSet("MAIN", n_features)` +
  `renameArguments` to feature names. Vectorized numpy implementations:
  - `_protected_div` (denominator floor 0.001, port as-is)
  - `gt`/`lt` soft comparisons (`tanh((x1-x2)*GT_SOFT_SCALE)`),
    `and`=`np.minimum`, `or`=`np.maximum`, `if_then`=`np.where(c>0,t,f)`
  - `abs`, `neg`, `max`, `min` mapped to numpy ufuncs.
- [ ] **Ephemeral constant**: `gp.Ephemeral("const", lambda: uniform(-1,1))`
  to reproduce gplearn's `const_range=(-1,1)` behaviour; ensure it stringifies
  as its numeric value.
- [ ] **Individual & fitness**: `creator.create("FitnessDir",
  base.Fitness, weights=(1.0,))`; `creator.create("Individual",
  gp.PrimitiveTree, fitness=..., feature_prior=..., n_features=...)`.
- [ ] **Native feature prior** (replaces the whole disk-patch hack):
  - Prior-aware full-tree generator: ramped half-and-half where every
    variable-terminal draw uses `np.random.choice(n_features, p=proba)`
    (fallback uniform when no prior).
  - Prior-aware subtree/point mutation terminal sampling via the same helper.
- [ ] **Operators**, registered on a `Toolbox`:
  - Mate: `gp.cxOnePoint`, wrapped in `gp.staticLimit(key=height,
    max_value=regime_depth_max)` and node-count limit 40 (`MAX_PROGRAM_LENGTH`).
  - Mutate: prior-aware subtree mutation (new tree depth ≤ regime max),
    custom hoist mutation (port gplearn's hoist semantics: pick a random
    subtree, lift it to root), prior-aware point mutation, ephemeral mutation.
  - Select: `tools.selTournament(tournsize=regime["tournament_size"])`.
- [ ] **Fitness evaluation** — exact port of `_directional_fitness`:
  Pearson term + harmonic direction-coverage term (weights from regime HP,
  incl. `fitness_pearson_w/direction_w` overrides) −
  `ATTR_PENALTY_WEIGHT × barrier_penalty` − `SPREAD_WEIGHT × spread_penalty`;
  near-constant predictor guard (`yp_std < 1e-9 → pearson_r=0`).
- [ ] **Custom evolve loop** (replaces `eaSimple`; this is what kills the
  warm-start phase hack):
  - Per generation: bagging — sample `GP_MAX_SAMPLES=0.9` fraction of rows
    (seeded RNG) for fitness evaluation.
  - Score = raw fitness − `parsimony_coefficient(phase) × len(ind)`.
  - Phases: gens `[0,P1)` use `parsimony_p1`, `[P1,P1+P2)` `parsimony_p2`,
    rest `parsimony_p3` (from regime HP / config; assert sum == GENERATIONS).
  - μ+λ elitism: top-μ parents carried forward unchanged each generation;
    offspring fill the rest via mate/mutate probs (`p_crossover` from regime,
    remainder split subtree/h oist/point like today's normalization block —
    port the probability-normalization logic).
  - Seeded runs: inject `elite_pool[:SEEDS_PER_GEN]` individuals into the
    initial population (exactly once each — keep FIX #6 semantics), pad with
    anti-convergence randoms (`ANTI_CONVERGENCE_FRACTION`), rest random.
    Sanitize any stale-terminal indices (n_features mismatch) as today.
  - Track per-generation best + final population for elite extraction.
- [ ] **Public surface** (used by main_pipeline):
  - `train_gp_model(X_train, y_train, seed_programs=None, fold=0,
    feature_proba=None, regime=None, df_raw_train=None, n_jobs=1)` → returns a
    `GPFoldResult` object exposing: `.predict(X)` (compiled best program),
    `.best_program` (DEAP individual), `.best_formula` (`str`),
    `.population` (final list of individuals), `.best_length`. Keep returning
    `None` when the bloat guard rejects the winner (preserve contract).
  - `extract_elite_programs(result_or_population, top_n=None,
    max_duplicates=2, diversity_fraction=0.30)` → port hard-dedup +
    incremental farthest-first trigram-Jaccard selection verbatim, operating
    on DEAP individuals.
  - `hash_formula(str)` unchanged.
  - Anti-convergence randoms become plain `toolbox.individual()` draws
    (delete `_create_random_programs` micro-GP).
  - Post-hoc trade-sim selection path stays behind
    `FITNESS_POSTHOC_TRADE_EVAL=False` (config default); adapt it to accept
    any object with `.predict` (already does via wrapper).
  - Regime HP resolution (`_get_regime_hp`, REGIME_HP map incl.
    `depth_max`, `tournament_size`, `p_crossover`, parsimony, fitness
    weights) ported unchanged.
- [ ] **Parallelism**: `n_jobs` param; when >1, evaluate raw fitness across
  the population with `joblib.Parallel` (workers pickle X once per batch);
  serial fallback default. Log a warning about memory when n_jobs>1.

### 3. Refactor `scripts/main_pipeline.py` to native idioms
- [ ] Replace all `gp._program` accesses: `formula_str = gp.best_formula`,
      `program_len = gp.best_length` (drop the `hasattr(gp._program,'program')`
      dance).
- [ ] Winners dict `'program'` stores `copy.deepcopy(gp.best_program)` (DEAP
      `PrimitiveTree` deep-copies cleanly); it is dropped from the parquet as
      today, only used in-process for seeding.
- [ ] `elite_pool` becomes `List[Individual]` — passes straight back into
      `train_gp_model(seed_programs=...)`.
- [ ] Structural/signal guards, OOS signature logic, stale-reset, feature
      prior construction (Dirichlet smoothing, concentration cap) — unchanged;
      they operate on strings/counters already.
- [ ] Pass `n_jobs` through from a new config key `GP_N_JOBS` (default 1).

### 4. Delete gplearn artifacts
- [ ] Remove `src/gplearn_prior_patch.py`.
- [ ] Remove the patch-bootstrap block at the top of `gp_engine.py`
      (`FEATURE_PRIOR_ACTIVE` machinery).

### 5. Rewrite `scripts/verify_fixes.py`
- [ ] Replace the prior-patch functional test with a native-DEAP version:
      fit tiny pop twice (uniform vs skewed prior) and assert terminal-share
      skew — same assertion shape as today's `terminal_shares`.
- [ ] Keep import-bootstrap, tokenization, cancellation-detector, config
      wiring, keep-pool-on-crash static tests (they don't touch gplearn).
- [ ] Replace the `gplearn==0.4.3` pin check with `deap` pin check.
- [ ] Add a smoke test: 2-phase mini GP run (pop ~200, ~4 gens) on synthetic
      data asserting: best program exists, `str(best)` parses through
      `_features_used`-style regex, predict returns finite vectors of right
      shape, seeded run accepts injected individuals.

### 6. Config additions (`src/config.py`)
- [ ] `GP_N_JOBS = 1` (documented: >1 enables joblib fitness parallelism).
- [ ] Leave all existing GP_*/GEL_*/REGIME_HP keys untouched — they carry
      over semantically (population size, generations, phases, depths,
      priors, tournament sizes, crossover probs, parsimony schedule).

## Risks / mitigations
- **Formula-format drift** breaks guards/leaderboards → mitigated by naming
  primitives identically + renaming arguments; verified by a dedicated
  verify_fixes check running real guard functions over generated formulas.
- **Performance parity**: gplearn ran `n_jobs=-1` over 3000×60 evolutions;
  serial DEAP may be slower per-gen. Mitigation: bagged vectorized eval +
  optional n_jobs; measure wall-clock/gen during validation before a long
  GEL run.
- **RNG reproducibility** changes by definition (different engine); keep
  `random_state=fold` seeding discipline via `random.seed`/`np.random.seed`
  per run and DEAP's per-toolbox `random` module usage.
- **Ephemeral-constant stringification**: DEAP may render lambdas/values
  inconsistently → add explicit formatting test in verify_fixes.

## Validation plan
1. `pip install deap` (or notebook cell), then
   `python scripts/verify_fixes.py` — all checks PASS.
2. Smoke GEL run on the real dataset with temporarily reduced
   `GEL_GENERATIONS=3`, `GP_POPULATION_SIZE=300`, `GP_GENERATIONS=6`
   (config override or env): confirm end-to-end — training, structural
   guards accepting/rejecting formulas, holdout VectorBT eval, winner log,
   elite extraction, seeded second generation.
3. Confirm `outputs/winning_formulas_gel.log` formula strings still parse
   with both `formula_leaderboard.py` copies'
   `extract_features_from_formula`.
4. Wall-clock per generation comparison note vs previous engine (serial),
   recorded in the run log; decide `GP_N_JOBS` for production runs.

## Out of scope
- Migrating historical outputs (`winning_formulas_gel.parquet` etc.) —
  string-based artifacts remain readable; pickled programs are abandoned.
- Any change to target generation, feature engineering, regime classifier,
  VectorBT evaluation, or survivor gates.
