import copy
import hashlib
import logging
import numpy as np
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicRegressor

try:
    import src.config as config
    POPULATION_SIZE = getattr(config, "GP_POPULATION_SIZE", 3000)
    SEED_FRACTION   = getattr(config, "GP_SEED_FRACTION", 0.20)
    MUTATION_BOOST  = getattr(config, "GP_MUTATION_BOOST", 0.15)
    GENERATIONS     = getattr(config, "GP_GENERATIONS", 60)
    PHASE1_GENS     = getattr(config, "GP_PHASE1_GENS", 10)
    PHASE2_GENS     = getattr(config, "GP_PHASE2_GENS", 30)
    PHASE3_GENS     = getattr(config, "GP_PHASE3_GENS", 20)
    INIT_DEPTH_MIN  = getattr(config, "GP_INIT_DEPTH_MIN", 4)
    INIT_DEPTH_MAX  = getattr(config, "GP_INIT_DEPTH_MAX", 8)
    TOURNAMENT_SIZE = getattr(config, "GP_TOURNAMENT_SIZE", 100)
    PEARSON_WEIGHT   = getattr(config, "FITNESS_PEARSON_WEIGHT", 0.70)
    DIRECTION_WEIGHT = getattr(config, "FITNESS_DIRECTION_WEIGHT", 0.30)
    HOIST_MUTATION   = getattr(config, "GP_HOIST_MUTATION", 0.1)
    POINT_MUTATION   = getattr(config, "GP_POINT_MUTATION", 0.1)
    MAX_SAMPLES      = getattr(config, "GP_MAX_SAMPLES", 0.7)
    GT_SOFT_SCALE    = getattr(config, "GT_SOFT_SCALE", 3.0)
except ImportError:
    POPULATION_SIZE = 3000
    SEED_FRACTION   = 0.15
    MUTATION_BOOST  = 0.15
    GENERATIONS     = 60
    PHASE1_GENS     = 10
    PHASE2_GENS     = 30
    PHASE3_GENS     = 20
    INIT_DEPTH_MIN  = 4
    INIT_DEPTH_MAX  = 8
    TOURNAMENT_SIZE = 100
    PEARSON_WEIGHT   = 0.70
    DIRECTION_WEIGHT = 0.30
    HOIST_MUTATION   = 0.1
    POINT_MUTATION   = 0.1
    MAX_SAMPLES      = 0.7
    GT_SOFT_SCALE    = 3.0

assert PHASE1_GENS + PHASE2_GENS + PHASE3_GENS == GENERATIONS, (
    f"Phase gens sum ({PHASE1_GENS}+{PHASE2_GENS}+{PHASE3_GENS}) "
    f"!= GP_GENERATIONS ({GENERATIONS}). Fix config.py."
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FIX-GP-2: REGIME → HYPERPARAMETER MAP
# ─────────────────────────────────────────────────────────────────────────────

try:
    _REGIME_HP = getattr(config, "REGIME_HP", {})
except NameError:
    _REGIME_HP = {
        # Strong trend: prefer longer trees; aggressive crossover for recombination
        "trending": {
            "parsimony_p1":   0.0,
            "parsimony_p2":   0.0002,
            "parsimony_p3":   0.002,
            "p_crossover":    0.70,
            "depth_max":      8,
            "tournament_size": 100,
        },
        # Mean-revert: hard complexity penalty — short precise rules generalise better
        "mean_reverting": {
            "parsimony_p1":   0.001,
            "parsimony_p2":   0.003,
            "parsimony_p3":   0.008,
            "p_crossover":    0.55,
            "depth_max":      6,
            "tournament_size": 75,
        },
        # Choppy: strongest length penalty + small tournaments → diversity pressure
        "choppy_random_walk": {
            "parsimony_p1":   0.002,
            "parsimony_p2":   0.005,
            "parsimony_p3":   0.010,
            "p_crossover":    0.50,
            "depth_max":      5,
            "tournament_size": 60,
        },
        # Trending but noisy: balanced
        "trending_random_walk": {
            "parsimony_p1":   0.0,
            "parsimony_p2":   0.001,
            "parsimony_p3":   0.004,
            "p_crossover":    0.65,
            "depth_max":      7,
            "tournament_size": 100,
        },
        # Fallback / uncertain
        "random_walk": {
            "parsimony_p1":   0.0,
            "parsimony_p2":   0.0005,
            "parsimony_p3":   0.003,
            "p_crossover":    0.60,
            "depth_max":      8,
            "tournament_size": 100,
        },
    }

_DEFAULT_REGIME_HP = _REGIME_HP.get("random_walk", {})


def _get_regime_hp(regime: str | None) -> dict:
    """Return hyperparameter overrides for the given regime label."""
    if regime is None:
        return _DEFAULT_REGIME_HP
    hp = _REGIME_HP.get(regime)
    if hp is None:
        logger.warning("Unknown regime '%s' — falling back to random_walk HP.", regime)
        return _DEFAULT_REGIME_HP
    return hp


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM LOGICAL & COMPARISON OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _gt_soft(x1, x2):  return np.tanh((x1 - x2) * GT_SOFT_SCALE)
def _lt_soft(x1, x2):  return np.tanh((x2 - x1) * GT_SOFT_SCALE)
def _and(x1, x2):      return np.minimum(x1, x2)
def _or(x1, x2):       return np.maximum(x1, x2)
def _if_then(c, t, f): return np.where(c > 0.0, t, f)

def _protected_div(x1, x2):
    """Division with denominator floor at 0.001 to prevent overflow."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, x1 / x2, np.ones_like(x1))

greater_than  = make_function(function=_gt_soft, name='gt',      arity=2)
less_than     = make_function(function=_lt_soft, name='lt',      arity=2)
logical_and   = make_function(function=_and,     name='and',     arity=2)
logical_or    = make_function(function=_or,      name='or',      arity=2)
if_then       = make_function(function=_if_then, name='if_then', arity=3)
protected_div = make_function(function=_protected_div, name='div', arity=2)

TRADING_FUNCTIONS = [
    'add', 'sub', 'mul', protected_div,
    'max', 'min', 'abs', 'neg',
    greater_than, less_than, logical_and, logical_or, if_then
]

# ---------------------------------------------------------------------------
# CUSTOM FITNESS METRIC
# ---------------------------------------------------------------------------

def _directional_fitness(y, y_pred, w):
    """
    Custom GP fitness: Pearson correlation gated by minimum directional coverage.
    Penalizes formulas that produce one-sided signals.

    Returns: float in [-1, 1], higher is better (gplearn maximizes by default)
    """
    EPS = 1e-8

    y_mean    = np.mean(y)
    yp_mean   = np.mean(y_pred)
    num       = np.sum((y - y_mean) * (y_pred - yp_mean))
    denom     = (np.std(y) * np.std(y_pred) * len(y)) + EPS
    pearson_r = num / denom

    long_mask  = (y == 1.0)
    short_mask = (y == -1.0)

    n_long_pred  = np.sum(y_pred[long_mask]  > 0) if np.any(long_mask)  else 0
    n_short_pred = np.sum(y_pred[short_mask] < 0) if np.any(short_mask) else 0
    n_long_true  = np.sum(long_mask)
    n_short_true = np.sum(short_mask)

    long_coverage  = n_long_pred  / (n_long_true  + EPS)
    short_coverage = n_short_pred / (n_short_true + EPS)

    dir_score = (2 * long_coverage * short_coverage) / (long_coverage + short_coverage + EPS)

    return float(PEARSON_WEIGHT * pearson_r + DIRECTION_WEIGHT * dir_score)


directional_metric = make_fitness(function=_directional_fitness, greater_is_better=True)


# ---------------------------------------------------------------------------
# POPULATION SEED UTILITIES
# ---------------------------------------------------------------------------

def hash_formula(program_str: str) -> str:
    """SHA-256 fingerprint of a formula string for deduplication."""
    return hashlib.sha256(program_str.encode()).hexdigest()


def extract_elite_programs(fitted_gp: SymbolicRegressor, top_n: int = None) -> list:
    """
    Extract deep-copies of the top-N fittest _Program objects from the
    last generation of a fitted SymbolicRegressor.
    """
    if not hasattr(fitted_gp, '_programs') or not fitted_gp._programs:
        logger.warning("No _programs found on fitted model — returning empty seed list.")
        return []

    last_gen = fitted_gp._programs[-1]
    top_n    = top_n or int(SEED_FRACTION * POPULATION_SIZE)

    valid = [p for p in last_gen if p is not None and hasattr(p, 'fitness_')]
    elite = sorted(valid, key=lambda p: p.fitness_, reverse=True)[:top_n]

    logger.info("Extracted %d elite programs (pool size=%d).", len(elite), len(valid))
    return [copy.deepcopy(p) for p in elite]


# ---------------------------------------------------------------------------
# FIX-GP-1: FEATURE PROBA SURVIVAL HELPER
# ---------------------------------------------------------------------------

def _apply_feature_proba(
    est_gp: SymbolicRegressor,
    feature_proba: np.ndarray | None,
    feature_names: list,
    fold: int,
) -> None:
    """
    Re-apply the feature sampling prior on `est_gp` before every fit() call.

    WHY THIS IS NEEDED:
    gplearn's fit() re-enters _fit() which rebuilds internal state from
    scratch even when warm_start=True. Any attribute injected between fit()
    calls (like _feature_proba) is silently overwritten. This helper stamps
    the normalised probability vector on the live estimator object immediately
    before each fit() so it is present during that call's terminal sampling.

    IMPORTANT: call this immediately before EVERY est_gp.fit(), never before.
    """
    if feature_proba is None:
        return
    if len(feature_proba) != len(feature_names):
        logger.warning("[Fold %d] feature_proba length mismatch — skipping prior.", fold)
        return

    arr = np.array(feature_proba, dtype=float)
    total = arr.sum()
    if total <= 0:
        logger.warning("[Fold %d] feature_proba sums to zero — skipping prior.", fold)
        return
    arr /= total
    est_gp._feature_proba = arr
    logger.debug("[Fold %d] _feature_proba applied — top: %s (p=%.4f)",
                 fold, feature_names[int(np.argmax(arr))], arr.max())


# ---------------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------------------------------

def train_gp_model(
    X_train,
    y_train,
    seed_programs: list = None,
    fold: int = 0,
    feature_proba: np.ndarray = None,
    regime: str = None,                      # FIX-GP-2: new parameter
) -> SymbolicRegressor:
    """
    Train a SymbolicRegressor with optional cross-fold warm-starting.

    Parameters
    ----------
    X_train       : pd.DataFrame — float32 feature matrix (fold-scaled)
    y_train       : pd.Series   — float32 oracle targets
    seed_programs : list[_Program] | None
    fold          : int — current fold index (used as random_state)
    feature_proba : np.ndarray | None — Dirichlet-smoothed feature prior
    regime        : str | None — label from classify_regime(), selects HP profile

    Injection Strategy (unchanged)
    ------------------
    Step A — Phase 1 bootstrap allocates _programs structure.
    Step B — Overwrite weakest n_seeds slots with elite seeds.
    Step C — Phase 2 + 3 resume with warm_start=True.
    FIX-GP-1: _apply_feature_proba() called before EACH fit().
    """
    logger.info("[Fold %d] Initialising GP Engine | regime=%s ...", fold, regime)
    feature_names = list(X_train.columns)
    n_features    = X_train.shape[1]

    # FIX-GP-2: resolve regime hyperparameters
    hp        = _get_regime_hp(regime)
    depth_max = hp["depth_max"] if not seed_programs else min(hp["depth_max"], 6)
    subtree_mut = MUTATION_BOOST if seed_programs else 0.10

    # Probability Normalization: ensure total evolution probability <= 1.0
    # Floating point errors can cause total_p=1.0000000000000002 which fails gplearn's internal check.
    # We scale ONLY the variable components to fit into a conservative 0.999 total target.
    fixed_p   = HOIST_MUTATION + POINT_MUTATION
    total_p   = hp["p_crossover"] + subtree_mut + fixed_p
    if total_p > 0.9999:
        variable_p        = hp["p_crossover"] + subtree_mut
        target_variable_p = 0.998 - fixed_p  # Leave a safe 0.002 margin
        if variable_p > 0:
            scale = target_variable_p / variable_p
            hp = dict(hp)  # shallow copy to avoid mutating the global registry
            hp["p_crossover"] = float(f"{hp['p_crossover'] * scale:.4f}")
            subtree_mut       = float(f"{subtree_mut * scale:.4f}")
            logger.info("[Fold %d] Probabilities normalized (scale: %.4f) | Sum: %.4f", 
                        fold, scale, hp["p_crossover"] + subtree_mut + fixed_p)

    logger.info(
        "[Fold %d] Regime HP | parsimony=(%.4f, %.4f, %.4f) | crossover=%.4f | depth_max=%d",
        fold,
        hp["parsimony_p1"], hp["parsimony_p2"], hp["parsimony_p3"],
        hp["p_crossover"], depth_max,
    )

    est_gp = SymbolicRegressor(
        population_size      = POPULATION_SIZE,
        generations          = GENERATIONS,
        tournament_size      = hp["tournament_size"],
        p_crossover          = hp["p_crossover"],
        p_subtree_mutation   = subtree_mut,
        p_hoist_mutation     = HOIST_MUTATION,
        p_point_mutation     = POINT_MUTATION,
        max_samples          = MAX_SAMPLES,
        parsimony_coefficient= 0.005,         # overridden per-phase below
        function_set         = TRADING_FUNCTIONS,
        init_depth           = (INIT_DEPTH_MIN, depth_max),
        metric               = directional_metric,
        feature_names        = feature_names,
        n_jobs               = 1,
        verbose              = 1,
        warm_start           = False,
        random_state         = fold,
    )

    if seed_programs:
        logger.info("[Fold %d] Starting 3-Phase Seeded Run...", fold)

        # ── PHASE 1: Bootstrap ──────────────────────────────────────────────
        est_gp.parsimony_coefficient = hp["parsimony_p1"]
        est_gp.init_depth            = (INIT_DEPTH_MIN, depth_max)
        est_gp.generations           = PHASE1_GENS
        est_gp.warm_start            = False
        _apply_feature_proba(est_gp, feature_proba, feature_names, fold)  # FIX-GP-1
        est_gp.fit(X_train.values, y_train.values)

        # ── SEED INJECTION ──────────────────────────────────────────────────
        n_seeds  = min(len(seed_programs), int(SEED_FRACTION * POPULATION_SIZE))
        last_gen = est_gp._programs[-1]
        rng      = np.random.RandomState(fold + 1000)
        logger.info("[Fold %d] Injecting %d seeds into Gen %d population.",
                    fold, n_seeds, PHASE1_GENS - 1)

        valid_pop   = [(i, p) for i, p in enumerate(last_gen)
                       if p is not None and hasattr(p, 'fitness_')]
        worst_slots = sorted(valid_pop, key=lambda t: t[1].fitness_)[:n_seeds]

        for slot_rank, (pop_idx, _) in enumerate(worst_slots):
            seed = copy.deepcopy(seed_programs[slot_rank % len(seed_programs)])
            if hasattr(seed, 'program'):
                seed.n_features = n_features
                for i in range(len(seed.program)):
                    if isinstance(seed.program[i], (int, np.integer)):
                        if seed.program[i] >= n_features:
                            seed.program[i] = rng.randint(0, n_features)
                if len(seed.program) > 2:
                    n_mutate = max(1, int(MUTATION_BOOST * len(seed.program)))
                    for _ in range(n_mutate):
                        idx = rng.randint(0, len(seed.program))
                        if isinstance(seed.program[idx], (int, np.integer)):
                            seed.program[idx] = rng.randint(0, n_features)
            last_gen[pop_idx] = seed

        n_corrupted = 0
        for i, p in enumerate(last_gen):
            if p is not None and hasattr(p, 'program'):
                p.n_features = n_features
                for j in range(len(p.program)):
                    if isinstance(p.program[j], (int, np.integer)):
                        if p.program[j] >= n_features:
                            p.program[j] = rng.randint(0, n_features)
                            n_corrupted += 1
                if len(p.program) == 0:
                    last_gen[i] = None

        est_gp._programs[-1]  = last_gen
        est_gp.n_features_in_ = n_features

        stale = [p for p in last_gen
                 if p is not None and getattr(p, 'n_features', n_features) != n_features]
        if stale:
            logger.error("[Fold %d] %d programs still have stale n_features after patch!",
                         fold, len(stale))
        else:
            logger.info("[Fold %d] Population sanitized — n_features=%d. Patched %d terminals.",
                        fold, n_features, n_corrupted)

        # ── PHASE 2 ─────────────────────────────────────────────────────────
        est_gp.parsimony_coefficient = hp["parsimony_p2"]
        est_gp.generations           = PHASE1_GENS + PHASE2_GENS
        est_gp.warm_start            = True
        est_gp.n_jobs                = 1
        _apply_feature_proba(est_gp, feature_proba, feature_names, fold)  # FIX-GP-1
        est_gp.fit(X_train.values, y_train.values)

        # ── PHASE 3 ─────────────────────────────────────────────────────────
        est_gp.parsimony_coefficient = hp["parsimony_p3"]
        est_gp.generations           = PHASE1_GENS + PHASE2_GENS + PHASE3_GENS
        est_gp.warm_start            = True
        est_gp.n_jobs                = 1
        _apply_feature_proba(est_gp, feature_proba, feature_names, fold)  # FIX-GP-1
        est_gp.fit(X_train.values, y_train.values)

    else:
        logger.info("[Fold %d] No seeds — starting phased cold-start evolution.", fold)

        # ── COLD PHASE 1 ─────────────────────────────────────────────────────
        est_gp.parsimony_coefficient = hp["parsimony_p1"]
        est_gp.generations           = PHASE1_GENS
        est_gp.warm_start            = False
        _apply_feature_proba(est_gp, feature_proba, feature_names, fold)  # FIX-GP-1
        est_gp.fit(X_train.values, y_train.values)

        # ── COLD PHASE 2 ─────────────────────────────────────────────────────
        est_gp.parsimony_coefficient = hp["parsimony_p2"]
        est_gp.generations           = PHASE1_GENS + PHASE2_GENS
        est_gp.warm_start            = True
        _apply_feature_proba(est_gp, feature_proba, feature_names, fold)  # FIX-GP-1
        est_gp.fit(X_train.values, y_train.values)

        # ── COLD PHASE 3 ─────────────────────────────────────────────────────
        est_gp.parsimony_coefficient = hp["parsimony_p3"]
        est_gp.generations           = GENERATIONS
        est_gp.warm_start            = True
        _apply_feature_proba(est_gp, feature_proba, feature_names, fold)  # FIX-GP-1
        est_gp.fit(X_train.values, y_train.values)

    # Length Check: Reject programs exceeding bloat limit
    best_len = len(est_gp._program.program)
    if hasattr(config, "MAX_PROGRAM_LENGTH") and best_len > config.MAX_PROGRAM_LENGTH:
        logger.warning("[Fold %d] Formula length %d > MAX=%d — rejecting.",
                       fold, best_len, config.MAX_PROGRAM_LENGTH)
        return None

    best = str(est_gp._program)
    logger.info("[Fold %d] Best formula: %s", fold, best)
    print(f"\n[Fold {fold}] Best Formula: {best}")
    return est_gp
