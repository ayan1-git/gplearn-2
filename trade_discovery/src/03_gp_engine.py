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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CUSTOM LOGICAL & COMPARISON OPERATORS
# ---------------------------------------------------------------------------

def _gt_soft(x1, x2):  return np.tanh((x1 - x2) * 3.0)
def _lt_soft(x1, x2):  return np.tanh((x2 - x1) * 3.0)
def _and(x1, x2):      return np.minimum(x1, x2)
def _or(x1, x2):       return np.maximum(x1, x2)
def _if_then(c, t, f): return np.where(c > 0.0, t, f)

greater_than = make_function(function=_gt_soft, name='gt',      arity=2)
less_than    = make_function(function=_lt_soft, name='lt',      arity=2)
logical_and  = make_function(function=_and,     name='and',     arity=2)
logical_or   = make_function(function=_or,      name='or',      arity=2)
if_then      = make_function(function=_if_then, name='if_then', arity=3)

TRADING_FUNCTIONS = [
    'add', 'sub', 'mul', 'div', 'max', 'min', 'abs', 'neg',
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
    
    # Base Pearson correlation
    y_mean    = np.mean(y)
    yp_mean   = np.mean(y_pred)
    num       = np.sum((y - y_mean) * (y_pred - yp_mean))
    denom     = (np.std(y) * np.std(y_pred) * len(y)) + EPS
    pearson_r = num / denom

    # Directional coverage penalty
    long_mask  = (y == 1.0)
    short_mask = (y == -1.0)
    
    n_long_pred  = np.sum(y_pred[long_mask]  > 0) if np.any(long_mask)  else 0
    n_short_pred = np.sum(y_pred[short_mask] < 0) if np.any(short_mask) else 0
    n_long_true  = np.sum(long_mask)
    n_short_true = np.sum(short_mask)
    
    long_coverage  = n_long_pred  / (n_long_true  + EPS)
    short_coverage = n_short_pred / (n_short_true + EPS)
    
    # Harmonic mean of directional coverages — zero if either side is dead
    dir_score = (2 * long_coverage * short_coverage) / (long_coverage + short_coverage + EPS)
    
    # Blended fitness
    return float(PEARSON_WEIGHT * pearson_r + DIRECTION_WEIGHT * dir_score)


directional_metric = make_fitness(function=_directional_fitness, greater_is_better=True)

# ---------------------------------------------------------------------------
# CONSTANTS (Imported from config or defaulted)
# ---------------------------------------------------------------------------


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

    Deep-copied to prevent shared-state mutation bugs between fold iterations.
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
# MAIN TRAINING FUNCTION
# ---------------------------------------------------------------------------

def train_gp_model(
    X_train,
    y_train,
    seed_programs: list = None,
    fold: int = 0
) -> SymbolicRegressor:
    """
    Train a SymbolicRegressor with optional cross-fold warm-starting.

    Parameters
    ----------
    X_train       : pd.DataFrame — float32 feature matrix (fold-scaled)
    y_train       : pd.Series   — float32 oracle targets
    seed_programs : list[_Program] | None
                    Elite _Program objects from extract_elite_programs() on
                    the previous fold's fitted model. None = cold start.
    fold          : int — current fold index.
                    Used as random_state for per-fold population diversity.

    Injection Strategy
    ------------------
    gplearn does not expose a public warm-start population API.
    The production-safe workaround:
      Step A — run 1 generation to allocate _programs[0] structure.
      Step B — overwrite the weakest n_seeds slots with elite seeds.
      Step C — resume with warm_start=True for remaining 59 generations.

    This avoids forking gplearn while achieving true cross-fold gene carryover.
    """
    logger.info("[Fold %d] Initialising GP Engine...", fold)
    feature_names = list(X_train.columns)

    # FIX 1 & novelty pressure:
    # - random_state=fold   → different initial population per fold
    # - subtree_mut elevated → seeded individuals are mutated, not cloned
    subtree_mut = MUTATION_BOOST if seed_programs else 0.10

    est_gp = SymbolicRegressor(
        population_size      = POPULATION_SIZE,
        generations          = GENERATIONS,
        tournament_size      = TOURNAMENT_SIZE,
        p_crossover          = 0.6,
        p_subtree_mutation   = subtree_mut,   # elevated when seeding
        p_hoist_mutation     = 0.1,
        p_point_mutation     = 0.1,
        max_samples          = 0.7,
        parsimony_coefficient= 0.005,
        function_set         = TRADING_FUNCTIONS,
        init_depth           = (3, 6),
        metric               = directional_metric,
        feature_names        = feature_names,
        n_jobs               = 2,
        verbose              = 1,
        warm_start           = False,         # managed manually below
        random_state         = fold,          # FIX 1: per-fold diversity
    )

    if seed_programs:
        logger.info("[Fold %d] Starting 3-Phase Seeded Run...", fold)

        # PHASE 1: Bootstrap — 10 gens, zero parsimony, high depth pressure
        # Purpose: let seeded complex programs compete fairly before simplification
        est_gp.parsimony_coefficient = 0.0      # no length penalty
        est_gp.init_depth            = (INIT_DEPTH_MIN, INIT_DEPTH_MAX)
        est_gp.generations           = PHASE1_GENS
        est_gp.warm_start            = False
        est_gp.fit(X_train.values, y_train.values)

        # INJECT SEEDS — same as current logic
        n_seeds = min(len(seed_programs), int(SEED_FRACTION * POPULATION_SIZE))
        logger.info("[Fold %d] Injecting %d seeds into Gen %d population.", fold, n_seeds, PHASE1_GENS-1)

        last_gen    = est_gp._programs[-1]
        rng         = np.random.RandomState(fold + 1000)
        n_features  = X_train.shape[1]

        valid_pop   = [(i, p) for i, p in enumerate(last_gen)
                       if p is not None and hasattr(p, 'fitness_')]
        worst_slots = sorted(valid_pop, key=lambda t: t[1].fitness_)[:n_seeds]

        for slot_rank, (pop_idx, _) in enumerate(worst_slots):
            seed = copy.deepcopy(seed_programs[slot_rank % len(seed_programs)])

            if hasattr(seed, 'program'):
                # ── Terminal Sanitization ──
                # Ensure all terminal indices are within current feature bounds.
                # Prevents IndexError if seeds came from a fold with more features (e.g. Rotation).
                for i in range(len(seed.program)):
                    if isinstance(seed.program[i], (int, np.integer)):
                        if seed.program[i] >= n_features:
                            seed.program[i] = rng.randint(0, n_features)

                # ── Variance Restoration ──
                # Mutate terminals to restore variance and maintain diversity
                if len(seed.program) > 2:
                    n_mutate = max(1, int(MUTATION_BOOST * len(seed.program)))
                    for _ in range(n_mutate):
                        idx  = rng.randint(0, len(seed.program))
                        if isinstance(seed.program[idx], (int, np.integer)):
                            seed.program[idx] = rng.randint(0, n_features)
            
            last_gen[pop_idx] = seed

        # PHASE 2: Exploit — 30 gens, mild parsimony, warm start
        est_gp.parsimony_coefficient = 0.0005
        est_gp.generations           = PHASE1_GENS + PHASE2_GENS
        est_gp.warm_start            = True
        est_gp.fit(X_train.values, y_train.values)

        # PHASE 3: Regularize — 20 gens, normal parsimony
        est_gp.parsimony_coefficient = 0.003
        est_gp.generations           = PHASE1_GENS + PHASE2_GENS + PHASE3_GENS
        est_gp.warm_start            = True
        est_gp.fit(X_train.values, y_train.values)

    else:
        logger.info("[Fold %d] No seeds — cold-start evolution.", fold)
        est_gp.fit(X_train.values, y_train.values)

    best = str(est_gp._program)
    logger.info("[Fold %d] Best formula: %s", fold, best)
    print(f"\n[Fold {fold}] Best Formula: {best}")
    return est_gp
