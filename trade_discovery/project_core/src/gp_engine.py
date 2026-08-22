"""
gp_engine.py — DEAP-based genetic-programming engine for trade_discovery.

This module replaces the previous gplearn SymbolicRegressor wrapper. DEAP
exposes the full evolution loop, so everything that previously required
hacking gplearn internals is now native:

  - Feature prior: the Dirichlet-smoothed feature probability vector from the
    GEL feedback loop drives EVERY variable-terminal draw (initial trees,
    subtree mutation, point mutation) via an inverse-CDF sample. The old
    src/gplearn_prior_patch.py on-disk source patch is gone.
  - 3-phase parsimony schedule: phases are just loop parameters — no
    warm_start re-fit gymnastics.
  - Seeding: elite individuals are injected directly into the initial
    population (each seed EXACTLY ONCE — preserves the FIX #6 contract),
    padded with anti-convergence randoms and fresh ramped half-and-half
    trees.

External surface used by scripts/main_pipeline.py:
  train_gp_model(...)        -> GPResult (or None on bloat-guard rejection)
  GPResult.predict(X)        -> ndarray signal vector (vectorbt-compatible)
  GPResult.best_formula      -> str, Lisp-style `fn(arg, ...)` formula string.
       Primitive names match the legacy gplearn set exactly
       (add/sub/mul/div/max/min/abs/neg/gt/lt/and/or/if_then) and argument
       terminals print real feature names, so all downstream string guards
       (_features_used, trivial-cancellation detector, bool-ratio check,
       leaderboard regexes) keep working unchanged.
  GPResult.best_program      -> DEAP PrimitiveTree individual (deep-copyable)
  GPResult.best_length       -> node count (bloat metric)
  extract_elite_programs(...) -> List[Individual] seeds for the next run
  hash_formula(str)
"""
import copy
import hashlib
import logging
import math
import operator as _op
import random
from functools import partial

import numpy as np

from deap import base, creator, gp, tools

try:
    import src.config as config
except ImportError:                                   # pragma: no cover
    config = None

logger = logging.getLogger(__name__)

# ── CONFIG RESOLUTION (identical keys to the gplearn era) ────────────────────

def _cfg(key, default):
    return getattr(config, key, default) if config is not None else default


POPULATION_SIZE = int(_cfg("GP_POPULATION_SIZE", 3000))
SEED_FRACTION   = float(_cfg("GP_SEED_FRACTION", 0.20))
MUTATION_BOOST  = float(_cfg("GP_MUTATION_BOOST", 0.15))
GENERATIONS     = int(_cfg("GP_GENERATIONS", 60))
PHASE1_GENS     = int(_cfg("GP_PHASE1_GENS", 15))
PHASE2_GENS     = int(_cfg("GP_PHASE2_GENS", 25))
PHASE3_GENS     = int(_cfg("GP_PHASE3_GENS", 20))
INIT_DEPTH_MIN  = int(_cfg("GP_INIT_DEPTH_MIN", 4))
INIT_DEPTH_MAX  = int(_cfg("GP_INIT_DEPTH_MAX", 8))
PEARSON_WEIGHT   = float(_cfg("FITNESS_PEARSON_WEIGHT", 0.70))
DIRECTION_WEIGHT = float(_cfg("FITNESS_DIRECTION_WEIGHT", 0.30))
HOIST_MUTATION   = float(_cfg("GP_HOIST_MUTATION", 0.1))
POINT_MUTATION   = float(_cfg("GP_POINT_MUTATION", 0.1))
MAX_SAMPLES      = float(_cfg("GP_MAX_SAMPLES", 0.9))
GT_SOFT_SCALE    = float(_cfg("GT_SOFT_SCALE", 3.0))
ATTR_PENALTY_WEIGHT = float(_cfg("FITNESS_ATTR_PENALTY_WEIGHT", 0.25))
POSTHOC_TRADE_EVAL  = bool(_cfg("FITNESS_POSTHOC_TRADE_EVAL", False))
SPREAD_FLOOR  = float(_cfg("FITNESS_SPREAD_FLOOR", 0.08))
SPREAD_WEIGHT = float(_cfg("FITNESS_SPREAD_WEIGHT", 5.0))
ANTI_CONVERGENCE_FRACTION = float(_cfg("GEL_ANTI_CONVERGENCE_FRACTION", 0.10))
MAX_PROGRAM_LENGTH = int(_cfg("MAX_PROGRAM_LENGTH", 40))
# Anti-collapse floors: the GEL structural guard rejects formulas with <3
# features ("too shallow"), so the FITNESS must enforce the same floor —
# otherwise trivial 1–2-feature formulas dominate train fitness + parsimony,
# tournament selection homogenises the population onto them within ~5 gens,
# and whole generations get wasted on guard rejections (observed empirically).
GP_MIN_NODES   = int(_cfg("GP_MIN_NODES", 8))
GP_MIN_FEATURES = int(_cfg("GP_MIN_FEATURES", 3))
GP_SHALLOW_PENALTY_PER_FEATURE = float(_cfg("GP_SHALLOW_PENALTY_PER_FEATURE", 0.15))
GP_SHALLOW_PENALTY_PER_NODE    = float(_cfg("GP_SHALLOW_PENALTY_PER_NODE", 0.02))
N_JOBS_DEFAULT     = int(_cfg("GP_N_JOBS", 1))
ELITE_FRACTION     = float(_cfg("GP_ELITE_FRACTION", 0.05))   # μ+λ elitism share
POINT_NODE_PROB    = float(_cfg("GP_POINT_NODE_PROB", 0.15))  # per-node point-mut prob

assert PHASE1_GENS + PHASE2_GENS + PHASE3_GENS == GENERATIONS, (
    f"Phase gens sum ({PHASE1_GENS}+{PHASE2_GENS}+{PHASE3_GENS}) "
    f"!= GP_GENERATIONS ({GENERATIONS}). Fix config.py."
)

# ─────────────────────────────────────────────────────────────────────────────
# FIX-GP-2: REGIME → HYPERPARAMETER MAP (unchanged semantics)
# ─────────────────────────────────────────────────────────────────────────────

_REGIME_HP = _cfg("REGIME_HP", {
    "trending": {
        "parsimony_p1":   0.0,
        "parsimony_p2":   0.0002,
        "parsimony_p3":   0.002,
        "p_crossover":    0.70,
        "depth_max":      8,
        "tournament_size": 12,
    },
    "mean_reverting": {
        "parsimony_p1":   0.001,
        "parsimony_p2":   0.003,
        "parsimony_p3":   0.008,
        "p_crossover":    0.55,
        "depth_max":      6,
        "tournament_size": 7,
    },
    "choppy_random_walk": {
        "parsimony_p1":   0.002,
        "parsimony_p2":   0.005,
        "parsimony_p3":   0.010,
        "p_crossover":    0.50,
        "depth_max":      5,
        "tournament_size": 5,
    },
    "trending_random_walk": {
        "parsimony_p1":   0.0,
        "parsimony_p2":   0.001,
        "parsimony_p3":   0.004,
        "p_crossover":    0.65,
        "depth_max":      7,
        "tournament_size": 10,
    },
    "random_walk": {
        "parsimony_p1":   0.0,
        "parsimony_p2":   0.0005,
        "parsimony_p3":   0.003,
        "p_crossover":    0.60,
        "depth_max":      8,
        "tournament_size": 7,
    },
})

_DEFAULT_REGIME_HP = _REGIME_HP.get("random_walk", {})


def _get_regime_hp(regime):
    if regime is None:
        return dict(_DEFAULT_REGIME_HP)
    hp = _REGIME_HP.get(regime)
    if hp is None:
        logger.warning("Unknown regime '%s' — falling back to random_walk HP.", regime)
        return dict(_DEFAULT_REGIME_HP)
    return dict(hp)


# ─────────────────────────────────────────────────────────────────────────────
# PRIMITIVE SET — names must match the legacy gplearn function set exactly;
# downstream string guards parse these tokens out of str(individual).
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


PRIMITIVE_NAMES = (
    'add', 'sub', 'mul', 'div', 'max', 'min', 'abs', 'neg',
    'gt', 'lt', 'and', 'or', 'if_then',
)

BOOL_OPS = ('lt', 'gt', 'and', 'or')          # used by main_pipeline guard
ALL_OP_NAMES = PRIMITIVE_NAMES                # superset for total-op counting


def _rand_const():
    return round(random.uniform(-1.0, 1.0), 4)


def build_pset(n_features: int, feature_names) -> gp.PrimitiveSet:
    """Build an untyped DEAP primitive set with named arguments.

    Argument terminals print as the real feature names in str(individual),
    matching what the pipeline guards and leaderboards expect. An ephemeral
    constant in U(-1, 1) mirrors gplearn's const_range=(-1, 1) behaviour with
    the same 1/(n+1) terminal-vs-constant draw split (see make_terminal_factory).
    """
    pset = gp.PrimitiveSet("MAIN", n_features)
    rename = {f"ARG{i}": str(name) for i, name in enumerate(feature_names)}
    pset.renameArguments(**rename)

    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    pset.addPrimitive(_protected_div, 2, name="div")
    pset.addPrimitive(np.maximum, 2, name="max")
    pset.addPrimitive(np.minimum, 2, name="min")
    pset.addPrimitive(np.abs, 1, name="abs")
    pset.addPrimitive(np.negative, 1, name="neg")
    pset.addPrimitive(_gt_soft, 2, name="gt")
    pset.addPrimitive(_lt_soft, 2, name="lt")
    pset.addPrimitive(_and, 2, name="and")
    pset.addPrimitive(_or, 2, name="or")
    pset.addPrimitive(_if_then, 3, name="if_then")

    pset.addEphemeralConstant("const", partial(_rand_const))
    # DEAP 1.4 does not expose a per-pset ephemeral registry and
    # addEphemeralConstant returns None — the ephemeral CLASS is registered in
    # pset.mapping under its name (see deap.gp._add). Keep our own handle.
    pset._trading_ephemerals = [pset.mapping["const"]]
    return pset


# ─────────────────────────────────────────────────────────────────────────────
# TREE EXECUTION — postfix walk over the prefix-order node list. Avoids
# gp.compile's lambdify entirely; numpy primitives broadcast over columns.
# ─────────────────────────────────────────────────────────────────────────────

def execute_tree(individual, X: np.ndarray, feat_index: dict, context: dict) -> np.ndarray:
    """Evaluate a PrimitiveTree column-wise over X (rows × features).

    `context` is the pset's name→callable registry (deap Primitive objects do
    not carry the python function; pset.context does).
    """
    n_rows = X.shape[0]
    stack = []
    for node in reversed(individual):
        if isinstance(node, gp.Primitive):
            if len(stack) < node.arity:
                raise RuntimeError("Malformed tree: stack underflow")
            args = [stack.pop() for _ in range(node.arity)]
            func = context[node.name]
            stack.append(func(*args))
        else:
            value = getattr(node, "value", None)
            if isinstance(value, str):                 # argument terminal
                col = feat_index.get(value)
                if col is None:
                    raise KeyError(f"Unknown feature terminal '{value}'")
                stack.append(X[:, col])
            else:                                      # ephemeral constant
                stack.append(value)
    out = stack.pop()
    if len(stack):
        raise RuntimeError("Malformed tree: leftover stack items")
    if np.isscalar(out):
        out = np.full(n_rows, float(out), dtype=np.float64)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE-PRIOR-AWARE TERMINALS & GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def make_terminal_factory(pset, feat_terms, feature_proba=None):
    """
    Draw one terminal. Mirrors gplearn's const_range split: a variable
    terminal is chosen with probability n/(n+1), a fresh U(-1,1) ephemeral
    with probability 1/(n+1). Variable index sampling honours the prior via
    inverse-CDF when provided, uniform otherwise.
    """
    n_feat = len(feat_terms)
    cum_proba = None
    if feature_proba is not None:
        arr = np.asarray(feature_proba, dtype=float)
        cum_proba = np.cumsum(arr / arr.sum())

    def draw_terminal():
        if random.random() * (n_feat + 1) < 1.0:
            return random.choice(pset._trading_ephemerals)()
        if cum_proba is None:
            idx = random.randrange(n_feat)
        else:
            idx = int(np.searchsorted(cum_proba, random.random(), side="right"))
            idx = min(idx, n_feat - 1)
        return feat_terms[idx]

    return draw_terminal


def generate_tree(pset, terminal_factory, min_, max_, type_=None):
    """Ramped half-and-half tree generator (grow/full mix per depth).

    Drop-in replacement for deap.gp.genGrow/genFull whose terminal draws route
    through `terminal_factory` so the feature prior is honoured everywhere a
    tree is built (initial population AND mutation subtrees).
    """
    def condition(height, depth):
        # full-style until `min_`, then grow-style coin flip (deap convention)
        return depth == height or (depth >= min_ and random.random() < 0.1)

    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_ or pset.ret)]
    while stack:
        depth, ttype = stack.pop()
        if condition(height, depth):
            expr.append(terminal_factory())
        else:
            prim = random.choice(pset.primitives[ttype])
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM MUTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def mut_hoist(individual):
    """gplearn-style hoist mutation: pick a random inner subtree, lift it to
    the root (strictly shrinks the tree — cannot violate depth limits)."""
    if len(individual) < 2:
        return individual,
    index = random.randrange(1, len(individual))   # never the root itself
    slice_ = individual.searchSubtree(index)
    nodes = list(individual[slice_])
    # Bypass PrimitiveTree.__setitem__: it rejects open-ended slices and
    # would also flag the intentional whole-tree replacement as invalid.
    list.__setitem__(individual, slice(None), nodes)
    return individual,


def make_mut_point(pset, terminal_factory, node_prob=POINT_NODE_PROB):
    """Point mutation: each node flips with probability `node_prob`.
    Primitives swap to same-arity primitives; terminals redraw through the
    prior-aware factory (this is where point mutations honour the prior)."""

    def _mutate(individual):
        for i, node in enumerate(individual):
            if random.random() >= node_prob:
                continue
            if isinstance(node, gp.Primitive):
                candidates = [p for p in pset.primitives[node.ret]
                              if p.arity == node.arity]
                if candidates:
                    individual[i] = random.choice(candidates)
            else:
                individual[i] = terminal_factory()
        return individual,

    return _mutate


# ─────────────────────────────────────────────────────────────────────────────
# FITNESS — exact port of the legacy _directional_fitness
# ─────────────────────────────────────────────────────────────────────────────

def directional_score(y, y_pred, pearson_w, direction_w):
    EPS = 1e-8

    y_mean  = np.mean(y)
    yp_mean = np.mean(y_pred)
    yp_std  = np.std(y_pred)

    if yp_std < 1e-9:
        pearson_r = 0.0
    else:
        num       = np.sum((y - y_mean) * (y_pred - yp_mean))
        denom     = (np.std(y) * yp_std * len(y)) + EPS
        pearson_r = num / denom

    long_mask  = (y == 1.0)
    short_mask = (y == -1.0)

    n_long_pred  = np.sum(y_pred[long_mask]  > 0) if np.any(long_mask)  else 0
    n_short_pred = np.sum(y_pred[short_mask] < 0) if np.any(short_mask) else 0
    n_long_true  = np.sum(long_mask)
    n_short_true = np.sum(short_mask)

    long_cov  = n_long_pred  / (n_long_true  + EPS)
    short_cov = n_short_pred / (n_short_true + EPS)
    dir_score = (2 * long_cov * short_cov) / (long_cov + short_cov + EPS)

    wrong_long  = np.mean(np.maximum(0.0, -y_pred[long_mask]))  if np.any(long_mask)  else 0.0
    wrong_short = np.mean(np.maximum(0.0,  y_pred[short_mask])) if np.any(short_mask) else 0.0
    barrier_penalty = (wrong_long + wrong_short) / 2.0

    spread_penalty = max(0.0, SPREAD_FLOOR - yp_std)

    base_score = pearson_w * pearson_r + direction_w * dir_score
    return float(base_score
                 - ATTR_PENALTY_WEIGHT * barrier_penalty
                 - SPREAD_WEIGHT * spread_penalty)


_REJECT_SCORE = -9999.0


def _features_used(individual) -> set:
    """Distinct named feature terminals in the tree (constants excluded)."""
    feats = set()
    for node in individual:
        if isinstance(node, gp.Primitive):
            continue
        v = getattr(node, "value", None)
        if isinstance(v, str):
            feats.add(v)
    return feats


def _shallow_penalty(individual, n_features_used: int) -> float:
    """Penalty pushing programs above the minimum complexity floors so that
    selection pressure aligns with main_pipeline's structural guard."""
    node_deficit = max(0, GP_MIN_NODES - len(individual))
    feat_deficit = max(0, GP_MIN_FEATURES - n_features_used)
    return (GP_SHALLOW_PENALTY_PER_NODE * node_deficit +
            GP_SHALLOW_PENALTY_PER_FEATURE * feat_deficit)


def evaluate_individual(individual, X, y, parsimony, feat_index, context,
                        pearson_w, direction_w):
    """Raw directional fitness minus parsimony×length and the shallow-formula
    penalty (mirrors how gplearn folded parsimony into fitness_)."""
    try:
        y_pred = execute_tree(individual, X, feat_index, context)
        if not np.all(np.isfinite(y_pred)):
            return (_REJECT_SCORE,)
        raw = directional_score(y, y_pred, pearson_w, direction_w)
    except Exception:
        return (_REJECT_SCORE,)
    raw -= _shallow_penalty(individual, len(_features_used(individual)))
    individual.raw_score = raw
    return (raw - parsimony * len(individual),)


def _evaluate_population(population, X_eval, y_eval, parsimony, feat_index,
                         context, pearson_w, direction_w, n_jobs):
    if n_jobs and n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            scores = Parallel(n_jobs=n_jobs, batch_size="auto")(
                delayed(evaluate_individual)(
                    ind, X_eval, y_eval, parsimony, feat_index, context,
                    pearson_w, direction_w)
                for ind in population
            )
        except Exception as exc:
            logger.warning("Parallel evaluation failed (%s) — falling back "
                           "to serial for this generation.", exc)
            scores = None
        if scores is not None:
            for ind, score in zip(population, scores):
                ind.fitness.values = score
            return
    for ind in population:
        ind.fitness.values = evaluate_individual(
            ind, X_eval, y_eval, parsimony, feat_index, context,
            pearson_w, direction_w)


# ─────────────────────────────────────────────────────────────────────────────
# CREATOR CLASSES (module-level, created once per process)
# ─────────────────────────────────────────────────────────────────────────────

if not hasattr(creator, "FitnessGPDir"):
    creator.create("FitnessGPDir", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessGPDir)


def _as_individual(nodes):
    ind = creator.Individual(nodes)
    return ind


def hash_formula(program_str: str) -> str:
    """SHA-256 fingerprint of a formula string for deduplication."""
    return hashlib.sha256(program_str.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# ELITE EXTRACTION — verbatim port of the diversity-enforced selection
# ─────────────────────────────────────────────────────────────────────────────

def extract_elite_programs(source, top_n: int = None, max_duplicates: int = 2,
                           diversity_fraction: float = 0.30) -> list:
    """
    Extract elite programs with diversity enforcement to prevent gene pool
    collapse.

    Strategy:
      1. Hard dedup — at most `max_duplicates` copies of any formula string.
      2. Top (1 − diversity_fraction) slots filled by raw fitness.
      3. Remaining slots filled via greedy farthest-first (trigram Jaccard
         distance) to maximise structural diversity in the seed pool.

    Accepts a GPResult or any iterable of evaluated DEAP individuals.
    Returns deep copies safe to hand to train_gp_model(seed_programs=...).
    """
    if isinstance(source, GPResult):
        last_gen = source.population
    else:
        last_gen = list(source) if source is not None else []

    if not last_gen:
        logger.warning("No programs found — returning empty seed list.")
        return []

    top_n = top_n or int(SEED_FRACTION * POPULATION_SIZE)

    candidates = []
    for p in last_gen:
        try:
            fit = p.fitness.values[0]
            s = str(p)
        except Exception:
            continue
        candidates.append((p, fit, s))

    if not candidates:
        return []

    candidates.sort(key=lambda t: t[1], reverse=True)

    # ── Hard dedup ────────────────────────────────────────────────────────
    formula_counts = {}
    deduped = []
    for p, fit, s in candidates:
        count = formula_counts.get(s, 0)
        if count < max_duplicates:
            deduped.append((p, fit, s))
            formula_counts[s] = count + 1

    n_unique = len(formula_counts)
    logger.info("Elite dedup: %d → %d (max_%d_copies | %d unique formulas).",
                len(candidates), len(deduped), max_duplicates, n_unique)

    if len(deduped) <= top_n:
        result = [copy.deepcopy(t[0]) for t in deduped]
        logger.info("Extracted %d elite programs (all deduped fit in pool).",
                    len(result))
        return result

    # ── Diversity-aware selection ─────────────────────────────────────────
    n_fitness = max(1, int(top_n * (1 - diversity_fraction)))
    selected_indices = list(range(n_fitness))

    def _trigrams(s):
        if len(s) < 3:
            return frozenset([s])
        return frozenset(s[i:i + 3] for i in range(len(s) - 2))

    all_tg = [_trigrams(s) for _, _, s in deduped]

    # Greedy farthest-first fill with incremental min-distance maintenance
    # (identical selection semantics to the gplearn-era implementation).
    n_diverse = top_n - n_fitness
    remaining = set(range(n_fitness, len(deduped)))

    def _jaccard_dist(i, j):
        inter = len(all_tg[i] & all_tg[j])
        union = len(all_tg[i] | all_tg[j])
        return 1.0 - (inter / max(union, 1))

    min_dist = {}
    for idx in remaining:
        md = 1.0
        for sel_idx in selected_indices:
            d = _jaccard_dist(idx, sel_idx)
            if d < md:
                md = d
                if md <= 0.0:
                    break
        min_dist[idx] = md

    for _ in range(min(n_diverse, len(remaining))):
        best_idx = -1
        best_score = -1.0
        for idx in remaining:
            rank_frac = 1.0 - idx / len(deduped)
            score = min_dist[idx] * (0.3 + 0.7 * rank_frac)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0:
            remaining.discard(best_idx)
            selected_indices.append(best_idx)
            for idx in remaining:
                if min_dist[idx] > 0.0:
                    d = _jaccard_dist(idx, best_idx)
                    if d < min_dist[idx]:
                        min_dist[idx] = d

    result = [copy.deepcopy(deduped[i][0]) for i in selected_indices]
    logger.info("Extracted %d elite programs (%d fitness + %d diversity | "
                "pool=%d, unique=%d).",
                len(result), n_fitness, len(result) - n_fitness,
                len(candidates), n_unique)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# RESULT WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class GPResult:
    """Stand-in for the fitted SymbolicRegressor consumed by main_pipeline."""

    def __init__(self, best_program, population, feat_index, feature_names,
                 context):
        self.best_program = best_program
        self.population = population
        self.feat_index = feat_index
        self.feature_names = list(feature_names)
        self.context = context

    def predict(self, X) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return execute_tree(self.best_program, X_arr, self.feat_index,
                            self.context)

    @property
    def best_formula(self) -> str:
        return str(self.best_program)

    @property
    def best_length(self) -> int:
        return len(self.best_program)

    def __len__(self):  # convenience parity with estimator API
        return self.best_length


# ─────────────────────────────────────────────────────────────────────────────
# POST-HOC TRADE-SIMULATION SELECTION (kept behind FITNESS_POSTHOC_TRADE_EVAL,
# disabled by default exactly as in the P2 config)
# ─────────────────────────────────────────────────────────────────────────────

def _score_program_trade_fitness(individual, feat_index, context, X_train_s,
                                 df_raw_train, entry_pct, exit_pct,
                                 tp_mult, sl_mult, atr_period):
    try:
        from trade_discovery.Audit_scripts.vectorbt_evaluator import \
            evaluate_formula_with_vectorbt

        class _Wrapper:
            def predict(self_inner, X):
                return execute_tree(individual, np.asarray(X, dtype=np.float64),
                                    feat_index, context)

        _, stats, meta = evaluate_formula_with_vectorbt(
            _Wrapper(), X_train_s, df_raw_train,
            long_pct_level=entry_pct, short_pct_level=exit_pct,
            tp_mult=tp_mult, sl_mult=sl_mult,
        )
        sharpe = float(stats.get("Sharpe Ratio", 0.0) if isinstance(stats, dict)
                       else stats.loc["Sharpe Ratio"] if "Sharpe Ratio" in stats.index else 0.0)
        n_trades = meta['n_long'] + meta['n_short']
        return sharpe, n_trades
    except Exception:
        return -999.0, 0


def _select_best_by_trade_fitness(gp_result: GPResult, X_train_s, df_raw_train,
                                  fold, entry_pct, exit_pct, tp_mult,
                                  sl_mult, atr_period) -> bool:
    """From the top-20 final-generation programs, pick the best train-set
    VectorBT Sharpe and promote it to result.best_program."""
    candidates = sorted(
        [ind for ind in gp_result.population],
        key=lambda p: getattr(p, "raw_score", _REJECT_SCORE),
        reverse=True,
    )[:20]
    best_prog, best_sharpe = None, -999.0
    for prog in candidates:
        sharpe, n_trades = _score_program_trade_fitness(
            prog, gp_result.feat_index, gp_result.context, X_train_s,
            df_raw_train, entry_pct, exit_pct, tp_mult, sl_mult, atr_period)
        if n_trades >= 30 and sharpe > best_sharpe:
            best_sharpe, best_prog = sharpe, prog

    if best_prog is not None:
        logger.info("[Fold %d] Post-hoc trade-Sharpe selection: promoted "
                    "program with train Sharpe=%.4f", fold, best_sharpe)
        gp_result.best_program = copy.deepcopy(best_prog)
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def _phase_parsimony(gen_idx: int, hp: dict) -> float:
    if gen_idx < PHASE1_GENS:
        return hp["parsimony_p1"]
    if gen_idx < PHASE1_GENS + PHASE2_GENS:
        return hp["parsimony_p2"]
    return hp["parsimony_p3"]


def _sanitize_seed(seed_nodes, pset, feat_terms, feat_index, n_features):
    """Convert stored seed nodes into a valid Individual for this fold's
    primitive set. Terminals referencing unknown features (stale fold
    layout) are replaced with random prior-drawn terminals."""
    known_names = {t.name for t in feat_terms}
    fixed = []
    for node in seed_nodes:
        if isinstance(node, gp.Primitive):
            fixed.append(node)
        else:
            value = getattr(node, "value", None)
            if isinstance(value, str) and value in known_names:
                fixed.append(node)
            else:
                fixed.append(random.choice(feat_terms))
    return creator.Individual(fixed)


def train_gp_model(X_train, y_train, seed_programs=None, fold: int = 0,
                   feature_proba=None, regime: str = None,
                   df_raw_train=None, n_jobs: int = None) -> GPResult | None:
    """
    Evolve trading formulas on (X_train, y_train) with optional warm-starting
    from the previous GEL generation's elite pool.

    Parameters
    ----------
    X_train       : pd.DataFrame — float32 feature matrix (fold-scaled)
    y_train       : pd.Series   — float32 oracle targets
    seed_programs : list[Individual] | None — DEAP individuals from
                    extract_elite_programs() of the previous generation
    fold          : int — current generation index (drives RNG seeding)
    feature_proba : np.ndarray | None — Dirichlet-smoothed feature prior;
                    NATIVELY honoured by every terminal draw (no patches)
    regime        : str | None — label from classify_regime(), selects HP profile
    df_raw_train  : pd.DataFrame | None — raw OHLC data for post-hoc
                    trade-simulation selection
    n_jobs        : int | None — >1 evaluates fitness across the population
                    with joblib (workers receive the fold matrix); defaults
                    to config.GP_N_JOBS (serial).

    Returns GPResult, or None when the bloat guard rejects the winner
    (contract preserved from the gplearn engine).
    """
    logger.info("[Fold %d] Initialising DEAP GP Engine | regime=%s ...",
                fold, regime)

    # Reproducibility: DEAP generators/operators use the stdlib `random`
    # module; the prior sampler uses numpy.
    random.seed(fold)
    np.random.seed(fold % (2 ** 32))

    if n_jobs is None:
        n_jobs = N_JOBS_DEFAULT

    feature_names = [str(c) for c in X_train.columns]
    n_features = X_train.shape[1]
    feat_index = {name: i for i, name in enumerate(feature_names)}

    hp = _get_regime_hp(regime)
    pearson_w   = hp.get("fitness_pearson_w", PEARSON_WEIGHT)
    direction_w = hp.get("fitness_direction_w", DIRECTION_WEIGHT)
    depth_max   = hp["depth_max"] if not seed_programs else min(hp["depth_max"], 6)
    tournament_size = hp["tournament_size"]
    p_crossover = hp["p_crossover"]

    logger.info("[Fold %d] Fitness weights | pearson=%.2f | direction=%.2f",
                fold, pearson_w, direction_w)
    logger.info("[Fold %d] Regime HP | parsimony=(%.4f, %.4f, %.4f) | "
                "crossover=%.4f | depth_max=%d | tournament=%d",
                fold, hp["parsimony_p1"], hp["parsimony_p2"], hp["parsimony_p3"],
                p_crossover, depth_max, tournament_size)

    # ── Feature prior normalisation (defensive; GEL already normalises) ───
    proba = None
    if feature_proba is not None:
        arr = np.array(feature_proba, dtype=float)
        if len(arr) != n_features:
            logger.warning("[Fold %d] feature_proba length mismatch — "
                           "skipping prior.", fold)
        elif arr.sum() <= 0:
            logger.warning("[Fold %d] feature_proba sums to zero — skipping "
                           "prior.", fold)
        else:
            proba = arr / arr.sum()

    # ── Primitive set / toolbox (fresh per fold: n_features varies) ───────
    pset = build_pset(n_features, feature_names)
    feat_terms = [pset.mapping[name] for name in feature_names]
    context = {k: v for k, v in pset.context.items() if k != "__builtins__"}
    terminal_factory = make_terminal_factory(pset, feat_terms, proba)

    toolbox = base.Toolbox()
    toolbox.register("clone", copy.deepcopy)

    subtree_gen_depth = max(1, min(4, depth_max))

    def _gen_subtree(pset=None, type_=None):
        # gp.mutUniform invokes expr(pset=pset, type_=type_) — keep kwarg names
        return generate_tree(pset, terminal_factory,
                             1, subtree_gen_depth, type_)

    def _rand_ind():
        if random.random() < 0.5:
            nodes = generate_tree(pset, terminal_factory,
                                  INIT_DEPTH_MIN, depth_max)
        else:
            nodes = generate_tree(pset, terminal_factory,
                                  max(INIT_DEPTH_MIN, 2), depth_max)
        return creator.Individual(nodes)

    toolbox.register("mate", tools.cxOnePoint)
    toolbox.decorate("mate", gp.staticLimit(
        key=_op.attrgetter("height"), max_value=depth_max))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_PROGRAM_LENGTH))

    toolbox.register("mut_subtree", gp.mutUniform, expr=_gen_subtree, pset=pset)
    toolbox.decorate("mut_subtree", gp.staticLimit(
        key=_op.attrgetter("height"), max_value=depth_max))
    toolbox.decorate("mut_subtree", gp.staticLimit(
        key=len, max_value=MAX_PROGRAM_LENGTH))
    toolbox.register("mut_hoist", mut_hoist)
    toolbox.register("mut_point", make_mut_point(pset, terminal_factory))
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("sel_best", tools.selBest)

    # Operator probabilities — port the legacy normalisation block: scale the
    # variable components so crossover + subtree-mut + hoist + point ≤ 0.999.
    subtree_mut = MUTATION_BOOST if seed_programs else 0.10
    fixed_ops = HOIST_MUTATION + POINT_MUTATION
    variable_p = p_crossover + subtree_mut
    target_variable_p = 0.998 - fixed_ops
    if variable_p > target_variable_p and variable_p > 0:
        scale = target_variable_p / variable_p
        p_crossover = float(f"{p_crossover * scale:.4f}")
        subtree_mut = float(f"{subtree_mut * scale:.4f}")
        logger.info("[Fold %d] Probabilities normalized (scale: %.4f) | "
                    "Sum: %.4f", fold, scale,
                    p_crossover + subtree_mut + fixed_ops)
    op_thresholds = np.cumsum([p_crossover, subtree_mut,
                               HOIST_MUTATION, POINT_MUTATION])

    X_all = np.ascontiguousarray(X_train.values, dtype=np.float64)
    y_all = np.asarray(y_train.values, dtype=np.float64)
    n_rows = X_all.shape[0]
    row_rng = np.random.RandomState(fold + 7000)
    bag_size = max(2, int(MAX_SAMPLES * n_rows))

    def _row_sample():
        idx = row_rng.choice(n_rows, size=bag_size, replace=False)
        return X_all[idx], y_all[idx]

    def _evaluate(pop, parsimony):
        X_bag, y_bag = _row_sample()
        _evaluate_population(pop, X_bag, y_bag, parsimony, feat_index,
                             context, pearson_w, direction_w, n_jobs)

    # ── Initial population: seeds (once each) + anti-conv randoms + fresh ─
    init_pop = []
    n_seeded = 0
    n_anticonv = 0
    if seed_programs:
        n_seeds = min(len(seed_programs), int(SEED_FRACTION * POPULATION_SIZE))
        for seed in seed_programs[:n_seeds]:
            init_pop.append(_sanitize_seed(seed, pset, feat_terms,
                                           feat_index, n_features))
            n_seeded += 1
        if ANTI_CONVERGENCE_FRACTION > 0:
            n_random = max(1, int(ANTI_CONVERGENCE_FRACTION *
                                  SEED_FRACTION * POPULATION_SIZE))
            for _ in range(n_random):
                init_pop.append(_rand_ind())
                n_anticonv += 1
        logger.info("[Fold %d] Starting 3-Phase Seeded Run: %d unique seeds "
                    "+ %d anti-conv randoms.", fold, n_seeded, n_anticonv)
    else:
        logger.info("[Fold %d] No seeds — starting phased cold-start "
                    "evolution.", fold)

    while len(init_pop) < POPULATION_SIZE:
        init_pop.append(_rand_ind())

    elite_n = max(1, int(ELITE_FRACTION * POPULATION_SIZE))

    def _vary(population):
        offspring = []
        need = POPULATION_SIZE - elite_n
        while len(offspring) < need:
            roll = random.random()
            if roll < op_thresholds[0]:                      # crossover
                done = False
                for _ in range(4):
                    p1, p2 = toolbox.select(population, 2)
                    c1, c2 = toolbox.clone(p1), toolbox.clone(p2)
                    try:
                        toolbox.mate(c1, c2)
                    except (OverflowError, ValueError):
                        # staticLimit overflow OR cxOnePoint's occasional
                        # invalid tail-swap on PrimitiveTree — retry.
                        continue
                    del c1.fitness.values, c2.fitness.values
                    offspring.extend((c1, c2) if len(offspring) + 2 <= need
                                     else (c1,))
                    done = True
                    break
                if not done:
                    offspring.append(toolbox.clone(toolbox.select(population, 1)[0]))
            elif roll < op_thresholds[1]:                    # subtree mutation
                parent = toolbox.select(population, 1)[0]
                child = toolbox.clone(parent)
                try:
                    toolbox.mut_subtree(child)
                except OverflowError:
                    pass                                      # keep unmutated
                del child.fitness.values
                offspring.append(child)
            elif roll < op_thresholds[2]:                    # hoist mutation
                child = toolbox.clone(toolbox.select(population, 1)[0])
                toolbox.mut_hoist(child)
                del child.fitness.values
                offspring.append(child)
            elif roll < op_thresholds[3]:                    # point mutation
                child = toolbox.clone(toolbox.select(population, 1)[0])
                toolbox.mut_point(child)
                del child.fitness.values
                offspring.append(child)
            else:                                            # reproduction
                offspring.append(toolbox.clone(toolbox.select(population, 1)[0]))
        return offspring

    # ── Evolution loop with phased parsimony ──────────────────────────────
    # μ+λ generational replacement: selection pressure lives ONLY in parent
    # choice (tournament), elites are carried unchanged, and the rest of the
    # next generation is fresh offspring. This mirrors gplearn's full
    # generational turnover; refilling survivors via tournament truncation
    # (the naive μ+λ variant) homogenises the population within a few
    # generations and starves extract_elite_programs of unique formulas.
    population = init_pop
    best_overall = None
    best_overall_key = -math.inf

    for gen in range(GENERATIONS):
        parsimony = _phase_parsimony(gen, hp)
        _evaluate(population, parsimony)

        gen_best = max(population, key=lambda i: i.fitness.values[0])
        # Track the best BLOAT-ELIGIBLE program: if the global best exceeds
        # MAX_PROGRAM_LENGTH, fall back to the best program under the cap
        # instead of discarding the entire evolution (legacy behaviour).
        eligible = [i for i in population if len(i) <= MAX_PROGRAM_LENGTH]
        if eligible:
            gen_best = max(eligible, key=lambda i: i.fitness.values[0])
        else:
            gen_best = max(population, key=lambda i: i.fitness.values[0])
        gen_best_key = gen_best.fitness.values[0]
        if gen_best_key > best_overall_key:
            best_overall_key = gen_best_key
            best_overall = toolbox.clone(gen_best)

        mean_len = sum(len(i) for i in population) / len(population)
        logger.info(
            "[Fold %d] Gen %3d/%d | phase-parsimony %.5f | best=%.5f "
            "(raw=%.5f, len=%d) | mean_len=%.1f",
            fold, gen + 1, GENERATIONS, parsimony, gen_best_key,
            getattr(gen_best, "raw_score", float("nan")),
            len(gen_best), mean_len)

        if gen + 1 < GENERATIONS:
            offspring = _vary(population)
            population = (toolbox.sel_best(population, elite_n) +
                          offspring[:POPULATION_SIZE - elite_n])

    assert best_overall is not None

    # Length Check: Reject programs exceeding bloat limit (legacy contract)
    if len(best_overall) > MAX_PROGRAM_LENGTH:
        logger.warning("[Fold %d] Formula length %d > MAX=%d — rejecting.",
                       fold, len(best_overall), MAX_PROGRAM_LENGTH)
        return None

    result = GPResult(best_overall, population, feat_index, feature_names,
                      context)

    # Phase-1: Post-hoc trade-simulation selection on train set
    if POSTHOC_TRADE_EVAL and df_raw_train is not None:
        try:
            _select_best_by_trade_fitness(
                result, X_train, df_raw_train, fold,
                _cfg("ENTRY_PCT", 85), _cfg("EXIT_PCT", 15),
                _cfg("TP_ATR_MULT", 2.4), _cfg("SL_ATR_MULT", 1.9),
                _cfg("ATR_PERIOD", 20),
            )
        except Exception as exc:
            logger.warning("[Fold %d] Post-hoc trade fitness selection "
                           "failed: %s", fold, exc)

    best = result.best_formula
    logger.info("[Fold %d] Best formula: %s", fold, best)
    print(f"\n[Fold {fold}] Best Formula: {best}")
    return result
