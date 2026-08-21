# =============================================================================
# PIPELINE CONFIGURATION  (trade_discovery/src/config.py)
# =============================================================================

# ── DATA ──────────────────────────────────────────────────────────────────
DATAPATH = "data/NIFTY MID SELECT_15minute.csv"

# ── TRIPLE BARRIER METHOD ─────────────────────────────────────────────────
ORACLE_MAX_HOLD = 26
TP_ATR_MULT     = 2.4
SL_ATR_MULT     = 1.9
ATR_PERIOD      = 20
DROP_NEUTRAL    = True    # If True, rows with target=0 are removed from training
DROP_WHIPSAW    = True    # If True, rows with target=-99 (Both-SL) are removed

# ── EXECUTION FRICTION ────────────────────────────────────────────────────
FEE_PER_SIDE = 0.0003
SLIPPAGE     = 0.0001

# ── SIGNAL THRESHOLDS ─────────────────────────────────────────────────────
ENTRY_PCT            = 85
EXIT_PCT             = 15
ABSOLUTE_EDGE_FLOOR  = 0.01
CAUSAL_RANK_WINDOW   = 500

# ── WALK-FORWARD ──────────────────────────────────────────────────────────
TRAIN_MONTHS    = 6
TEST_MONTHS     = 2
WFO_STEP_MONTHS = 1

# ── GP ENGINE ─────────────────────────────────────────────────────────────
GP_POPULATION_SIZE = 3000
GP_GENERATIONS     = 60        # total  — must equal sum of all 3 phases
GP_TOURNAMENT_SIZE = 7
GP_INIT_DEPTH_MIN  = 4
GP_INIT_DEPTH_MAX  = 8
GP_SEED_FRACTION   = 0.20
GP_MUTATION_BOOST  = 0.15
GP_HOIST_MUTATION  = 0.1
GP_POINT_MUTATION  = 0.1
GP_MAX_SAMPLES     = 0.9   # P2: more data per program → less bagging overfit
GP_FEATURE_PRIOR_ALPHA = 5.0   # Dirichlet smoothing strength

GEL_HOLDOUT_FRACTION   = 0.20
GEL_GENERATIONS        = 50
GEL_SEEDS_PER_GEN      = 200
GEL_ELITE_POOL_SIZE    = 100
GEL_MIN_HOLDOUT_TRADES = 30

# ── DIVERSITY ENFORCEMENT ─────────────────────────────────────────────────
GEL_STALE_RESET_GENS      = 2      # cold-restart elite pool if best unchanged N gens
GEL_MAX_SEED_DUPLICATES   = 2      # max copies of same formula string in elite pool
GEL_DIVERSITY_FRACTION    = 0.50   # fraction of elite pool reserved for diverse programs
GEL_FEATURE_PRIOR_DECAY   = 0.40   # exponential decay on feat_win_counts each gen
GEL_FEATURE_LOSS_PENALTY  = 0.50   # weight applied to feat_loss_counts vs feat_win_counts
GEL_WIN_DECAY_PER_GEN     = 0.70   # per-gen multiplicative decay on feat_win_counts (breaks
                                   # the one-way ratchet where a feature in every winner
                                   # accumulates unbounded prior mass, e.g. feat_icp)
GEL_PRIOR_MAX_CONCENTRATION = 1.8  # cap single-feature prior mass at Nx the uniform baseline
                                   # (was 3.0 — too permissive, let one feature dominate)
GEL_MAX_WINNER_SEED_FRACTION = 0.40  # max fraction of the seed pool taken by winner programs.
                                     # The rest is diverse in-sample elite. Prevents the pool
                                     # from being flooded by one validated lineage (which made
                                     # the GP re-derive math-equivalent permutations forever).
GEL_SIG_STALE_RESET_GENS  = 2      # nuke pool if N consecutive gens produce the SAME OOS
                                   # performance signature (equivalent-winner permutation loop)

# 3-Phase schedule — MUST sum to GP_GENERATIONS
GP_PHASE1_GENS = 15
GP_PHASE2_GENS = 25
GP_PHASE3_GENS = 20

# ── GP FITNESS ────────────────────────────────────────────────────────────
FITNESS_PEARSON_WEIGHT   = 0.70
FITNESS_DIRECTION_WEIGHT = 0.30
FITNESS_ATTR_PENALTY_WEIGHT = 0.25   # penalty for wrong-sign predictions
FITNESS_POSTHOC_TRADE_EVAL  = False  # P2: disable train-only trade-Sharpe selection (overfit)
# P2-fix: penalise near-constant (tiny-magnitude) predictors. Pearson is
# scale-invariant, so without this the GP collapses onto mul(vol,osc)-style
# formulas whose output is ~0 → 0 executable trades on the holdout.
FITNESS_SPREAD_FLOOR  = 0.08   # std(y_pred) below this is penalised
FITNESS_SPREAD_WEIGHT = 5.0    # strength of the spread penalty

# ── FORMULA QUALITY GUARDS ────────────────────────────────────────────────
MIN_FEATURES_IN_FORMULA = 3     # reject formulas using fewer features
MAX_PROGRAM_LENGTH      = 40    # P2: hard bloat cap (reject trees > 40 nodes)

# ── SIGNAL QUALITY GUARDS ─────────────────────────────────────────────────
SIGNAL_UNIQUE_FLOOR = 0.10      # raised from 0.05 → rejects near-constant signals
SIGNAL_SEP_FLOOR    = 0.05      # min (85th-15th pct) threshold separation (rejects ~0-mag)

# ── OOS SURVIVOR THRESHOLDS ───────────────────────────────────────────────
# GEL survivor gate — relaxed (2026-07-09) so the loop's feedback mechanisms
# (P1 winner-seeding, P3 feature prior) can engage. The holdout regime
# (2025-07-18→2026-01-23) has an empirical ceiling of ~0.9% return / ~0.9
# Sharpe for this signal class, so the old 1.5% / 1.2 floors were unreachable
# every generation → winners=0 → prior stayed uniform → evolution was unguided.
# PF>1.1 (below) is now the binding economic gate (genuine edge exists).
OOS_MIN_RETURN        = 0.5
OOS_MIN_SHARPE        = 0.3
OOS_MAX_DRAWDOWN      = 25.0
OOS_MAX_DRAWDOWN_HIGH_SHARPE = 35.0   # relaxed gate for Sharpe > 3.5
HIGH_SHARPE_THRESHOLD        = 3.5    # above this, apply relaxed DD gate
MIN_OOS_TRADES               = 30     # reject strategies with insufficient OOS sample
OOS_MIN_PROFIT_FACTOR         = 1.1   # minimum profit factor to survive
OOS_MIN_COVERAGE_PCT          = 15.0  # minimum executable coverage % to survive

# ── PROBABILISTIC SEED DECAY ──────────────────────────────────────────────
SEED_SOFT_THRESHOLD_SHARPE = 0.5   # keep seeds if Sharpe ≥ 0.5 × OOS_MIN_SHARPE
SEED_DECAY_FRACTION        = 0.50  # retain only top 50% of elites when soft-passing

# ── FEATURE ENGINEERING (ENGINE) ──────────────────────────────────────────
OB_ATR_MULT        = 0.5
OB_INTERNAL_LB     = 5
OB_SWING_LB        = 20
OB_MAX_OBS         = 5
OB_IOU_THRESHOLD   = 0.85
FE_MISSING_FILL    = 5.0

# ── FEATURE ENGINEERING (INDICATORS) ──────────────────────────────────────
FE_MOMENTUM_PERIOD   = 14
FE_VOL_SHORT_PERIOD  = 6
FE_VOL_LONG_PERIOD   = 100
FE_SKEW_PERIOD       = 28
FE_ZSCORE_PERIOD     = 50
FE_ICP_PERIOD        = 14
FE_MDS_FAST_WINDOW   = 5
FE_MDS_SLOW_WINDOW   = 30
FE_VOL_ASYM_WINDOW   = 20
FE_STOCH_PERIOD      = 14
FE_ADX_PERIOD        = 14
FE_BAR_PER_DAY       = 13

# ── REGIME CLASSIFICATION ─────────────────────────────────────────────────
HURST_TREND_THRESH  = 0.55
HURST_MR_THRESH     = 0.45
ADX_TREND_THRESH    = 25.0
CHOP_TREND_THRESH   = 38.2
CHOP_CHOPPY_THRESH  = 61.8
CHOP_PERIOD         = 28

# ── PIPELINE ROTATION ─────────────────────────────────────────────────────
ROTATION_FEATURE = "feat_vol_asymmetry"

# ── REGIME HYPERPARAMETERS ───────────────────────────────────────────────
REGIME_HP: dict = {
    # Strong trend: prefer longer trees; aggressive crossover for recombination
    "trending": {
        "parsimony_p1":   0.001,
        "parsimony_p2":   0.0003,
        "parsimony_p3":   0.005,
        "p_crossover":    0.70,
        "depth_max":      6,             # GP_INIT_DEPTH_MAX
        "tournament_size": 12,
        "fitness_pearson_w":   0.40,
        "fitness_direction_w": 0.60,
    },
    # Mean-revert: hard complexity penalty — short precise rules generalise better
    "mean_reverting": {
        "parsimony_p1":   0.001,
        "parsimony_p2":   0.003,
        "parsimony_p3":   0.008,
        "p_crossover":    0.55,
        "depth_max":      6,
        "tournament_size": 7,
    },
    # Choppy: strongest length penalty + small tournaments → diversity pressure
    "choppy_random_walk": {
        "parsimony_p1":   0.002,
        "parsimony_p2":   0.005,
        "parsimony_p3":   0.010,
        "p_crossover":    0.50,
        "depth_max":      5,
        "tournament_size": 5,
    },
    # Trending but noisy: balanced
    "trending_random_walk": {
        "parsimony_p1":   0.001,   # P2: penalise bloat from the start
        "parsimony_p2":   0.004,   # P2: stronger mid-run complexity pressure
        "parsimony_p3":   0.010,   # P2: hard length penalty late
        "p_crossover":    0.65,
        "depth_max":      6,       # P2: cap tree depth (was 7)
        "tournament_size": 10,
    },
    # Fallback / uncertain
    "random_walk": {
        "parsimony_p1":   0.001,
        "parsimony_p2":   0.0005,
        "parsimony_p3":   0.003,
        "p_crossover":    0.60,
        "depth_max":      8,             # GP_INIT_DEPTH_MAX
        "tournament_size": 7,
    },
}

# ── NUMERIC REGULARISATION ───────────────────────────────────────────────
GT_SOFT_SCALE = 3.0
EPS           = 1e-8
STD_FLOOR     = 1e-6
PF_SMOOTH_K   = 1e-2
PF_MAX        = 100.0
SHARPE_LAMBDA = 0.10
RETURN_LAMBDA = 2.0

# ── CONSTRAINTS ──────────────────────────────────────────────────────────
SESSION_OPEN = "09:15"
SESSION_CLOSE = "15:30"
SESSION_TZ = "Asia/Kolkata"

MIN_LONG     = 75
MIN_SHORT    = 75
MIN_TRADES   = 180

# ── OPTIONAL / FUTURE ─────────────────────────────────────────────────────
GP_RESTARTS      = 3
IMBALANCE_LAMBDA = 2.0
ACTIVITY_FLOOR   = 0.08
ACTIVITY_LAMBDA  = 0.5

# ── TALIB FEATURE EXPANSION ───────────────────────────────────────────────────
USE_TALIB_FEATURES = True   # set False to skip without touching feature_engineering.py
