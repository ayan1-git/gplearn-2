# =============================================================================
# PIPELINE CONFIGURATION  (trade_discovery/src/config.py)
# =============================================================================

# DATA
DATAPATH = "data/NIFTY 50_30minute 1(in).csv"

# TRIPLE BARRIER METHOD
ORACLE_MAX_HOLD = 52
TP_ATR_MULT     = 4.0
SL_ATR_MULT     = 1.7

# EXECUTION FRICTION
FEE_PER_SIDE = 0.0003
SLIPPAGE     = 0.0001

# SIGNAL THRESHOLDS
ENTRY_PCT            = 80
EXIT_PCT             = 20
ABSOLUTE_EDGE_FLOOR  = 0.01
CAUSAL_RANK_WINDOW   = 500

# WALK-FORWARD
TRAIN_MONTHS    = 60
TEST_MONTHS     = 6
WFO_STEP_MONTHS = 1

# GP ENGINE
GP_POPULATION_SIZE = 3000
GP_GENERATIONS     = 60        # total  — must equal sum of all 3 phases
GP_TOURNAMENT_SIZE = 100
GP_INIT_DEPTH_MIN  = 4
GP_INIT_DEPTH_MAX  = 8
GP_SEED_FRACTION   = 0.20
GP_MUTATION_BOOST  = 0.15
GP_FEATURE_PRIOR_ALPHA = 2.0   # Dirichlet smoothing strength

# 3-Phase schedule — MUST sum to GP_GENERATIONS
GP_PHASE1_GENS = 10
GP_PHASE2_GENS = 30
GP_PHASE3_GENS = 20

# FITNESS
FITNESS_PEARSON_WEIGHT   = 0.70
FITNESS_DIRECTION_WEIGHT = 0.30

# ── FORMULA QUALITY GUARDS ──────────────────────────────────────────────────
MIN_FEATURES_IN_FORMULA = 3     # reject formulas using fewer features
MAX_PROGRAM_LENGTH      = 80    # reject bloated trees

# SIGNAL QUALITY GUARDS
# Removed SIGNAL_STD_FLOOR — unique_ratio is the correct degeneracy detector.
# std can legitimately be high (1.0–1.5) for valid boolean-composed signals.
SIGNAL_UNIQUE_FLOOR = 0.15     # raised from 0.005 → rejects true constants

# OOS survivor thresholds
OOS_MIN_RETURN        = 2.0
OOS_MIN_SHARPE        = 1.5
OOS_MAX_DRAWDOWN      = 15.0
OOS_MAX_DRAWDOWN_HIGH_SHARPE = 28.0   # relaxed gate for Sharpe > 3.5
HIGH_SHARPE_THRESHOLD        = 3.5    # above this, apply relaxed DD gate

# Probabilistic Seed Decay
SEED_SOFT_THRESHOLD_SHARPE = 0.5   # keep seeds if Sharpe ≥ 0.5 × OOS_MIN_SHARPE
SEED_DECAY_FRACTION        = 0.50  # retain only top 50% of elites when soft-passing

# NUMERIC REGULARISATION
EPS           = 1e-8
STD_FLOOR     = 1e-6
PF_SMOOTH_K   = 1e-2
PF_MAX        = 100.0
SHARPE_LAMBDA = 0.10
RETURN_LAMBDA = 2.0

# FEATURE ENGINEERING
OB_ATR_MULT  = 0.5
MIN_LONG     = 75
MIN_SHORT    = 75
MIN_TRADES   = 180

# OPTIONAL / FUTURE
GP_RESTARTS      = 3
IMBALANCE_LAMBDA = 2.0
ACTIVITY_FLOOR   = 0.08
ACTIVITY_LAMBDA  = 0.5
