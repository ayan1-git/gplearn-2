from __future__ import annotations

import logging
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sys

# ── Ensure `src/` is resolvable regardless of working directory ───────────────
# src/ lives in project_core/ after the restructure; regime_classifier lives in
# trade_discovery/Audit_scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "project_core"))
sys.path.insert(0, PROJECT_ROOT)
_discovery_root = os.path.dirname(PROJECT_ROOT)
_audit_scripts = os.path.join(_discovery_root, "Audit_scripts")
if _audit_scripts not in sys.path:
    sys.path.insert(0, _audit_scripts)
os.chdir(PROJECT_ROOT)
# ──────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("leaderboard")

# ---------------------------------------------------------------------------
# Config — mirrors src/config.py so grading thresholds are consistent
# ---------------------------------------------------------------------------
try:
    import src.config as cfg

    OOS_MIN_RETURN         = cfg.OOS_MIN_RETURN
    OOS_MIN_SHARPE         = cfg.OOS_MIN_SHARPE
    OOS_MAX_DRAWDOWN       = cfg.OOS_MAX_DRAWDOWN
    HIGH_SHARPE_THRESHOLD  = cfg.HIGH_SHARPE_THRESHOLD
    OOS_MAX_DD_HIGH_SHARPE = cfg.OOS_MAX_DRAWDOWN_HIGH_SHARPE
    MIN_TRADES             = cfg.MIN_TRADES
    TEST_MONTHS            = cfg.TEST_MONTHS
except (ImportError, AttributeError):
    OOS_MIN_RETURN         = 2.0
    OOS_MIN_SHARPE         = 1.5
    OOS_MAX_DRAWDOWN       = 15.0
    HIGH_SHARPE_THRESHOLD  = 3.5
    OOS_MAX_DD_HIGH_SHARPE = 28.0
    MIN_TRADES             = 180
    TEST_MONTHS            = 6

# ---------------------------------------------------------------------------
# Grading thresholds
# ---------------------------------------------------------------------------
FW_SURVIVAL_GRADE_A   = 0.60   # ≥60% of subsequent folds are Sharpe-positive
FW_SURVIVAL_GRADE_B   = 0.30
REGIME_GRADE_A        = 2      # survived in ≥2 distinct regimes
REGIME_GRADE_B        = 1
FEATURE_FRESH_GRADE_A = 0.60   # ≥60% of formula's features still appear in recent winners
FEATURE_FRESH_GRADE_B = 0.30
SHARPE_STD_GRADE_A    = 0.80   # std of Sharpe across forward folds ≤0.80 → stable
SHARPE_STD_GRADE_B    = 1.50
MIN_FORWARD_FOLDS     = 2      # need at least this many post-discovery folds to grade

# Kelly position sizing caps
KELLY_CAP_A = 0.20
KELLY_CAP_B = 0.10
KELLY_CAP_C = 0.00   # paper trade only

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs")
WF_PARQUET = OUTPUT_DIR / "winning_formulas.parquet"
FM_PARQUET = OUTPUT_DIR / "fold_metadata.parquet"


# ===========================================================================
# DATA LOADERS
# ===========================================================================

def load_winning_formulas(path: Path = WF_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"winning_formulas.parquet not found at '{path}'.\n"
            f"Run main_pipeline.py first to generate WFO outputs."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded %d winning formulas from %s", len(df), path)
    _validate_wf_schema(df)
    return df


def _validate_wf_schema(df: pd.DataFrame) -> None:
    required = {"fold", "formula_hash", "return_pct", "sharpe",
                "win_rate", "max_dd", "profit_factor",
                "buy_threshold", "sell_threshold",
                "n_long", "n_short", "coverage_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"winning_formulas.parquet is missing columns: {missing}")


def load_fold_metadata(path: Path = FM_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"fold_metadata.parquet not found at '{path}'.\n"
            f"Run main_pipeline.py first."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded fold metadata — %d rows, %d folds",
                len(df), df['fold'].nunique())
    _validate_fm_schema(df)
    return df


def _validate_fm_schema(df: pd.DataFrame) -> None:
    required = {"fold", "return_pct", "sharpe", "max_dd",
                "win_rate", "n_features", "prog_length",
                "coverage_pct", "winner", "regime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fold_metadata.parquet is missing columns: {missing}")


# ===========================================================================
# FEATURE EXTRACTION  (word-boundary regex — fixes substring false-positives)
# ===========================================================================

try:
    from src.feature_engineering import PASSTHROUGH_FEATURES, SCALE_FEATURES
    ALL_KNOWN_FEATURES = PASSTHROUGH_FEATURES + SCALE_FEATURES
except ImportError:
    ALL_KNOWN_FEATURES = [
        "feat_ob_supp_active", "feat_ob_res_active",
        "feat_session_sin", "feat_session_cos",
        "feat_icp", "feat_efficiency",
        "feat_ob_supp_touches", "feat_ob_res_touches",
        "feat_momentum_rsi", "feat_rejection_upper",
        "feat_rejection_lower", "feat_local_structure",
        "feat_session_gap",
        "feat_volatility_regime", "feat_dist_skew",
        "feat_zscore", "feat_momentum_mds",
        "feat_vol_asymmetry", "feat_ob_dist_supp",
        "feat_ob_dist_res", "feat_vol_squeeze",
    ]
    logger.warning("src.feature_engineering not importable — using hardcoded feature list.")


def extract_features_from_formula(formula_str: str) -> List[str]:
    """
    Extract feature names from a formula string using word-boundary regex.
    Prevents 'feat_rsi' matching inside 'feat_momentum_rsi'.
    """
    found = []
    for feat in ALL_KNOWN_FEATURES:
        pattern = r'(?<![\w])' + re.escape(feat) + r'(?![\w])'
        if re.search(pattern, formula_str):
            found.append(feat)
    return found


# ===========================================================================
# FORWARD-WALK SIMULATION
# ===========================================================================

def compute_forward_walk_stats(
    discovery_fold: int,
    fold_meta: pd.DataFrame,
) -> Dict:
    """
    Structural forward-walk: uses fold_metadata Sharpe/Return as a proxy
    for how the market continued to be exploitable after discovery.

    NOTE: This is NOT a true re-backtest of the formula (that would require
    storing the gp_model object). It is a market-structure signal: if later
    folds had positive Sharpe winners, the discovered formula's signal class
    had a higher probability of remaining valid. The fw_survival_rate
    measures what % of post-discovery folds the market was "in regime".
    """
    future_folds = fold_meta[fold_meta['fold'] > discovery_fold].copy()
    n_future = len(future_folds)

    if n_future < MIN_FORWARD_FOLDS:
        return {
            "fw_n_folds": n_future,
            "fw_survival_rate": np.nan,
            "fw_mean_sharpe": np.nan,
            "fw_std_sharpe": np.nan,
            "fw_positive_folds": 0,
            "fw_winner_folds": 0,
            "fw_sharpe_decay": np.nan,
            "fw_data_sufficient": False,
        }

    fw_positive = int((future_folds['sharpe'] > 0).sum())
    fw_winners  = int(future_folds['winner'].sum())
    mean_sh     = float(future_folds['sharpe'].mean())
    std_sh      = float(future_folds['sharpe'].std())

    # Linear regression slope of Sharpe vs fold index
    # Negative = decaying market structure; positive = improving
    x = np.arange(n_future, dtype=float)
    y = future_folds['sharpe'].values.astype(float)
    slope = float(np.polyfit(x, y, 1)[0]) if (n_future >= 3 and np.std(y) > 1e-8) else 0.0

    return {
        "fw_n_folds":          n_future,
        "fw_survival_rate":    float(fw_positive / n_future),
        "fw_mean_sharpe":      mean_sh,
        "fw_std_sharpe":       std_sh,
        "fw_positive_folds":   fw_positive,
        "fw_winner_folds":     fw_winners,
        "fw_sharpe_decay":     slope,
        "fw_data_sufficient":  True,
    }


# ===========================================================================
# FEATURE FRESHNESS
# ===========================================================================

def compute_feature_freshness(
    formula_features: List[str],
    discovery_fold: int,
    winning_formulas: pd.DataFrame,
    fold_meta: pd.DataFrame,
    recency_window: int = 5,
) -> float:
    """
    What fraction of this formula's features are still appearing
    in the most recent `recency_window` winning formulas?

    Returns float in [0, 1], or np.nan if unresolvable.
    """
    if not formula_features:
        return 0.0

    subsequent_winners = (
        winning_formulas[winning_formulas['fold'] > discovery_fold]
        .sort_values('fold')
        .tail(recency_window)
    )

    if subsequent_winners.empty:
        return np.nan

    # If formula strings are present in parquet (optional column)
    if 'formula' in subsequent_winners.columns:
        recent_features: List[str] = []
        for _, row in subsequent_winners.iterrows():
            recent_features.extend(extract_features_from_formula(str(row['formula'])))
        recent_feature_set = set(recent_features)
        if not recent_feature_set:
            return 0.0
        overlap = sum(1 for f in formula_features if f in recent_feature_set)
        return float(overlap / len(formula_features))

    # Fallback: no formula strings in parquet — return neutral uncertainty
    return 0.5


# ===========================================================================
# REGIME COVERAGE
# ===========================================================================

def compute_regime_coverage(
    discovery_fold: int,
    formula_discovery_regime: str,
    fold_meta: pd.DataFrame,
) -> Tuple[int, List[str]]:
    """
    Count distinct market regimes in which the formula's signal class was viable.
    Discovery regime counts as 1 (the formula was tested there).
    Any subsequent fold with a winner in a different regime adds to the count.
    """
    regimes_seen = {formula_discovery_regime} if formula_discovery_regime else set()

    future_winners = fold_meta[
        (fold_meta['fold'] > discovery_fold) & (fold_meta['winner'])
    ]
    for regime in future_winners['regime'].dropna().unique():
        regimes_seen.add(str(regime))

    return len(regimes_seen), sorted(regimes_seen)


# ===========================================================================
# PERMUTATION SIGNIFICANCE TEST
# ===========================================================================

def permutation_test_sharpe(
    sharpe: float,
    n_trades: int,
    n_permutations: int = 2000,
    seed: int = 42,
) -> float:
    """
    Monte-Carlo null test: simulate N random strategies with zero true edge
    and the same trade count. Returns the p-value (fraction of null strategies
    whose Sharpe ≥ observed Sharpe).

    A Sharpe of 1.5 on 200 trades is far more significant than on 12 trades.
    Accept: p < 0.05. Reject for live trading: p > 0.10.
    """
    if n_trades < 10:
        return 1.0

    rng = np.random.default_rng(seed)
    # Correct MC loop for faster performance
    null_sharpes = []
    for _ in range(n_permutations):
        r = rng.standard_normal(n_trades)
        std_r = np.std(r)
        if std_r < 1e-8:
            null_sharpes.append(0.0)
        else:
            null_sharpes.append(np.mean(r) / std_r * np.sqrt(n_trades))
    
    null_sharpes = np.array(null_sharpes)
    return float(np.mean(null_sharpes >= sharpe))


# ===========================================================================
# POSITION SIZING  (fractional Kelly, grade-capped)
# ===========================================================================

def fractional_kelly(
    win_rate: float,
    profit_factor: float,
    kelly_fraction: float = 0.25,
    cap: float = 0.20,
) -> float:
    """
    Full Kelly = (W*b - L) / b
    We use 25% fractional Kelly to reduce variance.
    Hard-capped at `cap` to prevent ruin in adverse regimes.
    """
    if profit_factor <= 1.0 or not (0 < win_rate < 100):
        return 0.0
    W = win_rate / 100.0
    L = 1.0 - W
    b = profit_factor - 1.0 # b is the odds (profit factor - 1 for typical PF definition)
    if b <= 0: return 0.0
    full_kelly = (W * b - L) / b
    if full_kelly <= 0:
        return 0.0
    return float(min(full_kelly * kelly_fraction, cap))


# ===========================================================================
# KILL-SWITCH PARAMETERS
# ===========================================================================

def compute_kill_switch_params(
    grade: str,
    sharpe: float,
    max_dd: float,
) -> Dict:
    """
    Define live-trading kill conditions per grade.
    These are guardrails — not stop-losses on individual trades.
    They trigger a full strategy suspension for human review.
    """
    if grade == "A":
        return {
            "dd_kill_pct":          min(max_dd * 2.0, OOS_MAX_DRAWDOWN * 1.5),
            "consecutive_loss_kill": 8,
            "sharpe_rolling_kill":  max(0.0, sharpe * 0.30),
            "review_after_folds":   3,
        }
    elif grade == "B":
        return {
            "dd_kill_pct":          min(max_dd * 1.5, OOS_MAX_DRAWDOWN),
            "consecutive_loss_kill": 5,
            "sharpe_rolling_kill":  max(0.0, sharpe * 0.50),
            "review_after_folds":   2,
        }
    else:  # C / DISCARD
        return {
            "dd_kill_pct":          5.0,
            "consecutive_loss_kill": 3,
            "sharpe_rolling_kill":  0.0,
            "review_after_folds":   1,
        }


# ===========================================================================
# CURRENT REGIME CHECK
# ===========================================================================

def get_current_regime(data_path: Optional[str] = None) -> Optional[str]:
    """
    Classify the current live market regime from the last 3 months of data.
    Returns the regime string or None if data unavailable.
    """
    try:
        from regime_classifier import classify_regime
        import src.config as cfg

        dp = data_path or cfg.DATAPATH
        if not os.path.exists(dp):
            logger.warning("Data not found at '%s' — skipping regime check.", dp)
            return None

        df_raw = pd.read_csv(dp, parse_dates=['datetime'], index_col='datetime')
        df_raw.sort_index(inplace=True)
        df_raw.columns = [c.lower() for c in df_raw.columns]

        # Last 3 months of bars
        recent = df_raw.iloc[-int(cfg.FE_BAR_PER_DAY * 22 * 3):]
        regime = classify_regime(recent)
        logger.info("Current market regime detected: %s", regime)
        return regime

    except Exception as exc:
        logger.warning("Could not determine current regime: %s", exc)
        return None


# ===========================================================================
# GRADING ENGINE
# ===========================================================================

def grade_formula(
    fw_survival_rate: float,
    n_regimes: int,
    feature_freshness: float,
    sharpe_std: float,
    fw_data_sufficient: bool,
    p_value: float,
    fw_sharpe_decay: float,
) -> Tuple[str, List[str]]:
    """
    Multi-criteria scoring → final grade.

    Grade A  → Trade at full Kelly-capped size
    Grade B  → Reduced size with kill-switch active
    Grade C  → Paper trade only, no capital
    DISCARD  → Archive, do not use
    """
    reasons: List[str] = []
    strikes = 0

    # ── Hard DISCARD gates ────────────────────────────────────────────────
    if p_value > 0.10:
        return "DISCARD", [f"Statistically insignificant (p={p_value:.3f} > 0.10)"]

    if fw_data_sufficient and fw_survival_rate < 0.10:
        return "DISCARD", [f"Forward-walk collapse (survival={fw_survival_rate:.0%})"]

    if fw_data_sufficient and fw_sharpe_decay < -0.20:
        reasons.append(f"Severe Sharpe decay (slope={fw_sharpe_decay:.3f})")
        strikes += 2

    # ── Positive signal scoring ───────────────────────────────────────────
    a_signals = 0

    if fw_data_sufficient:
        if fw_survival_rate >= FW_SURVIVAL_GRADE_A:
            a_signals += 1
        elif fw_survival_rate >= FW_SURVIVAL_GRADE_B:
            reasons.append(f"Moderate forward-walk survival ({fw_survival_rate:.0%})")
        else:
            reasons.append(f"Weak forward-walk survival ({fw_survival_rate:.0%})")
            strikes += 1
    else:
        reasons.append("Insufficient forward-walk data (too few post-discovery folds)")
        strikes += 1

    if n_regimes >= REGIME_GRADE_A:
        a_signals += 1
    elif n_regimes >= REGIME_GRADE_B:
        reasons.append(f"Single-regime formula (regimes: {n_regimes})")
    else:
        reasons.append("No regime coverage data")
        strikes += 1

    if not np.isnan(feature_freshness):
        if feature_freshness >= FEATURE_FRESH_GRADE_A:
            a_signals += 1
        elif feature_freshness >= FEATURE_FRESH_GRADE_B:
            reasons.append(f"Moderate feature freshness ({feature_freshness:.0%})")
        else:
            reasons.append(f"Stale features — only {feature_freshness:.0%} still active")
            strikes += 1
    else:
        reasons.append("Feature freshness unknown (no formula string in parquet)")

    if not np.isnan(sharpe_std):
        if sharpe_std <= SHARPE_STD_GRADE_A:
            a_signals += 1
        elif sharpe_std <= SHARPE_STD_GRADE_B:
            reasons.append(f"Moderate Sharpe volatility (std={sharpe_std:.2f})")
        else:
            reasons.append(f"Inconsistent Sharpe across folds (std={sharpe_std:.2f})")
            strikes += 1

    # ── Final grade ────────────────────────────────────────────────────────
    if strikes >= 3:
        grade = "DISCARD"
    elif a_signals >= 3 and strikes == 0:
        grade = "A"
    elif a_signals >= 2 and strikes <= 1:
        grade = "B"
    else:
        grade = "C"

    return grade, reasons


# ===========================================================================
# MAIN LEADERBOARD BUILDER
# ===========================================================================

def build_leaderboard(
    winning_formulas: pd.DataFrame,
    fold_meta: pd.DataFrame,
    current_regime: Optional[str] = None,
) -> pd.DataFrame:
    """
    Core scoring loop. Produces a fully scored, graded, and ranked DataFrame.
    One row per winning formula.
    """
    max_fold = fold_meta['fold'].max()
    records  = []

    logger.info("Scoring %d formulas across %d total folds ...",
                len(winning_formulas), max_fold)

    for _, row in winning_formulas.iterrows():
        disc_fold     = int(row['fold'])
        fhash         = str(row['formula_hash'])
        sharpe        = float(row['sharpe'])
        ret_pct       = float(row['return_pct'])
        max_dd        = float(row['max_dd'])
        win_rate      = float(row['win_rate'])
        profit_factor = float(row['profit_factor'])
        n_long        = int(row['n_long'])
        n_short       = int(row['n_short'])
        n_trades      = n_long + n_short
        coverage_pct  = float(row['coverage_pct'])

        # Discovery fold regime
        disc_meta   = fold_meta[fold_meta['fold'] == disc_fold]
        disc_regime = str(disc_meta['regime'].iloc[0]) if not disc_meta.empty else "unknown"

        # Formula features (only if 'formula' column present in parquet)
        formula_str = str(row.get('formula', ''))
        feat_list   = extract_features_from_formula(formula_str) if formula_str else []
        n_features  = int(row.get('n_features', len(feat_list)))

        # ── Forward-walk ──────────────────────────────────────────────────
        fw = compute_forward_walk_stats(disc_fold, fold_meta)

        # ── Regime coverage ───────────────────────────────────────────────
        n_regimes, regimes_list = compute_regime_coverage(
            disc_fold, disc_regime, fold_meta
        )

        # ── Feature freshness ─────────────────────────────────────────────
        feat_fresh = compute_feature_freshness(
            feat_list, disc_fold, winning_formulas, fold_meta
        )

        # ── Statistical significance ──────────────────────────────────────
        p_val = permutation_test_sharpe(sharpe, n_trades)

        # ── Regime match ──────────────────────────────────────────────────
        regime_match = (
            current_regime is not None and disc_regime == current_regime
        )

        # ── Grade ─────────────────────────────────────────────────────────
        grade, reasons = grade_formula(
            fw_survival_rate   = fw['fw_survival_rate'],
            n_regimes          = n_regimes,
            feature_freshness  = feat_fresh,
            sharpe_std         = fw['fw_std_sharpe'],
            fw_data_sufficient = fw['fw_data_sufficient'],
            p_value            = p_val,
            fw_sharpe_decay    = fw['fw_sharpe_decay'],
        )

        # ── Position sizing ───────────────────────────────────────────────
        grade_cap = {"A": KELLY_CAP_A, "B": KELLY_CAP_B}.get(grade, 0.0)
        pos_size  = fractional_kelly(win_rate, profit_factor, cap=grade_cap)
        # +20% bonus for current regime alignment
        if regime_match and grade in ("A", "B"):
            pos_size = min(pos_size * 1.20, grade_cap)

        # ── Kill-switch ───────────────────────────────────────────────────
        ks = compute_kill_switch_params(grade, sharpe, max_dd)

        # ── Composite rank score (sort key only, not the grade itself) ────
        fw_surv      = fw['fw_survival_rate'] if not np.isnan(fw['fw_survival_rate']) else 0.0
        feat_f       = feat_fresh if not np.isnan(feat_fresh) else 0.5
        sh_std_norm  = 1.0 / (1.0 + (fw['fw_std_sharpe']
                               if not np.isnan(fw['fw_std_sharpe']) else 1.0))
        recency      = float(disc_fold / max_fold) if max_fold > 0 else 0.0

        composite_score = (
            0.35 * fw_surv
            + 0.20 * (n_regimes / 5.0)
            + 0.15 * feat_f
            + 0.10 * sh_std_norm
            + 0.10 * recency
            + 0.10 * (1.0 - p_val)
        )

        records.append({
            # Identity
            "formula_hash":          fhash[:16],
            "discovery_fold":        disc_fold,
            "discovery_regime":      disc_regime,
            # OOS metrics at discovery
            "oos_return_pct":        ret_pct,
            "oos_sharpe":            sharpe,
            "oos_max_dd":            max_dd,
            "oos_win_rate":          win_rate,
            "oos_profit_factor":     profit_factor,
            "n_trades":              n_trades,
            "n_long":                n_long,
            "n_short":               n_short,
            "coverage_pct":          coverage_pct,
            "n_features_used":       n_features,
            # Forward-walk
            "fw_n_folds":            fw['fw_n_folds'],
            "fw_survival_rate":      round(fw_surv, 3),
            "fw_mean_sharpe":        round(fw['fw_mean_sharpe'], 3)
                                     if not np.isnan(fw['fw_mean_sharpe']) else np.nan,
            "fw_std_sharpe":         round(fw['fw_std_sharpe'], 3)
                                     if not np.isnan(fw['fw_std_sharpe']) else np.nan,
            "fw_sharpe_decay":       round(fw['fw_sharpe_decay'], 4)
                                     if not np.isnan(fw['fw_sharpe_decay']) else np.nan,
            "fw_winner_folds":       fw['fw_winner_folds'],
            # Regime
            "n_regimes_covered":     n_regimes,
            "regimes_list":          "|".join(regimes_list),
            "current_regime_match":  regime_match,
            # Feature health
            "feature_freshness":     round(feat_f, 3),
            "features_used":         "|".join(feat_list) if feat_list else "unknown",
            # Significance
            "p_value":               round(p_val, 4),
            "significant_p05":       p_val < 0.05,
            # Grade & sizing
            "grade":                 grade,
            "position_size_frac":    round(pos_size, 4),
            "regime_size_bonus":     regime_match and grade in ("A", "B"),
            "kill_dd_pct":           ks['dd_kill_pct'],
            "kill_consec_losses":    ks['consecutive_loss_kill'],
            "kill_rolling_sharpe":   ks['sharpe_rolling_kill'],
            "review_after_folds":    ks['review_after_folds'],
            # Meta
            "recency_score":         round(recency, 3),
            "composite_score":       round(composite_score, 4),
            "grade_reasons":         " | ".join(reasons) if reasons else "All criteria met",
        })

    df = pd.DataFrame(records)

    # Sort: A → B → C → DISCARD, then by composite_score descending within each grade
    grade_order = {"A": 0, "B": 1, "C": 2, "DISCARD": 3}
    df['_grade_ord'] = df['grade'].map(grade_order)
    df = (df.sort_values(['_grade_ord', 'composite_score'], ascending=[True, False])
            .drop(columns=['_grade_ord'])
            .reset_index(drop=True))
    df.index += 1
    df.index.name = 'rank'
    return df


# ===========================================================================
# REPORT WRITER
# ===========================================================================

def write_report(
    lb: pd.DataFrame,
    current_regime: Optional[str],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    lines = []
    SEP = "=" * 80
    sep = "-" * 80

    lines += [
        SEP,
        "  FORMULA LEADERBOARD — PRODUCTION SELECTION REPORT",
        f"  Generated     : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Current Regime: {current_regime or 'Unknown'}",
        f"  Formulas scored: {len(lb)}",
        SEP,
    ]

    grade_counts = lb['grade'].value_counts()
    grade_labels = {
        'A': 'Trade at full Kelly size',
        'B': 'Trade at reduced size + kill-switch',
        'C': 'Paper trade only',
        'DISCARD': 'Archive — no capital',
    }
    lines.append("  Grade Summary")
    for g in ['A', 'B', 'C', 'DISCARD']:
        lines.append(f"    Grade {g} ({grade_labels[g]}) : {grade_counts.get(g, 0)}")
    lines.append(sep)

    for rank, row in lb.iterrows():
        if row['grade'] == 'DISCARD':
            continue
        lines += [
            f"  RANK #{rank} | Grade: {row['grade']} | Hash: {row['formula_hash']}",
            f"  Discovered: Fold {row['discovery_fold']} | "
            f"Regime: {row['discovery_regime']}",
            f"  OOS: Return={row['oos_return_pct']:.1f}% | "
            f"Sharpe={row['oos_sharpe']:.2f} | "
            f"MaxDD={row['oos_max_dd']:.1f}% | "
            f"WR={row['oos_win_rate']:.1f}% | "
            f"PF={row['oos_profit_factor']:.2f} | "
            f"Trades={row['n_trades']}",
            f"  Forward-Walk : {row['fw_n_folds']} folds | "
            f"Survival={row['fw_survival_rate']:.0%} | "
            f"Sharpe decay slope={row['fw_sharpe_decay']:.4f}",
            f"  Regimes      : {row['n_regimes_covered']} covered "
            f"({row['regimes_list']})",
            f"  Features     : {row['features_used']}",
            f"  Freshness    : {row['feature_freshness']:.0%} | "
            f"p-value: {row['p_value']:.4f} | "
            f"Significant: {row['significant_p05']}",
            f"  ► Position size : {row['position_size_frac']:.1%}"
            + (" [+Regime match bonus]" if row['regime_size_bonus'] else ""),
            f"  ► Kill-switch   : DD>{row['kill_dd_pct']:.1f}% | "
            f"ConsecLoss>{row['kill_consec_losses']} | "
            f"RollingS<{row['kill_rolling_sharpe']:.2f}",
        ]
        if row['grade_reasons'] != "All criteria met":
            lines.append(f"  ⚠ Notes: {row['grade_reasons']}")
        lines.append(sep)

    # Discard section
    discards = lb[lb['grade'] == 'DISCARD']
    if not discards.empty:
        lines += ["", "  DISCARDED (archive only):"]
        for rank, row in discards.iterrows():
            lines.append(
                f"    #{rank} | {row['formula_hash']} | "
                f"Fold {row['discovery_fold']} | {row['grade_reasons']}"
            )

    lines += [SEP, "  END OF REPORT", SEP]

    text = "\n".join(lines)
    out  = output_dir / "leaderboard_report.txt"
    out.write_text(text)
    print(text)
    logger.info("Report written → %s", out)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info("=" * 60)
    logger.info("FORMULA LEADERBOARD ENGINE — START")
    logger.info("=" * 60)

    try:
        winning_formulas = load_winning_formulas()
        fold_meta        = load_fold_metadata()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    if winning_formulas.empty:
        logger.warning("No winning formulas found. Nothing to rank.")
        return

    current_regime = get_current_regime()

    lb = build_leaderboard(winning_formulas, fold_meta, current_regime)

    csv_path     = OUTPUT_DIR / "formula_leaderboard.csv"
    parquet_path = OUTPUT_DIR / "formula_leaderboard.parquet"
    lb.to_csv(csv_path)
    lb.to_parquet(parquet_path, index=True)
    logger.info("Leaderboard → %s + %s", csv_path, parquet_path)

    write_report(lb, current_regime, OUTPUT_DIR)

    tradeable = lb[lb['grade'].isin(['A', 'B'])]
    logger.info("=" * 60)
    logger.info("DONE — %d tradeable formulas (Grade A/B) out of %d total.",
                len(tradeable), len(lb))
    if not tradeable.empty:
        best = tradeable.iloc[0]
        logger.info(
            "TOP PICK → Hash: %s | Grade: %s | Size: %.1f%% | "
            "OOS Sharpe: %.2f | FW Survival: %.0f%%",
            best['formula_hash'], best['grade'],
            best['position_size_frac'] * 100,
            best['oos_sharpe'],
            best['fw_survival_rate'] * 100,
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
