"""
main_pipeline_gel.py — Global Evolutionary Loop (GEL) Orchestrator

Architecture:
  - One fixed temporal split: train on full history, test on fixed future holdout (last HOLDOUT_FRACTION).
  - Each GEL generation trains GP on the FULL train set, seeds from the previous generation's elite pool.
  - A formula must survive on the FIXED holdout (2024-present) to be a winner — not just on a 2-month slice.
  - Elite pool (top SEEDS_PER_GEN programs by fitness) always passes forward, regardless of OOS outcome.
  - Winners are ranked by Sharpe and capped at ELITE_POOL_SIZE.

This replaces the WFO (walk_forward_optimization) paradigm entirely.
No other src/ files need changes.
"""
import copy
import gc
import logging
import os
import sys

# ── Ensure project paths are resolvable regardless of working directory ───────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TRADE_DISCOVERY_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(TRADE_DISCOVERY_ROOT, ".."))
for candidate in (PROJECT_ROOT, TRADE_DISCOVERY_ROOT, WORKSPACE_ROOT):
    if candidate and candidate not in sys.path:
        sys.path.insert(0, candidate)
os.chdir(PROJECT_ROOT)  # ensures all relative paths (data/, outputs/) work
# ──────────────────────────────────────────────────────────────────────────────
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── CONFIG ───────────────────────────────────────────────────────────────────
import src.config as cfg

ORACLE_MAX_HOLD     = cfg.ORACLE_MAX_HOLD
TP_ATR_MULT         = cfg.TP_ATR_MULT
SL_ATR_MULT         = cfg.SL_ATR_MULT
DATAPATH            = cfg.DATAPATH
ENTRY_PCT           = cfg.ENTRY_PCT
EXIT_PCT            = cfg.EXIT_PCT
MIN_FEATURES        = cfg.MIN_FEATURES_IN_FORMULA
MAX_PROG_LEN        = cfg.MAX_PROGRAM_LENGTH
SIGNAL_UNIQUE_FLOOR = cfg.SIGNAL_UNIQUE_FLOOR
SIGNAL_SEP_FLOOR    = cfg.SIGNAL_SEP_FLOOR

OOS_MIN_RETURN   = getattr(cfg, 'OOS_MIN_RETURN', 1.5)
OOS_MIN_SHARPE   = getattr(cfg, 'OOS_MIN_SHARPE', 1.2)
OOS_MAX_DRAWDOWN = getattr(cfg, 'OOS_MAX_DRAWDOWN', 25.0)
OOS_MIN_PROFIT_FACTOR = getattr(cfg, 'OOS_MIN_PROFIT_FACTOR', 1.1)
OOS_MIN_COVERAGE_PCT = getattr(cfg, 'OOS_MIN_COVERAGE_PCT', 15.0)
MIN_HOLDOUT_TRADES   = getattr(cfg, 'MIN_OOS_TRADES', 30)

# GEL-SPECIFIC CONFIG ──────────────────────────────────────────────────────
# Add these keys to src/config.py to tune them; defaults are applied below.
HOLDOUT_FRACTION   = getattr(cfg, 'GEL_HOLDOUT_FRACTION',   0.20)
GEL_GENERATIONS    = getattr(cfg, 'GEL_GENERATIONS',        50)
SEEDS_PER_GEN      = getattr(cfg, 'GEL_SEEDS_PER_GEN',      200)
ELITE_POOL_SIZE    = getattr(cfg, 'GEL_ELITE_POOL_SIZE',    100)

# Diversity enforcement
STALE_RESET_GENS      = getattr(cfg, 'GEL_STALE_RESET_GENS',     2)
MAX_SEED_DUPLICATES   = getattr(cfg, 'GEL_MAX_SEED_DUPLICATES',  2)
DIVERSITY_FRACTION    = getattr(cfg, 'GEL_DIVERSITY_FRACTION',   0.50)
FEATURE_PRIOR_DECAY   = getattr(cfg, 'GEL_FEATURE_PRIOR_DECAY',  0.40)
FEATURE_LOSS_PENALTY  = getattr(cfg, 'GEL_FEATURE_LOSS_PENALTY', 0.50)
WIN_DECAY_PER_GEN     = getattr(cfg, 'GEL_WIN_DECAY_PER_GEN',    0.70)
PRIOR_MAX_CONCENTRATION = getattr(cfg, 'GEL_PRIOR_MAX_CONCENTRATION', 1.8)
MAX_WINNER_SEED_FRACTION = getattr(cfg, 'GEL_MAX_WINNER_SEED_FRACTION', 0.40)
SIG_STALE_RESET_GENS  = getattr(cfg, 'GEL_SIG_STALE_RESET_GENS',  2)

# Phase-2 anti-convergence: inject fresh random programs into seed pool
ANTI_CONVERGENCE_FRACTION = getattr(cfg, 'GEL_ANTI_CONVERGENCE_FRACTION', 0.10)

# ── IMPORTS ──────────────────────────────────────────────────────────────────
from src.feature_engineering import (calculate_features,
                                      PASSTHROUGH_FEATURES,
                                      SCALE_FEATURES,
                                      SessionConfig)
from src.target_generator     import generate_tbm_targets
from src.gp_engine            import (train_gp_model,
                                      extract_elite_programs,
                                      hash_formula)
from trade_discovery.Audit_scripts.vectorbt_evaluator   import evaluate_formula_with_vectorbt
from trade_discovery.Audit_scripts.regime_classifier    import classify_regime


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _safe_stat(stats, key: str, default: float = 0.0) -> float:
    """Compatible with pd.Series (vectorbt) and dict."""
    try:
        if isinstance(stats, pd.Series):
            return float(stats.loc[key]) if key in stats.index else default
        return float(stats.get(key, default) or default)
    except (KeyError, TypeError, ValueError):
        return default


def _features_used(formula_str: str) -> tuple:
    return tuple(sorted({
        f for f in PASSTHROUGH_FEATURES + SCALE_FEATURES
        if f in formula_str
    }))


def _count_formula_features(formula_str: str) -> int:
    """Count distinct features used in a formula string."""
    return len(_features_used(formula_str))


def tanh_scale_train_apply_test(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    scale_cols:       list,
    passthrough_cols: list,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Zero-Preserving Tanh Scaling. Fit on train ONLY, apply to both."""
    original_columns  = X_train.columns
    actual_scale_cols = [c for c in scale_cols       if c in X_train.columns]
    actual_pass_cols  = [c for c in passthrough_cols if c in X_train.columns]
    EPS = 1e-8

    parts_train = []
    parts_test = []

    if actual_scale_cols:
        train_view    = X_train[actual_scale_cols]
        scale_factors = np.maximum(np.percentile(np.abs(train_view), 75, axis=0), EPS)
        
        scaled_train_vals = np.tanh(train_view / scale_factors).astype(np.float32)
        scaled_test_vals  = np.tanh(X_test[actual_scale_cols] / scale_factors).astype(np.float32)
        
        parts_train.append(pd.DataFrame(scaled_train_vals, index=X_train.index, columns=actual_scale_cols))
        parts_test.append(pd.DataFrame(scaled_test_vals, index=X_test.index, columns=actual_scale_cols))

    if actual_pass_cols:
        parts_train.append(X_train[actual_pass_cols].astype(np.float32))
        parts_test.append(X_test[actual_pass_cols].astype(np.float32))

    X_train_scaled = pd.concat(parts_train, axis=1) if parts_train else pd.DataFrame(index=X_train.index)
    X_test_scaled  = pd.concat(parts_test, axis=1)  if parts_test  else pd.DataFrame(index=X_test.index)

    return X_train_scaled[original_columns], X_test_scaled[original_columns]


def setup_directories() -> None:
    for d in ["data", "outputs/gel_stats", "outputs/seed_checkpoints"]:
        os.makedirs(d, exist_ok=True)


# ── DATA LOADING ─────────────────────────────────────────────────────────────

def load_and_prepare_data(filepath: str):
    resolved_path = filepath
    candidates = []
    if filepath:
        candidates.append(filepath)
        if not os.path.isabs(filepath):
            candidates.extend([
                os.path.join(PROJECT_ROOT, filepath),
                os.path.join(TRADE_DISCOVERY_ROOT, filepath),
                os.path.join(WORKSPACE_ROOT, filepath),
            ])
    for candidate in candidates:
        if os.path.exists(candidate):
            resolved_path = candidate
            break
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Data file not found at any candidate path: {candidates}")
    logger.info("Loading raw data from %s", resolved_path)
    df_raw = pd.read_csv(resolved_path)
    df_raw.columns = [col.lower() for col in df_raw.columns]

    if 'datetime' in df_raw.columns:
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        df_raw.set_index('datetime', inplace=True)
    elif 'date' in df_raw.columns:
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw.set_index('date', inplace=True)
        df_raw.index.name = 'datetime'
    else:
        raise ValueError(
            f"CSV must have a 'date' or 'datetime' column. Found: {df_raw.columns.tolist()}")

    df_raw.sort_index(inplace=True)

    df_features = calculate_features(
        df_raw,
        add_session_features=True,
        session=SessionConfig(),
        clip_outside_session=True,
        mds_fast_window=cfg.FE_MDS_FAST_WINDOW,
        mds_slow_window=cfg.FE_MDS_SLOW_WINDOW,
        vol_asym_window=cfg.FE_VOL_ASYM_WINDOW,
    )

    nan_counts = df_features.isna().sum()
    logger.info("NaN counts per column:\n%s", nan_counts)

    n_pre = len(df_features)
    df_features = df_features.dropna()
    logger.info("Features built: %d rows -> %d rows after dropna", n_pre, len(df_features))

    df_features, y_targets = generate_tbm_targets(
        df_raw, df_features,
        max_hold=ORACLE_MAX_HOLD,
        tp_mult=TP_ATR_MULT,
        sl_mult=SL_ATR_MULT,
        atr_period=cfg.ATR_PERIOD,
        drop_both_sl=getattr(cfg, "DROP_WHIPSAW", True),
        drop_neutral=getattr(cfg, "DROP_NEUTRAL", False),
    )
    logger.info("Final aligned dataset: %d features, %d targets",
                len(df_features), len(y_targets))

    df_raw = df_raw.loc[df_features.index].astype(np.float32)
    return df_raw, df_features, y_targets


# ── FIXED TEMPORAL SPLIT ─────────────────────────────────────────────────────

def split_train_holdout(
    df_features: pd.DataFrame,
    y_targets:   pd.Series,
    df_raw:      pd.DataFrame,
    fraction:    float = HOLDOUT_FRACTION,
):
    """
    Split the full dataset into a fixed train and holdout window.
    The holdout is the last `fraction` of data by time and NEVER changes.
    This is what makes GEL different from WFO: a formula discovered in
    generation 1 and a formula discovered in generation 50 are both judged
    on exactly the same future data window.

    An embargo of ORACLE_MAX_HOLD bars is removed from the end of the train
    set to prevent look-ahead leakage from TBM forward-looking labels.
    """
    n          = len(df_features)
    split_idx  = int(n * (1 - fraction))
    split_date = df_features.index[split_idx]

    embargo_rows = ORACLE_MAX_HOLD
    if split_idx <= embargo_rows:
        raise ValueError(
            f"Train set too short after embargo ({split_idx} rows, need > {embargo_rows}). "
            "Reduce GEL_HOLDOUT_FRACTION or use more data."
        )
    train_end_idx = split_idx - embargo_rows

    X_train  = df_features.iloc[:train_end_idx]
    y_train  = y_targets.iloc[:train_end_idx]
    X_hold   = df_features.iloc[split_idx:]
    raw_hold = df_raw.iloc[split_idx:]
    y_hold   = y_targets.iloc[split_idx:]   # kept for reference logging only

    logger.info("=" * 60)
    logger.info("FIXED TEMPORAL SPLIT")
    logger.info("  Train  : %s → %s  (%d rows, -embargo=%d bars)",
                X_train.index.min().date(), X_train.index.max().date(),
                len(X_train), embargo_rows)
    logger.info("  Holdout: %s → %s  (%d rows)",
                X_hold.index.min().date(), X_hold.index.max().date(), len(X_hold))
    logger.info("  Split date   : %s", split_date.date())
    logger.info("  Train labels → Long: %d | Short: %d | Neutral: %d",
                int((y_train == 1).sum()), int((y_train == -1).sum()), int((y_train == 0).sum()))
    logger.info("  Holdout labels → Long: %d | Short: %d | Neutral: %d",
                int((y_hold == 1).sum()), int((y_hold == -1).sum()), int((y_hold == 0).sum()))
    logger.info("=" * 60)

    return X_train, y_train, X_hold, raw_hold, split_date


# ── SIGNAL QUALITY GUARD ─────────────────────────────────────────────────────

def _check_signal_quality(train_signals: np.ndarray, gen: int) -> bool:
    """
    Returns True if signal passes quality checks.
    Checks unique ratio, entropy, and one-sidedness.
    """
    unique_ratio = len(np.unique(np.round(train_signals, 5))) / len(train_signals)

    hist, _      = np.histogram(train_signals, bins=20)
    hist_p       = hist / (hist.sum() + 1e-8)
    norm_entropy = -np.sum(hist_p * np.log(hist_p + 1e-8)) / np.log(20)

    entry_thr = np.percentile(train_signals, ENTRY_PCT)
    exit_thr  = np.percentile(train_signals, EXIT_PCT)
    long_cov  = float(np.mean(train_signals >= entry_thr))
    short_cov = float(np.mean(train_signals <= exit_thr))

    is_one_sided = (
        (long_cov > 0.55 and short_cov < 0.03)
        or (short_cov > 0.55 and long_cov < 0.03)
    )
    # P2-fix: the execution layer trades off the 85th/15th percentile bands, so
    # a near-constant (tiny-magnitude) signal generates 0 trades even if its
    # Pearson correlation looks fine. Require real separation between the
    # buy/sell thresholds to reject ~0-magnitude signals that slip past the
    # (weak) unique_ratio check (e.g. mul(vol, osc) → Buy~0.001, Sell~-0.0004).
    threshold_sep = float(entry_thr - exit_thr)
    is_degenerate = (
        unique_ratio < SIGNAL_UNIQUE_FLOOR
        or norm_entropy < 0.25
        or is_one_sided
        or threshold_sep < SIGNAL_SEP_FLOOR
    )

    if is_degenerate:
        logger.warning(
            "[Gen %d] Degenerate signal — unique_ratio=%.4f | norm_entropy=%.4f | "
            "long_cov=%.3f | short_cov=%.3f | one_sided=%s | threshold_sep=%.4f",
            gen, unique_ratio, norm_entropy, long_cov, short_cov, is_one_sided, threshold_sep
        )
        return False

    logger.info(
        "[Gen %d] Signal OK — unique_ratio=%.4f | norm_entropy=%.4f | "
        "long_cov=%.3f | short_cov=%.3f | Buy>%.4f | Sell<%.4f | sep=%.4f",
        gen, unique_ratio, norm_entropy, long_cov, short_cov, entry_thr, exit_thr, threshold_sep
    )
    return True


# ── FORMULA STRUCTURAL GUARDS ─────────────────────────────────────────────────

def _check_formula_structure(formula_str: str, program_len: int, gen: int) -> bool:
    """
    Returns True if formula passes all structural guards:
    - Min distinct features used
    - Max program length (bloat)
    - No trivial cancellation: sub(X,X) or add(X,neg(X))
    - Not boolean-heavy (>60% comparison/logic ops)
    """
    import re

    n_features_used = len(_features_used(formula_str))

    if n_features_used < MIN_FEATURES:
        logger.warning("[Gen %d] Only %d feature(s) — too shallow. Skipping.",
                       gen, n_features_used)
        return False

    if program_len > MAX_PROG_LEN:
        logger.warning("[Gen %d] Program length=%d > MAX=%d — bloat guard. Skipping.",
                       gen, program_len, MAX_PROG_LEN)
        return False

    if re.findall(r'sub\((\w+),\s*\1\)', formula_str) or        re.findall(r'add\((\w+),\s*neg\(\1\)\)', formula_str):
        logger.warning("[Gen %d] Trivial cancellation (sub(X,X)/add(X,neg(X))). Skipping: %s",
                       gen, formula_str[:120])
        return False

    bool_ops  = len(re.findall(r'\b(lt|gt|and|or)\b', formula_str))
    total_ops = len(re.findall(r'\b(lt|gt|and|or|add|sub|mul|div|max|min|abs|neg|if_then)\b', formula_str))
    bool_ratio = bool_ops / (total_ops + 1e-8)
    if bool_ratio > 0.60 and total_ops > 4:
        logger.warning("[Gen %d] Boolean-heavy formula (%.0f%% bool ops) — likely degenerate. Skipping: %s",
                       gen, bool_ratio * 100, formula_str[:120])
        return False

    logger.info("[Gen %d] Formula OK — features=%d | length=%d | bool_ratio=%.2f",
                gen, n_features_used, program_len, bool_ratio)
    return True


# ── WINNER PERSISTENCE ────────────────────────────────────────────────────────

def _save_winners(winners: List[dict]) -> None:
    if not winners:
        logger.warning("No robust strategies found across all generations.")
        return

    log_path = "outputs/winning_formulas_gel.log"
    with open(log_path, "w") as f:
        f.write(f"GEL Run  : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Holdout  : {HOLDOUT_FRACTION:.0%} of data | Generations: {GEL_GENERATIONS}\n")
        f.write("=" * 70 + "\n\n")

        for w in winners:
            f.write(
                f"Gen {w['gen']:>3} | Ret: {w['return_pct']:>8.2f}% "
                f"| Sharpe: {w['sharpe']:>6.2f} | WR: {w['win_rate']:>5.1f}% "
                f"| DD: {w['max_dd']:>5.1f}% | Trades: {w['n_trades']}\n"
            )
            f.write(f"  Hash       : {w['formula_hash']}\n")
            f.write(f"  Thresholds : Buy>{w['buy_threshold']:.4f}  Sell<{w['sell_threshold']:.4f}\n")
            f.write(f"  Logic      : {w['formula']}\n\n")

        all_feats = [feat for w in winners for feat in _features_used(w['formula'])]
        f.write("\n=== Feature Frequency in Winning Formulas ===\n")
        for feat, count in Counter(all_feats).most_common():
            f.write(f"  {count:>3}x  {feat}\n")

    pd.DataFrame(winners).drop(columns=['formula', 'program'], errors='ignore').to_parquet(
        "outputs/winning_formulas_gel.parquet", index=False)

    logger.info("Winners saved → %s + outputs/winning_formulas_gel.parquet", log_path)


# ── MAIN GEL LOOP ─────────────────────────────────────────────────────────────

def gel_loop(df_raw: pd.DataFrame, df_features: pd.DataFrame, y_targets: pd.Series) -> List[dict]:
    """
    Global Evolutionary Loop — outer evolutionary engine.

    Every generation trains on the full historical train set and is judged
    on the SAME fixed holdout. No formula can survive by being good in only
    one regime era. To reach the leaderboard it must generalise to the most
    recent data the system has.
    """
    logger.info("=" * 60)
    logger.info("GEL START | holdout=%.0f%% | generations=%d | pool_size=%d | seeds/gen=%d",
                HOLDOUT_FRACTION * 100, GEL_GENERATIONS, ELITE_POOL_SIZE, SEEDS_PER_GEN)
    logger.info("=" * 60)

    # ── Fixed split — done once ───────────────────────────────────────────────
    X_train, y_train, X_hold, raw_hold, split_date = split_train_holdout(
        df_features, y_targets, df_raw)

    # ── Scaling — fit on train, apply to both ────────────────────────────────
    logger.info("Fitting Tanh scaler on train set (%d rows)...", len(X_train))
    X_train_s, X_hold_s = tanh_scale_train_apply_test(
        X_train, X_hold, SCALE_FEATURES, PASSTHROUGH_FEATURES)
    logger.info("Scaling done. Train: %s | Holdout: %s", X_train_s.shape, X_hold_s.shape)

    # ── Regime — classified once on train window ──────────────────────────────
    train_regime = classify_regime(df_raw.loc[:split_date])
    logger.info("Global train regime: %s  (used for GP hyperparameter selection)", train_regime)

    # ── State ─────────────────────────────────────────────────────────────────
    elite_pool:    List    = []
    winners:       List    = []
    seen_hashes:   set     = set()
    winner_signatures: set = set()   # collapse math-equivalent winners (same OOS backtest)
    feat_win_counts        = Counter()
    feat_loss_counts       = Counter()   # Phase-1.5: negative feature feedback
    gen_meta_rows: List    = []

    # Stale attractor tracking (anti-collapse)
    _stale_count    = 0
    _last_best_hash = None
    # Signature-stale tracking — detects the "equivalent-winner permutation" loop
    # where the GP keeps emitting different formula STRINGS that produce the
    # identical OOS backtest (same 3-feature cluster reshuffled).
    _sig_stale_count = 0
    _last_sig        = None

    # ── Generational loop ─────────────────────────────────────────────────────
    for gen in range(1, GEL_GENERATIONS + 1):
        logger.info("")
        logger.info("━" * 60)
        logger.info("GEN %d / %d | Pool: %d programs | Winners: %d / %d | Stale: %d/%d",
                    gen, GEL_GENERATIONS, len(elite_pool), len(winners),
                    ELITE_POOL_SIZE, _stale_count, STALE_RESET_GENS)
        logger.info("━" * 60)

        # Feature prior (Dirichlet-smoothed from winner/loser history)
        all_feats  = list(X_train_s.columns)
        alpha      = cfg.GP_FEATURE_PRIOR_ALPHA

        # ── Break the win-tally ratchet ───────────────────────────────────────
        # A feature that appears in EVERY winner (e.g. feat_icp) would otherwise
        # accumulate win counts without bound, permanently pinning the prior at
        # its concentration cap and starving all other features. Decaying the
        # win tallies each generation makes old wins fade, so a feature must keep
        # winning to retain prior mass. Steady-state tally ≈ 1/(1-decay), which
        # is bounded (e.g. ≈3.3 at decay=0.7) instead of climbing to ∞.
        for _f in list(feat_win_counts.keys()):
            feat_win_counts[_f] *= WIN_DECAY_PER_GEN
            if feat_win_counts[_f] < 0.1:
                del feat_win_counts[_f]

        # Balanced feedback: wins add, losses subtract (with penalty weight)
        net_scores = np.array(
            [feat_win_counts.get(f, 0) - FEATURE_LOSS_PENALTY * feat_loss_counts.get(f, 0)
             for f in all_feats],
            dtype=float,
        )
        net_scores = np.maximum(net_scores, 0.0)  # floor at zero

        # Exponential decay to prevent positive-feedback lock-in on early winners
        raw_counts = net_scores * FEATURE_PRIOR_DECAY
        feat_proba = (raw_counts + alpha) / (raw_counts + alpha).sum()

        # P3: cap concentration — no single feature may dominate the prior.
        # Without this the prior locks onto an early winner's features (e.g.
        # ret_norm_130d) and the search never explores elsewhere. Clip the
        # max probability to Nx the uniform baseline, then renormalise.
        uniform_p   = 1.0 / len(all_feats)
        max_p       = PRIOR_MAX_CONCENTRATION * uniform_p
        feat_proba  = np.minimum(feat_proba, max_p)
        feat_proba  = feat_proba / feat_proba.sum()

        if raw_counts.max() > 0:
            top_feat = all_feats[int(np.argmax(raw_counts))]
            logger.info("[Gen %d] Feature prior — most rewarded: '%s' (decayed=%.1f)",
                        gen, top_feat, raw_counts.max())
        else:
            logger.info("[Gen %d] Feature prior — uniform (no winners yet, alpha=%.1f)", gen, alpha)

        # ── GP Training ───────────────────────────────────────────────────────
        logger.info("[Gen %d] Training GP on %d rows | seeds: %d | regime: %s",
                    gen, len(X_train_s),
                    min(len(elite_pool), SEEDS_PER_GEN), train_regime)
        try:
            gp = train_gp_model(
                X_train_s, y_train,
                seed_programs = elite_pool[:SEEDS_PER_GEN] if elite_pool else None,
                fold          = gen,
                feature_proba = feat_proba,
                regime        = train_regime,
                df_raw_train  = df_raw.loc[X_train_s.index],
            )
        except Exception as exc:
            logger.error("[Gen %d] GP training crashed: %s", gen, exc, exc_info=True)
            elite_pool = []
            continue

        if gp is None:
            logger.warning("[Gen %d] GP returned None (internal bloat guard). Skipping generation.", gen)
            continue

        formula_str  = str(gp._program)
        program_len  = len(gp._program.program) if hasattr(gp._program, 'program') else 0
        formula_hash = hash_formula(formula_str)

        # ── Structural guards ─────────────────────────────────────────────────
        if not _check_formula_structure(formula_str, program_len, gen):
            logger.info("[Gen %d] Structural guard failed — extracting elite pool anyway.", gen)
            elite_pool = extract_elite_programs(gp, top_n=SEEDS_PER_GEN)
            logger.info("[Gen %d] Elite pool: %d programs → gen %d.", gen, len(elite_pool), gen + 1)
            del gp; gc.collect()
            continue

        # ── Signal quality on train set ───────────────────────────────────────
        logger.info("[Gen %d] Checking signal quality on train set...", gen)
        train_signals = gp.predict(X_train_s.values)
        if not _check_signal_quality(train_signals, gen):
            elite_pool = extract_elite_programs(gp, top_n=SEEDS_PER_GEN)
            logger.info("[Gen %d] Elite pool: %d programs → gen %d.", gen, len(elite_pool), gen + 1)
            del gp; gc.collect()
            continue

        # ── Deduplication ─────────────────────────────────────────────────────
        if formula_hash in seen_hashes:
            logger.info("[Gen %d] Duplicate formula (hash=%s...) — skipping OOS eval. "
                        "Elite pool extracted to diversify next gen.",
                        gen, formula_hash[:16])
            elite_pool = extract_elite_programs(gp, top_n=SEEDS_PER_GEN)
            del gp; gc.collect()
            continue

        # ── Fixed holdout evaluation ──────────────────────────────────────────
        logger.info("[Gen %d] Evaluating on FIXED holdout | %d rows | %s → %s",
                    gen, len(X_hold_s),
                    X_hold_s.index.min().date(), X_hold_s.index.max().date())
        try:
            _, stats, meta = evaluate_formula_with_vectorbt(
                gp, X_hold_s, raw_hold,
                ENTRY_PCT, EXIT_PCT,
                tp_mult=TP_ATR_MULT, sl_mult=SL_ATR_MULT,
                atr_period=cfg.ATR_PERIOD,
            )
        except Exception as exc:
            logger.error("[Gen %d] Holdout evaluation crashed: %s", gen, exc, exc_info=True)
            elite_pool = extract_elite_programs(gp, top_n=SEEDS_PER_GEN)
            del gp; gc.collect()
            continue

        total_ret  = _safe_stat(stats, 'Total Return [%]',  0.0)
        sharpe     = _safe_stat(stats, 'Sharpe Ratio',      0.0)
        max_dd     = _safe_stat(stats, 'Max Drawdown [%]', 100.0)
        win_rate   = _safe_stat(stats, 'Win Rate [%]',      0.0)
        profit_fac = _safe_stat(stats, 'Profit Factor',     0.0)
        n_trades   = meta['n_long'] + meta['n_short']
        coverage   = meta['coverage_pct']

        logger.info(
            "[Gen %d] Holdout → Return: %.2f%% | Sharpe: %.2f | DD: %.2f%% | "
            "WR: %.1f%% | PF: %.2f | Trades: %d (L=%d S=%d) | Coverage: %.1f%%",
            gen, total_ret, sharpe, max_dd, win_rate, profit_fac,
            n_trades, meta['n_long'], meta['n_short'], coverage
        )

        # Save per-gen holdout stats
        try:
            stat_df = (stats.to_frame(name='value') if isinstance(stats, pd.Series)
                       else pd.DataFrame.from_dict(stats, orient='index', columns=['value']))
            stat_df.to_csv(f"outputs/gel_stats/gen_{gen:03d}_holdout.csv")
        except Exception as e:
            logger.warning("[Gen %d] Could not save per-gen CSV: %s", gen, e)

        # Accumulate metadata
        gen_meta_rows.append({
            'gen': gen, 'return_pct': total_ret, 'sharpe': sharpe,
            'max_dd': max_dd, 'win_rate': win_rate, 'n_trades': n_trades,
            'coverage_pct': coverage, 'prog_length': program_len,
            'n_features': len(_features_used(formula_str)),
            'formula_hash': formula_hash[:16], 'winner': False,
        })

        # ── OOS-signature dedup ───────────────────────────────────────────────
        # Distinct formula STRINGS that produce the identical OOS backtest are
        # math-equivalent (e.g. reshuffled {feat_icp, talib_adx, talib_adxr}).
        # The string-hash dedup above cannot catch these, so they inflate the
        # leaderboard with the same edge counted many times. Collapse them by a
        # rounded performance signature, and use repeats to drive a stale-break.
        oos_sig = (round(total_ret, 2), round(sharpe, 2),
                   round(profit_fac, 2), n_trades)

        if oos_sig == _last_sig:
            _sig_stale_count += 1
        else:
            _sig_stale_count = 0
            _last_sig = oos_sig

        if oos_sig in winner_signatures:
            logger.info(
                "[Gen %d] Duplicate OOS signature %s — math-equivalent to an "
                "existing winner. Skipping leaderboard add (sig-stale %d/%d).",
                gen, oos_sig, _sig_stale_count, SIG_STALE_RESET_GENS
            )
            if _sig_stale_count >= SIG_STALE_RESET_GENS:
                logger.warning(
                    "[Gen %d] SIGNATURE-STALE — nuking elite pool + decaying "
                    "prior to force exploration off the equivalent-winner cluster.",
                    gen
                )
                elite_pool = []
                _sig_stale_count = 0
                _last_sig = None
                for _k in list(feat_win_counts.keys()):
                    feat_win_counts[_k] *= 0.3
                    if feat_win_counts[_k] < 0.1:
                        del feat_win_counts[_k]
            else:
                elite_pool = extract_elite_programs(
                    gp, top_n=SEEDS_PER_GEN,
                    max_duplicates=MAX_SEED_DUPLICATES,
                    diversity_fraction=DIVERSITY_FRACTION,
                )
            del gp; gc.collect()
            continue

        # ── Survivor gate ─────────────────────────────────────────────────────
        oos_evaluated = False
        if n_trades < MIN_HOLDOUT_TRADES:
            logger.info("[Gen %d] FAIL — trades=%d < min=%d",
                        gen, n_trades, MIN_HOLDOUT_TRADES)
            oos_evaluated = True
        elif total_ret <= OOS_MIN_RETURN:
            logger.info("[Gen %d] FAIL — return=%.2f%% ≤ floor=%.2f%%",
                        gen, total_ret, OOS_MIN_RETURN)
            oos_evaluated = True
        elif sharpe <= OOS_MIN_SHARPE:
            logger.info("[Gen %d] FAIL — Sharpe=%.2f ≤ floor=%.2f",
                        gen, sharpe, OOS_MIN_SHARPE)
            oos_evaluated = True
        elif max_dd >= cfg.OOS_MAX_DRAWDOWN:
            logger.info("[Gen %d] FAIL — DD=%.2f%% ≥ max=%.2f%%",
                        gen, max_dd, cfg.OOS_MAX_DRAWDOWN)
            oos_evaluated = True
        elif profit_fac < OOS_MIN_PROFIT_FACTOR:
            logger.info("[Gen %d] FAIL — PF=%.2f < min=%.2f",
                        gen, profit_fac, OOS_MIN_PROFIT_FACTOR)
            oos_evaluated = True
        elif coverage < OOS_MIN_COVERAGE_PCT:
            logger.info("[Gen %d] FAIL — Coverage=%.1f%% < min=%.1f%%",
                        gen, coverage, OOS_MIN_COVERAGE_PCT)
            oos_evaluated = True
        else:
            # ✓ Passed all gates
            seen_hashes.add(formula_hash)
            winner_signatures.add(oos_sig)   # collapse future math-equivalent copies

            buy_thresh  = float(np.percentile(train_signals, ENTRY_PCT))
            sell_thresh = float(np.percentile(train_signals, EXIT_PCT))

            winners.append({
                'gen': gen, 'formula': formula_str, 'formula_hash': formula_hash,
                'return_pct': total_ret, 'sharpe': sharpe, 'max_dd': max_dd,
                'win_rate': win_rate, 'profit_factor': profit_fac, 'n_trades': n_trades,
                'buy_threshold': buy_thresh, 'sell_threshold': sell_thresh,
                'coverage_pct': coverage,
                'program': copy.deepcopy(gp._program),   # P1: retain for OOS-seeded evolution
            })
            gen_meta_rows[-1]['winner'] = True

            # Phase-1.5b — reset negative feature feedback on success. The
            # loss penalty (GEL_FEATURE_LOSS_PENALTY=0.5, unbounded) would
            # otherwise accumulate ~40-50 tallies from prior failing gens and
            # cancel a winner's feature boost (net = wins - 0.5*losses < 0 →
            # prior floored to uniform), silently undoing P3. Clearing here
            # lets the validated lineage persist in the prior.
            feat_loss_counts.clear()

            for feat in _features_used(formula_str):
                feat_win_counts[feat] += 1

            # Sort by Sharpe, trim to ELITE_POOL_SIZE
            winners.sort(key=lambda w: w['sharpe'], reverse=True)
            if len(winners) > ELITE_POOL_SIZE:
                evicted = len(winners) - ELITE_POOL_SIZE
                winners = winners[:ELITE_POOL_SIZE]
                logger.info("[Gen %d] Leaderboard trimmed — evicted %d lower-Sharpe formulas.",
                            gen, evicted)

            logger.info(
                "[Gen %d] ✓ WINNER | Return=%.2f%% | Sharpe=%.2f | WR:%.1f%% | "
                "DD:%.2f%% | PF:%.2f | Leaderboard: %d/%d",
                gen, total_ret, sharpe, win_rate, max_dd, profit_fac,
                len(winners), ELITE_POOL_SIZE
            )

            # Print top-5 leaderboard after each new winner
            logger.info("[Gen %d] Leaderboard (top 5 by Sharpe):", gen)
            for rank, w in enumerate(winners[:5], 1):
                logger.info("  #%d Gen%d | Sharpe=%.2f | Ret=%.2f%% | DD=%.2f%%",
                            rank, w['gen'], w['sharpe'], w['return_pct'], w['max_dd'])

        # ── Negative feedback: learn from OOS failures ─────────────────────────
        # Phase-1.5 — every OOS-evaluated formula that did NOT become a winner
        # contributes a "loss" signal for each of its features.  This prevents
        # the feature prior from only rewarding winners; it also actively
        # deprioritises features that keep appearing in losing formulas.
        if oos_evaluated:
            loser_feats = _features_used(formula_str)
            for feat in loser_feats:
                feat_loss_counts[feat] += 1
            logger.info(
                "[Gen %d] Loss feedback — %d features penalised for OOS failure",
                gen, len(loser_feats)
            )

        # ── Build next-gen seed pool ─────────────────────────────────────────
        # P1: The holdout must drive evolution. Once we have OOS survivors
        # (winners), seed the next generation PRIMARILY from them, padding the
        # remaining slots with in-sample elite for diversity. Before any winner
        # exists we fall back to the pure in-sample elite (bootstrap behaviour).
        # This replaces the old design where the seed pool was always the
        # in-sample-fittest programs, which optimised train fitness and
        # monotonically overfit (OOB 0.23→0.33 while holdout collapsed).
        if winners:
            winner_progs = [copy.deepcopy(w['program']) for w in winners]
            # Seed from OOS-proven winners so the search REFINES a validated
            # lineage — but CAP their share of the pool. Flooding the pool with
            # winners (the old SEEDS_PER_GEN-20 = 90% share) made every seed a
            # permutation of the same 3-feature cluster, so the GP re-derived
            # math-equivalent formulas forever (identical OOB fitness, identical
            # OOS backtest). Reserving the majority for diverse in-sample elite
            # + the 300 anti-conv randoms keeps genuine exploration alive.
            max_winner_seeds = max(1, int(SEEDS_PER_GEN * MAX_WINNER_SEED_FRACTION))
            n_winner = min(len(winner_progs), max_winner_seeds)
            in_sample = extract_elite_programs(
                gp, top_n=SEEDS_PER_GEN - n_winner,
                max_duplicates=MAX_SEED_DUPLICATES,
                diversity_fraction=DIVERSITY_FRACTION,
            )
            elite_pool = winner_progs[:n_winner] + in_sample
            elite_pool = elite_pool[:SEEDS_PER_GEN]
            logger.info("[Gen %d] Elite pool: %d programs (winners=%d + elite=%d) → gen %d.",
                        gen, len(elite_pool), n_winner, len(in_sample), gen + 1)
        else:
            elite_pool = extract_elite_programs(
                gp, top_n=SEEDS_PER_GEN,
                max_duplicates=MAX_SEED_DUPLICATES,
                diversity_fraction=DIVERSITY_FRACTION,
            )
            logger.info("[Gen %d] Elite pool: %d programs → seeding gen %d.",
                        gen, len(elite_pool), gen + 1)

        # ── Stale attractor detection ───────────────────────────────────────
        try:
            _current_best_hash = hash_formula(formula_str)
        except Exception:
            _current_best_hash = None

        if _current_best_hash and _current_best_hash == _last_best_hash:
            _stale_count += 1
        else:
            _stale_count = 0
            _last_best_hash = _current_best_hash

        if _stale_count >= STALE_RESET_GENS:
            logger.warning(
                "█" * 60 + "\n"
                "[Gen %d] STALE ATTRACTOR DETECTED (%d gens unchanged)\n"
                "  Formula: %s\n"
                "  ACTION : Nuking elite pool → cold restart next gen.\n" +
                "█" * 60,
                gen, _stale_count, formula_str[:120]
            )
            elite_pool = []
            _stale_count = 0
            _last_best_hash = None
            # Also decay feature prior harder to break the attractor
            for k in list(feat_win_counts.keys()):
                feat_win_counts[k] *= 0.3
                if feat_win_counts[k] < 0.1:
                    del feat_win_counts[k]
            for k in list(feat_loss_counts.keys()):
                feat_loss_counts[k] = int(feat_loss_counts[k] * 0.3)
                if feat_loss_counts[k] <= 0:
                    del feat_loss_counts[k]

        del gp
        gc.collect()

    # ── Post-run ──────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("GEL COMPLETE — %d winning formulas | %d generations ran.",
                len(winners), GEL_GENERATIONS)
    logger.info("=" * 60)

    if gen_meta_rows:
        meta_df = pd.DataFrame(gen_meta_rows)
        meta_df.to_parquet("outputs/gel_generation_metadata.parquet", index=False)
        meta_df.to_csv("outputs/gel_generation_metadata.csv", index=False)
        logger.info("Generation metadata → outputs/gel_generation_metadata.parquet / .csv")

        n_win      = int(meta_df['winner'].sum())
        avg_sharpe = meta_df['sharpe'].mean()
        best       = meta_df.loc[meta_df['sharpe'].idxmax()]
        logger.info("Run summary: winners=%d | avg_holdout_sharpe=%.2f | "
                    "best_gen=%d (sharpe=%.2f return=%.2f%%)",
                    n_win, avg_sharpe, int(best['gen']), best['sharpe'], best['return_pct'])

    if winners:
        try:
            from trade_discovery.project_secondary.equity_stitcher import stitch_equity_curves
            equity_df = stitch_equity_curves(winners, output_dir="outputs")
            if equity_df is not None:
                logger.info("Combined equity curve → outputs/combined_equity.parquet")
        except Exception as e:
            logger.warning("Equity stitching failed (non-fatal): %s", e)

    _save_winners(winners)
    return winners


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_directories()
    if not os.path.exists(DATAPATH):
        raise FileNotFoundError(f"Data not found at '{DATAPATH}'. "
                                f"Expected: {os.path.abspath(DATAPATH)}")
    try:
        df_raw, df_features, y_targets = load_and_prepare_data(DATAPATH)
        gel_loop(df_raw, df_features, y_targets)
    except Exception as exc:
        logger.critical("Pipeline crashed: %s", exc, exc_info=True)
        raise