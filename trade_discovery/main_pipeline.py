"""
main_pipeline.py — Walk-Forward Optimisation Orchestrator

Production fixes (2026-03-19):
  1. Imports via PEP-8 shim names — no more importlib numeric hacks.
  2. _safe_stat() — compatible with pd.Series and dict stats output.
  3. SIGNAL_UNIQUE_FLOOR = 0.005 from config (was hard-coded 0.01).
  4. All formula guards driven from config constants.
  5. Feature frequency counter counts individual features, not combos.
  6. Parquet + CSV dual persistence for winning_formulas.
  7. Per-fold metadata DataFrame saved for post-hoc analysis.
"""
import gc
import logging
import os
from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── CONFIG ──────────────────────────────────────────────────────────────────
import src.config as cfg

ORACLE_MAX_HOLD     = cfg.ORACLE_MAX_HOLD
TP_ATR_MULT         = cfg.TP_ATR_MULT
SL_ATR_MULT         = cfg.SL_ATR_MULT
ABSOLUTE_EDGE_FLOOR = cfg.ABSOLUTE_EDGE_FLOOR
TRAIN_MONTHS        = cfg.TRAIN_MONTHS
TEST_MONTHS         = cfg.TEST_MONTHS
WFO_STEP_MONTHS     = cfg.WFO_STEP_MONTHS
DATAPATH            = cfg.DATAPATH
ENTRY_PCT           = cfg.ENTRY_PCT
EXIT_PCT            = cfg.EXIT_PCT

MIN_FEATURES        = cfg.MIN_FEATURES_IN_FORMULA
MAX_PROG_LEN        = cfg.MAX_PROGRAM_LENGTH
SIGNAL_STD_FLOOR    = cfg.SIGNAL_STD_FLOOR
SIGNAL_UNIQUE_FLOOR = cfg.SIGNAL_UNIQUE_FLOOR   # FIX 3: 0.005

OOS_MIN_RETURN   = cfg.OOS_MIN_RETURN
OOS_MIN_SHARPE   = cfg.OOS_MIN_SHARPE
OOS_MAX_DRAWDOWN = cfg.OOS_MAX_DRAWDOWN

# ── FIX 1: Clean imports via shim modules ───────────────────────────────────
from src.feature_engineering import (calculate_features,
                                      PASSTHROUGH_FEATURES,
                                      SCALE_FEATURES,
                                      SessionConfig)
from src.target_generator     import generate_tbm_targets
from src.gp_engine            import (train_gp_model,
                                      extract_elite_programs,
                                      hash_formula)
from src.vectorbt_evaluator   import evaluate_formula_with_vectorbt


# ── FIX 2: stats helper ─────────────────────────────────────────────────────
def _safe_stat(stats, key: str, default: float = 0.0) -> float:
    """Works on both pd.Series (vectorbt) and dict."""
    try:
        if isinstance(stats, pd.Series):
            return float(stats.loc[key]) if key in stats.index else default
        return float(stats.get(key, default) or default)
    except (KeyError, TypeError, ValueError):
        return default


def extract_features_used(formula_str: str) -> tuple:
    return tuple(sorted({
        f for f in PASSTHROUGH_FEATURES + SCALE_FEATURES
        if f in formula_str
    }))


def setup_directories() -> None:
    for d in ["data", "outputs/vectorbt_stats", "outputs/seed_checkpoints"]:
        os.makedirs(d, exist_ok=True)


def load_and_prepare_data(filepath: str):
    logger.info("Loading raw data from %s", filepath)
    df_raw = pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
    df_raw.sort_index(inplace=True)
    df_raw.columns = [col.lower() for col in df_raw.columns]

    df_features = calculate_features(
        df_raw,
        add_session_features=True,
        session=SessionConfig(open_time="09:15", close_time="15:30", tz="Asia/Kolkata"),
        clip_outside_session=True,
        mds_fast_window=5,
        mds_slow_window=30,
        vol_asym_window=20,
    )
    logger.info("Features: %s", list(df_features.columns))

    df_features, y_targets = generate_tbm_targets(
        df_raw, df_features,
        max_hold=ORACLE_MAX_HOLD,
        tp_mult=TP_ATR_MULT,
        sl_mult=SL_ATR_MULT,
    )
    df_raw = df_raw.loc[df_features.index].astype(np.float32)
    return df_raw, df_features, y_targets


def tanh_scale_train_apply_test(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    scale_cols:       list,
    passthrough_cols: list,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Zero-Preserving Tanh Scaling. Fit on train ONLY."""
    original_columns  = X_train.columns
    actual_scale_cols = [c for c in scale_cols       if c in X_train.columns]
    actual_pass_cols  = [c for c in passthrough_cols if c in X_train.columns]
    EPS = 1e-8

    X_train_scaled = pd.DataFrame(index=X_train.index)
    X_test_scaled  = pd.DataFrame(index=X_test.index)

    if actual_scale_cols:
        train_view    = X_train[actual_scale_cols]
        scale_factors = np.maximum(np.percentile(np.abs(train_view), 75, axis=0), EPS)
        X_train_scaled[actual_scale_cols] = np.tanh(train_view / scale_factors).astype(np.float32)
        X_test_scaled[actual_scale_cols]  = np.tanh(X_test[actual_scale_cols] / scale_factors).astype(np.float32)

    if actual_pass_cols:
        X_train_scaled[actual_pass_cols] = X_train[actual_pass_cols].astype(np.float32)
        X_test_scaled[actual_pass_cols]  = X_test[actual_pass_cols].astype(np.float32)

    return X_train_scaled[original_columns], X_test_scaled[original_columns]


# ── WALK-FORWARD OPTIMISATION ───────────────────────────────────────────────

def walk_forward_optimization(
    df_raw, df_features, y_targets,
    train_months=TRAIN_MONTHS,
    test_months=TEST_MONTHS,
    step_months=WFO_STEP_MONTHS,
    data_path="Unknown",
):
    logger.info("=" * 60)
    logger.info("WFO START | train=%dm | test=%dm | step=%dm",
                train_months, test_months, step_months)
    logger.info("=" * 60)

    start_date = df_features.index.min()
    end_date   = df_features.index.max()
    current_train_start = start_date
    fold               = 1
    winning_formulas   = []
    fold_meta_rows     = []       # FIX 7: per-fold Parquet accumulator
    seen_feature_combos = set()
    seen_hashes         = set()
    seed_programs       = None

    while True:
        train_end = current_train_start + pd.DateOffset(months=train_months)
        test_end  = train_end           + pd.DateOffset(months=test_months)

        if test_end > end_date:
            logger.info("End of dataset reached. WFO complete after %d folds.", fold - 1)
            break

        logger.info("--- FOLD %d | Train: %s→%s | Test: %s→%s ---",
                    fold,
                    current_train_start.date(), train_end.date(),
                    train_end.date(), test_end.date())

        train_end_incl = train_end - pd.Timedelta(nanoseconds=1)
        X_train_raw = df_features.loc[current_train_start : train_end_incl]
        y_train_raw = y_targets  .loc[current_train_start : train_end_incl]
        X_test      = df_features.loc[train_end : test_end]
        raw_test    = df_raw     .loc[train_end : test_end]

        # Embargo purge
        if len(X_train_raw) > ORACLE_MAX_HOLD:
            X_train = X_train_raw.iloc[:-ORACLE_MAX_HOLD]
            y_train = y_train_raw.iloc[:-ORACLE_MAX_HOLD]
        else:
            logger.warning("Fold %d: too short after purge — skipping.", fold)
            current_train_start += pd.DateOffset(months=step_months); fold += 1; continue

        if len(X_train) < 500 or len(X_test) < 200:
            logger.warning("Fold %d: insufficient data — skipping.", fold)
            current_train_start += pd.DateOffset(months=step_months); fold += 1; continue

        # Rotation fold
        is_rotation_fold = (fold % 5 == 0) and ("feat_ob_dist_supp" in X_train.columns)
        if is_rotation_fold:
            logger.info("[Fold %d] ROTATION — feat_ob_dist_supp excluded.", fold)
            X_train = X_train.drop(columns=["feat_ob_dist_supp"])
            X_test  = X_test.drop(columns=["feat_ob_dist_supp"])

        X_train, X_test = tanh_scale_train_apply_test(
            X_train, X_test,
            scale_cols=SCALE_FEATURES,
            passthrough_cols=PASSTHROUGH_FEATURES,
        )

        # ── GP Training ──
        try:
            gp_model    = train_gp_model(X_train, y_train,
                                          seed_programs=seed_programs, fold=fold)
            formula_str = str(gp_model._program)
            features_used   = extract_features_used(formula_str)
            n_features_used = len(features_used)
            program_len = len(gp_model._program.program) if hasattr(gp_model._program, 'program') else 0

            # Guard 1: shallow formula
            if n_features_used < MIN_FEATURES:
                logger.warning("[Fold %d] Only %d feature(s) — too shallow.", fold, n_features_used)
                seed_programs = None
                current_train_start += pd.DateOffset(months=step_months); fold += 1; continue

            # Guard 2: bloat
            if program_len > MAX_PROG_LEN:
                logger.warning("[Fold %d] Program length=%d — bloat guard.", fold, program_len)
                seed_programs = None
                current_train_start += pd.DateOffset(months=step_months); fold += 1; continue

            train_signals = gp_model.predict(X_train.values)
            entry_pct     = np.percentile(train_signals, ENTRY_PCT)
            exit_pct      = np.percentile(train_signals, EXIT_PCT)

            # Guard 3: degenerate signal  (FIX 3: uses 0.005 from config)
            signal_std   = float(np.std(train_signals))
            unique_ratio = len(np.unique(np.round(train_signals, 3))) / len(train_signals)
            if (signal_std < SIGNAL_STD_FLOOR
                    or unique_ratio < SIGNAL_UNIQUE_FLOOR
                    or (entry_pct >= 0.95 and exit_pct <= 0.05)):
                logger.warning("[Fold %d] Degenerate signal — std=%.4f, unique_ratio=%.4f.",
                               fold, signal_std, unique_ratio)
                seed_programs = None
                current_train_start += pd.DateOffset(months=step_months); fold += 1; continue

            logger.info("[Fold %d] Thresholds → Buy: %.4f | Sell: %.4f",
                        fold, entry_pct, exit_pct)

        except Exception as exc:
            logger.error("GP training failed on fold %d: %s", fold, exc, exc_info=True)
            seed_programs = None
            current_train_start += pd.DateOffset(months=step_months); fold += 1; continue

        # ── OOS Evaluation ──
        portfolio, stats, metadata = evaluate_formula_with_vectorbt(
            gp_model, X_test, raw_test, ENTRY_PCT, EXIT_PCT,
            tp_mult=TP_ATR_MULT, sl_mult=SL_ATR_MULT,
        )

        # FIX 2: _safe_stat is Series + dict compatible
        total_return  = _safe_stat(stats, 'Total Return [%]',  0.0)
        sharpe        = _safe_stat(stats, 'Sharpe Ratio',      0.0)
        max_dd        = _safe_stat(stats, 'Max Drawdown [%]', 100.0)
        win_rate      = _safe_stat(stats, 'Win Rate [%]',      0.0)
        profit_factor = _safe_stat(stats, 'Profit Factor',     0.0)

        # FIX 7: accumulate metadata regardless of win/loss
        fold_meta_rows.append({
            'fold': fold, 'return_pct': total_return, 'sharpe': sharpe,
            'max_dd': max_dd, 'win_rate': win_rate,
            'n_features': n_features_used, 'prog_length': program_len,
            'coverage_pct': metadata['coverage_pct'], 'winner': False,
        })

        if (total_return > OOS_MIN_RETURN
                and sharpe   > OOS_MIN_SHARPE
                and max_dd   < OOS_MAX_DRAWDOWN):

            feat_combo   = frozenset(extract_features_used(formula_str))
            formula_hash = hash_formula(formula_str)

            if feat_combo in seen_feature_combos:
                logger.warning("[Fold %d] Redundant feature combo — skipping.", fold)
            elif formula_hash in seen_hashes:
                logger.warning("[Fold %d] Duplicate hash — skipping.", fold)
            else:
                seen_feature_combos.add(feat_combo)
                seen_hashes.add(formula_hash)
                logger.info("[Fold %d] ✓ SURVIVOR | Return: %.2f%% | Sharpe: %.2f",
                            fold, total_return, sharpe)
                winning_formulas.append({
                    'fold': fold, 'formula': formula_str,
                    'formula_hash': formula_hash,
                    'return_pct': total_return, 'sharpe': sharpe,
                    'win_rate': win_rate, 'max_dd': max_dd,
                    'profit_factor': profit_factor,
                    'buy_threshold': float(entry_pct),
                    'sell_threshold': float(exit_pct),
                    'n_long': metadata['n_long'],
                    'n_short': metadata['n_short'],
                    'coverage_pct': metadata['coverage_pct'],
                })
                fold_meta_rows[-1]['winner'] = True

                # Per-fold stats CSV
                out = (stats.to_frame(name='value')
                       if isinstance(stats, pd.Series)
                       else pd.DataFrame.from_dict(stats, orient='index', columns=['value']))
                out.to_csv(f"outputs/vectorbt_stats/fold_{fold}_winner.csv")

            if not is_rotation_fold:
                seed_programs = extract_elite_programs(gp_model)
                logger.info("[Fold %d] Extracted %d seeds → fold %d.",
                            fold, len(seed_programs), fold + 1)
            else:
                seed_programs = None
        else:
            logger.info("[Fold %d] ✗ FAILED OOS | Return: %.2f%% | Sharpe: %.2f",
                        fold, total_return, sharpe)
            seed_programs = None

        current_train_start += pd.DateOffset(months=step_months)
        fold += 1

        try:
            del X_train, y_train, X_test, raw_test, gp_model, portfolio
        except NameError:
            pass
        gc.collect()

    # ── Final Report ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RUN COMPLETE — %d unique winning formulas.", len(winning_formulas))
    logger.info("=" * 60)

    # FIX 7: persist fold metadata
    if fold_meta_rows:
        pd.DataFrame(fold_meta_rows).to_parquet(
            "outputs/fold_metadata.parquet", index=False)
        logger.info("Fold metadata → outputs/fold_metadata.parquet")

    if winning_formulas:
        log_file = "outputs/winning_formulas.log"
        with open(log_file, "a") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Run    : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data   : {data_path}\n")
            f.write(f"WFO    : train={train_months}m | test={test_months}m | step={step_months}m\n")
            f.write("-" * 60 + "\n")
            for w in winning_formulas:
                f.write(
                    f"Fold {w['fold']:>3} | Ret: {w['return_pct']:>8.2f}% "
                    f"| Sharpe: {w['sharpe']:>5.2f} | WR: {w['win_rate']:>5.1f}% "
                    f"| DD: {w['max_dd']:>5.1f}% | PF: {w['profit_factor']:>5.2f}\n"
                )
                f.write(f"Hash       : {w['formula_hash']}\n")
                f.write(f"Thresholds : Buy>{w['buy_threshold']:.4f}  Sell<{w['sell_threshold']:.4f}\n")
                f.write(f"Logic      : {w['formula']}\n\n")

            # FIX 5: individual feature frequency (not combo frequency)
            all_feats = [
                feat
                for w in winning_formulas
                for feat in extract_features_used(w['formula'])
            ]
            f.write("\n=== Individual Feature Frequency in Winning Formulas ===\n")
            for feat, count in Counter(all_feats).most_common():
                f.write(f"  {count:>3}x  {feat}\n")

        # FIX 6: machine-readable Parquet (formula string excluded — can be huge)
        wf_df = pd.DataFrame(winning_formulas).drop(columns=['formula'])
        wf_df.to_parquet("outputs/winning_formulas.parquet", index=False)
        logger.info("Winners → %s + outputs/winning_formulas.parquet", log_file)
    else:
        logger.warning("No robust strategies found.")


if __name__ == "__main__":
    setup_directories()
    if not os.path.exists(DATAPATH):
        raise FileNotFoundError(f"Data not found at '{DATAPATH}'.")
    try:
        df_raw, df_features, y_targets = load_and_prepare_data(DATAPATH)
        walk_forward_optimization(
            df_raw, df_features, y_targets,
            train_months=TRAIN_MONTHS, test_months=TEST_MONTHS,
            step_months=WFO_STEP_MONTHS, data_path=DATAPATH,
        )
    except Exception as exc:
        logger.critical("Pipeline crashed: %s", exc, exc_info=True)
