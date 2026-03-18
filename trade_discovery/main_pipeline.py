import gc
import importlib
import logging
import os
from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GLOBAL CONFIG (Imported from src.config)
# ---------------------------------------------------------------------------
cfg = importlib.import_module("src.config")
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

# ---------------------------------------------------------------------------
# MODULE IMPORTS
# ---------------------------------------------------------------------------
fe      = importlib.import_module("src.01_feature_engineering")
tg      = importlib.import_module("src.02_target_generator")
gp      = importlib.import_module("src.03_gp_engine")
vbteval = importlib.import_module("src.04_vectorbt_evaluator")

calculate_features             = fe.calculate_features
generate_tbm_targets           = tg.generate_tbm_targets
train_gp_model                 = gp.train_gp_model
extract_elite_programs         = gp.extract_elite_programs
hash_formula                   = gp.hash_formula             # NEW
evaluate_formula_with_vectorbt = vbteval.evaluate_formula_with_vectorbt

PASSTHROUGH_FEATURES = fe.PASSTHROUGH_FEATURES
SCALE_FEATURES       = fe.SCALE_FEATURES


def extract_features_used(formula_str):
    """Extract which features from our defined pools appear in the formula."""
    return tuple(sorted(set(
        f for f in PASSTHROUGH_FEATURES + SCALE_FEATURES
        if f in formula_str
    )))

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def setup_directories():
    os.makedirs("data",                     exist_ok=True)
    os.makedirs("outputs/vectorbt_stats",   exist_ok=True)
    os.makedirs("outputs/seed_checkpoints", exist_ok=True)


def load_and_prepare_data(filepath: str):
    logger.info("Loading raw data from %s", filepath)
    df_raw = pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
    df_raw.sort_index(inplace=True)
    df_raw.columns = [col.lower() for col in df_raw.columns]

    feature_kwargs = dict(
        add_session_features = True,
        session = fe.SessionConfig(
            open_time="09:15", close_time="15:30", tz="Asia/Kolkata"
        ),
        clip_outside_session = True,
        mds_fast_window      = 5,
        mds_slow_window      = 30,
        vol_asym_window      = 20,
    )

    df_features = calculate_features(df_raw, **feature_kwargs)
    logger.info("Features computed. Columns: %s", list(df_features.columns))

    df_features, y_targets = generate_tbm_targets(
        df_raw, df_features,
        max_hold = ORACLE_MAX_HOLD,
        tp_mult  = TP_ATR_MULT,
        sl_mult  = SL_ATR_MULT
    )

    df_raw = df_raw.loc[df_features.index].astype(np.float32)
    return df_raw, df_features, y_targets


def tanh_scale_train_apply_test(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    scale_cols:       list,
    passthrough_cols: list
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zero-Preserving Tanh Scaling.
    Fit on train ONLY — strict lookahead-bias prevention.
    EPS guard for flat-market zero-division.
    """
    original_columns  = X_train.columns
    actual_scale_cols = [c for c in scale_cols       if c in X_train.columns]
    actual_pass_cols  = [c for c in passthrough_cols if c in X_train.columns]
    EPS = 1e-8

    X_train_scaled = pd.DataFrame(index=X_train.index)
    X_test_scaled  = pd.DataFrame(index=X_test.index)

    if actual_scale_cols:
        train_view    = X_train[actual_scale_cols]
        scale_factors = np.percentile(np.abs(train_view), 75, axis=0)
        scale_factors = np.maximum(scale_factors, EPS)
        X_train_scaled[actual_scale_cols] = np.tanh(train_view / scale_factors).astype(np.float32)
        X_test_scaled[actual_scale_cols]  = np.tanh(X_test[actual_scale_cols] / scale_factors).astype(np.float32)

    if actual_pass_cols:
        X_train_scaled[actual_pass_cols] = X_train[actual_pass_cols].astype(np.float32)
        X_test_scaled[actual_pass_cols]  = X_test[actual_pass_cols].astype(np.float32)

    return X_train_scaled[original_columns], X_test_scaled[original_columns]


# ---------------------------------------------------------------------------
# WALK-FORWARD OPTIMISATION
# ---------------------------------------------------------------------------

def walk_forward_optimization(
    df_raw,
    df_features,
    y_targets,
    train_months: int = TRAIN_MONTHS,
    test_months:  int = TEST_MONTHS,
    step_months:  int = WFO_STEP_MONTHS,      # FIX: advance from config
    data_path:    str = "Unknown"
):
    """
    Walk-Forward Optimisation with cross-fold formula evolution.

    Fixes vs v1
    -----------
    1. step_months decoupled from test_months — dense rolling folds.
    2. seed_programs carries elite _Program objects from winning folds forward.
    3. seed_programs reset to None when a fold fails — bad genes not propagated.
    4. Formula SHA-256 deduplication — identical formulas logged only once.
    5. random_state=fold in GP engine — each fold explores different space.
    6. MUTATION_BOOST on seeded individuals — novelty pressure, no gene lock-in.
    """
    logger.info("=" * 60)
    logger.info("WFO START | train=%dm | test=%dm | step=%dm",
                train_months, test_months, step_months)
    logger.info("=" * 60)

    start_date = df_features.index.min()
    end_date   = df_features.index.max()

    current_train_start = start_date
    fold             = 1
    winning_formulas = []
    seen_feature_combos = set()    # Layer 2: frozenset(features_used)
    seen_hashes         = set()    # SHA-256 dedup
    seed_programs       = None     # FIX: cross-fold elite seed carrier

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

        # Embargo purge — prevent label leakage at train/test boundary
        if len(X_train_raw) > ORACLE_MAX_HOLD:
            X_train = X_train_raw.iloc[:-ORACLE_MAX_HOLD]
            y_train = y_train_raw.iloc[:-ORACLE_MAX_HOLD]
        else:
            logger.warning("Fold %d: train too short after purge — skipping.", fold)
            current_train_start += pd.DateOffset(months=step_months)
            fold += 1
            continue

        if len(X_train) < 500 or len(X_test) < 200:
            logger.warning("Fold %d: insufficient data (train=%d, test=%d) — skipping.",
                           fold, len(X_train), len(X_test))
            current_train_start += pd.DateOffset(months=step_months)
            fold += 1
            continue

        # Layer 3: Feature Rotation — force exploration by excluding dominant features
        if fold % 5 == 0 and "feat_ob_dist_supp" in X_train.columns:
            logger.info("[Fold %d] 🔄 ROTATION fold — seeds will NOT be carried forward.", fold)
            X_train = X_train.drop(columns=["feat_ob_dist_supp"])
            X_test  = X_test.drop(columns=["feat_ob_dist_supp"])
            is_rotation_fold = True
        else:
            is_rotation_fold = False

        X_train, X_test = tanh_scale_train_apply_test(
            X_train, X_test,
            scale_cols       = SCALE_FEATURES,
            passthrough_cols = PASSTHROUGH_FEATURES
        )

        try:
            # FIX 1-3: pass seed_programs + fold number
            gp_model    = train_gp_model(X_train, y_train, seed_programs=seed_programs, fold=fold)
            formula_str = str(gp_model._program)

            # Layer 1: Minimum Feature Diversity Guard
            features_used = extract_features_used(formula_str)
            n_features_used = len(features_used)
            if n_features_used < 3:
                logger.warning("[Fold %d] Formula uses only %d feature(s) — too shallow. Skipping.", fold, n_features_used)
                seed_programs = None
                current_train_start += pd.DateOffset(months=step_months)
                fold += 1
                continue

            # ── Degenerate formula guard ──
            program_len = len(gp_model._program.program) if hasattr(gp_model._program, 'program') else 0
            if program_len > 80:
                logger.warning("[Fold %d] Formula too complex (length=%d) — likely bloat. Skipping OOS.", fold, program_len)
                seed_programs = None
                current_train_start += pd.DateOffset(months=step_months)
                fold += 1
                continue

            train_signals = gp_model.predict(X_train.values)
            entry_pct     = np.percentile(train_signals, ENTRY_PCT)
            exit_pct      = np.percentile(train_signals, EXIT_PCT)

            # ── Degenerate signal distribution guard ──
            signal_std   = float(np.std(train_signals))
            unique_ratio = len(np.unique(np.round(train_signals, 3))) / len(train_signals)

            if signal_std < 0.05 or unique_ratio < 0.01 or (entry_pct >= 0.95 and exit_pct <= 0.05):
                logger.warning("[Fold %d] Degenerate signal distribution — std=%.4f, unique_ratio=%.4f. Skipping.",
                               fold, signal_std, unique_ratio)
                seed_programs = None
                current_train_start += pd.DateOffset(months=step_months)
                fold += 1
                continue

            logger.info("[Fold %d] Thresholds → Buy: %.4f | Sell: %.4f",
                        fold, entry_pct, exit_pct)

        except Exception as exc:
            logger.error("GP training failed on fold %d: %s", fold, exc, exc_info=True)
            seed_programs = None   # never carry corrupted state
            current_train_start += pd.DateOffset(months=step_months)
            fold += 1
            continue

        portfolio, stats, metadata = evaluate_formula_with_vectorbt(
            gp_model, X_test, raw_test, ENTRY_PCT, EXIT_PCT,
            tp_mult = TP_ATR_MULT,
            sl_mult = SL_ATR_MULT
        )

        total_return = float(stats.get('Total Return [%]', 0) or 0)
        sharpe       = float(stats.get('Sharpe Ratio',     0) or 0)
        max_dd       = float(stats.get('Max Drawdown [%]', 100) or 100)

        if total_return > 2.0 and sharpe > 1.5 and max_dd < 15.0:
            # Bug 3: Two-layer dedup — semantic first, then hash
            feat_combo   = frozenset(extract_features_used(formula_str))
            formula_hash = hash_formula(formula_str)

            if feat_combo in seen_feature_combos:
                logger.warning(
                    "[Fold %d] Semantically redundant formula (same feature combo %s) — skipping log entry. Formula: %s",
                    fold, sorted(feat_combo), formula_str
                )
            elif formula_hash in seen_hashes:
                logger.warning("[Fold %d] Identical formula (hash match) — skipping log entry.", fold)
            else:
                seen_feature_combos.add(feat_combo)
                seen_hashes.add(formula_hash)
                logger.info("[Fold %d] ✓ SURVIVOR | Return: %.2f%% | Sharpe: %.2f",
                            fold, total_return, sharpe)
                winning_formulas.append({
                    'fold':          fold,
                    'formula':       formula_str,
                    'formula_hash':  formula_hash,
                    'return_pct':    total_return,
                    'sharpe':        sharpe,
                    'win_rate':      float(stats.get('Win Rate [%]', 0) or 0),
                    'max_dd':        float(stats.get('Max Drawdown [%]', 0) or 0),
                    'profit_factor': float(stats.get('Profit Factor', 0) or 0),
                    'buy_threshold': float(entry_pct),
                    'sell_threshold':float(exit_pct),
                    'n_long':        metadata['n_long'],
                    'n_short':       metadata['n_short'],
                    'coverage_pct':  metadata['coverage_pct']
                })
                out = stats.to_frame(name='value') if hasattr(stats, 'to_frame') else pd.DataFrame(stats)
                out.to_csv(f"outputs/vectorbt_stats/fold_{fold}_winner.csv")

            # Extract elite seeds for next fold regardless of dedup status
            if not is_rotation_fold:
                seed_programs = extract_elite_programs(gp_model)
                logger.info("[Fold %d] Extracted %d elite seeds → fold %d.",
                            fold, len(seed_programs), fold + 1)
            else:
                seed_programs = None   # rotation fold — don't poison next fold's seeds
                logger.info("[Fold %d] Rotation fold: seed carry-over suppressed.", fold)

        else:
            logger.info("[Fold %d] ✗ FAILED OOS | Return: %.2f%% | Sharpe: %.2f",
                        fold, total_return, sharpe)
            # Reset: do not propagate losing genetic material
            seed_programs = None

        # FIX: advance by step_months, not test_months
        current_train_start += pd.DateOffset(months=step_months)
        fold += 1

        # Memory cleanup - explicit deletion to ensure local scope is cleared
        try:
            del X_train, y_train, X_test, raw_test, gp_model, portfolio
        except NameError:
            pass
        gc.collect()

    # Final report
    logger.info("=" * 60)
    logger.info("RUN COMPLETE — %d unique winning formulas.", len(winning_formulas))
    logger.info("=" * 60)

    if winning_formulas:
        log_file = "outputs/winning_formulas.log"
        with open(log_file, "a") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Run       : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data      : {data_path}\n")
            f.write(f"Params    : TP={TP_ATR_MULT} | SL={SL_ATR_MULT} | MAX_HOLD={ORACLE_MAX_HOLD}\n")
            f.write(f"WFO       : train={train_months}m | test={test_months}m | step={step_months}m\n")
            f.write("-" * 60 + "\n")
            for w in winning_formulas:
                f.write(f"Fold {w['fold']:>3} | Ret: {w['return_pct']:>8.2f}% | Sharpe: {w['sharpe']:>5.2f} | WR: {w['win_rate']:>5.1f}% | DD: {w['max_dd']:>5.1f}% | PF: {w['profit_factor']:>5.2f}\n")
                f.write(f"Hash      : {w['formula_hash']}\n")
                f.write(f"Thresholds: Buy>{w['buy_threshold']:.4f}  Sell<{w['sell_threshold']:.4f}\n")
                f.write(f"Logic     : {w['formula']}\n\n")
            
            # --- CROSS-FOLD PATTERN ANALYSIS ---
            feature_usage = Counter(extract_features_used(w['formula']) for w in winning_formulas)
            
            summary_header = "\n=== Feature Combinations in Winning Formulas ==="
            print(summary_header)
            f.write(summary_header + "\n")
            
            for combo, count in feature_usage.most_common(5):
                line = f"  {count} folds: {combo}"
                print(line)
                f.write(line + "\n")

        logger.info("Winners written to %s", log_file)
    else:
        logger.warning("No robust strategies found. Adjust multipliers or provide more data.")



# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_directories()

    if not os.path.exists(DATAPATH):
        raise FileNotFoundError(
            f"CRITICAL: Data file not found at '{DATAPATH}'. Check filename and directory."
        )

    try:
        df_raw, df_features, y_targets = load_and_prepare_data(DATAPATH)
        walk_forward_optimization(
            df_raw, df_features, y_targets,
            train_months = TRAIN_MONTHS,
            test_months  = TEST_MONTHS,
            step_months  = WFO_STEP_MONTHS,     # dense rolling from config
            data_path    = DATAPATH
        )
    except Exception as exc:
        logger.critical("Pipeline crashed: %s", exc, exc_info=True)
