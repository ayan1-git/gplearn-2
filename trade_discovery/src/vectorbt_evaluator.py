import importlib

import numpy as np
import pandas as pd
import vectorbt as vbt

pd.set_option("future.no_silent_downcasting", True)

try:
    config = importlib.import_module("src.config")
    DEFAULT_FEES             = float(config.FEE_PER_SIDE)
    DEFAULT_SLIPPAGE         = float(config.SLIPPAGE)
    # FIX-VBT-1: Pull TP/SL from config — MUST match target_generator contract
    DEFAULT_TP_MULT          = float(config.TP_ATR_MULT)      # was hardcoded 3.0
    DEFAULT_SL_MULT          = float(config.SL_ATR_MULT)      # was hardcoded 1.5
    DEFAULT_ABSOLUTE_EDGE_FLOOR = float(config.ABSOLUTE_EDGE_FLOOR)
    DEFAULT_RANK_WINDOW      = int(getattr(config, "CAUSAL_RANK_WINDOW", 500))
except (ImportError, AttributeError):
    DEFAULT_FEES             = 0.0003
    DEFAULT_SLIPPAGE         = 0.0001
    # FIX-VBT-1: Fallback values now match config.py defaults exactly
    DEFAULT_TP_MULT          = 4.0    # was 3.0 — corrected to match TP_ATR_MULT
    DEFAULT_SL_MULT          = 1.7    # was 1.5 — corrected to match SL_ATR_MULT
    DEFAULT_ABSOLUTE_EDGE_FLOOR = 0.0010
    DEFAULT_RANK_WINDOW      = 500


def _compute_wilder_atr(df_raw: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df_raw["high"] - df_raw["low"]
    high_close = (df_raw["high"] - df_raw["close"].shift(1)).abs()
    low_close = (df_raw["low"] - df_raw["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    return true_range.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def _validate_inputs(
    df_features_oos: pd.DataFrame,
    df_raw_oos: pd.DataFrame,
    long_pct_level: float,
    short_pct_level: float,
    rolling_window: int,
    absolute_edge_floor: float,
    tp_mult: float,
    sl_mult: float,
) -> None:
    if not 0.0 <= short_pct_level <= 100.0:
        raise ValueError("short_pct_level must be in [0, 100]")
    if not 0.0 <= long_pct_level <= 100.0:
        raise ValueError("long_pct_level must be in [0, 100]")
    if short_pct_level >= long_pct_level:
        raise ValueError("short_pct_level must be < long_pct_level")
    if rolling_window <= 1:
        raise ValueError("rolling_window must be > 1")
    if absolute_edge_floor < 0.0:
        raise ValueError("absolute_edge_floor must be >= 0")
    if tp_mult <= 0.0 or sl_mult <= 0.0:
        raise ValueError("tp_mult and sl_mult must be > 0")

    required_cols = {"open", "high", "low", "close"}
    missing_cols = required_cols - set(df_raw_oos.columns)
    if missing_cols:
        raise ValueError(f"df_raw_oos missing required columns: {sorted(missing_cols)}")

    if len(df_features_oos) == 0 or len(df_raw_oos) == 0:
        raise ValueError("OOS features and raw data must be non-empty")

    if not df_features_oos.index.equals(df_raw_oos.index):
        raise ValueError("df_features_oos and df_raw_oos must share the same index")

    if df_features_oos.isna().any().any():
        raise ValueError("df_features_oos contains NaNs")
    if df_raw_oos[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("df_raw_oos contains NaNs in OHLC columns")


def evaluate_formula_with_vectorbt(
    gp_model,
    df_features_oos: pd.DataFrame,
    df_raw_oos: pd.DataFrame,
    long_pct_level: float,
    short_pct_level: float,
    fees: float = DEFAULT_FEES,
    slippage: float = DEFAULT_SLIPPAGE,
    rolling_window: int = DEFAULT_RANK_WINDOW,
    absolute_edge_floor: float = DEFAULT_ABSOLUTE_EDGE_FLOOR,
    tp_mult: float = DEFAULT_TP_MULT,
    sl_mult: float = DEFAULT_SL_MULT,
    atr_period: int = 20,
):
    """
    Evaluate GP formula out-of-sample using next-open execution with fixed
    ATR-based TP/SL barriers aligned to the target generator contract.
    """
    _validate_inputs(
        df_features_oos=df_features_oos,
        df_raw_oos=df_raw_oos,
        long_pct_level=long_pct_level,
        short_pct_level=short_pct_level,
        rolling_window=rolling_window,
        absolute_edge_floor=absolute_edge_floor,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
    )

    print("Predicting signals on Out-of-Sample data...")
    raw_scores = gp_model.predict(df_features_oos.values)
    scores = pd.Series(raw_scores, index=df_features_oos.index, name="gp_score", dtype=np.float32)

    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Non-finite GP scores detected on OOS data")

    min_periods = min(50, rolling_window)
    rolling_ranks = scores.rolling(window=rolling_window, min_periods=min_periods).rank(pct=True)
    expanding_ranks = scores.expanding(min_periods=1).rank(pct=True)
    causal_ranks = rolling_ranks.fillna(expanding_ranks)

    long_rank_mask = causal_ranks >= (long_pct_level / 100.0)
    short_rank_mask = causal_ranks <= (short_pct_level / 100.0)

    long_edge_mask = scores >= absolute_edge_floor
    short_edge_mask = scores <= -absolute_edge_floor

    long_entries_raw = (long_rank_mask & long_edge_mask)
    short_entries_raw = (short_rank_mask & short_edge_mask)

    overlap_mask = long_entries_raw & short_entries_raw
    if overlap_mask.any():
        long_entries_raw = long_entries_raw & (~overlap_mask)
        short_entries_raw = short_entries_raw & (~overlap_mask)

    entries_series = long_entries_raw.astype(bool)
    short_entries_series = short_entries_raw.astype(bool)

    n_long_raw = int(entries_series.sum())
    n_short_raw = int(short_entries_series.sum())
    coverage_raw = ((n_long_raw + n_short_raw) / max(len(scores), 1)) * 100.0

    print(
        f"-> OOS raw signal coverage | Longs: {n_long_raw} | Shorts: {n_short_raw} | "
        f"Total: {coverage_raw:.1f}% | TP/SL: {tp_mult}/{sl_mult}"
    )

    atr_series = _compute_wilder_atr(df_raw_oos, period=atr_period).reindex(df_features_oos.index)

    close_prices = df_raw_oos["close"].astype(np.float32)
    open_prices = df_raw_oos["open"].astype(np.float32)

    sl_pct_series = ((atr_series / open_prices) * sl_mult).astype(np.float32)
    tp_pct_series = ((atr_series / open_prices) * tp_mult).astype(np.float32)

    valid_barrier_mask = (
        atr_series.notna()
        & np.isfinite(open_prices)
        & np.isfinite(sl_pct_series)
        & np.isfinite(tp_pct_series)
        & (open_prices > 0.0)
        & (sl_pct_series > 0.0)
        & (tp_pct_series > 0.0)
    )

    if not valid_barrier_mask.any():
        raise ValueError("No valid ATR barrier rows available for OOS evaluation")

    entries_shifted = entries_series.shift(1, fill_value=False)
    short_entries_shifted = short_entries_series.shift(1, fill_value=False)

    entries_shifted = (entries_shifted & valid_barrier_mask).astype(bool)
    short_entries_shifted = (short_entries_shifted & valid_barrier_mask).astype(bool)

    overlap_shifted = entries_shifted & short_entries_shifted
    if overlap_shifted.any():
        entries_shifted = entries_shifted & (~overlap_shifted)
        short_entries_shifted = short_entries_shifted & (~overlap_shifted)

    exits_shifted = pd.Series(False, index=df_features_oos.index, dtype=bool)
    short_exits_shifted = pd.Series(False, index=df_features_oos.index, dtype=bool)

    sl_pct_series = sl_pct_series.where(valid_barrier_mask)
    tp_pct_series = tp_pct_series.where(valid_barrier_mask)

    n_long_exec = int(entries_shifted.sum())
    n_short_exec = int(short_entries_shifted.sum())
    coverage_exec = ((n_long_exec + n_short_exec) / max(len(scores), 1)) * 100.0

    print(
        f"-> Executable OOS coverage | Longs: {n_long_exec} | Shorts: {n_short_exec} | "
        f"Total: {coverage_exec:.1f}%"
    )

    if n_long_exec == 0 and n_short_exec == 0:
        print("-> WARNING: No OOS signals passed execution and ATR validity filters.")

    print("Running VectorBT backtest (fixed ATR barriers, no trailing stop)...")
    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        price=open_prices,
        entries=entries_shifted,
        exits=exits_shifted,
        short_entries=short_entries_shifted,
        short_exits=short_exits_shifted,
        fees=fees,
        slippage=slippage,
        sl_stop=sl_pct_series,
        tp_stop=tp_pct_series,
        sl_trail=False,
        upon_opposite_entry="close",
        freq="30min",
    )

    stats = portfolio.stats()
    print("\n--- Out-of-Sample Results ---")
    print(stats[["Total Return [%]", "Max Drawdown [%]", "Win Rate [%]", "Sharpe Ratio"]])

    metadata = {
        "n_long": n_long_exec,
        "n_short": n_short_exec,
        "coverage_pct": float(coverage_exec),
        "raw_coverage_pct": float(coverage_raw),
        "tp_mult": float(tp_mult),
        "sl_mult": float(sl_mult),
        "absolute_edge_floor": float(absolute_edge_floor),
    }

    return portfolio, stats, metadata
