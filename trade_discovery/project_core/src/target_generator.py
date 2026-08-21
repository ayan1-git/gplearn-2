import importlib
from typing import Dict

import numba
import numpy as np
import pandas as pd

try:
    config = importlib.import_module("src.config")
    DEFAULT_MAX_HOLD = int(config.ORACLE_MAX_HOLD)
    DEFAULT_TP_MULT  = float(config.TP_ATR_MULT)
    DEFAULT_SL_MULT  = float(config.SL_ATR_MULT)
except (ImportError, AttributeError):
    DEFAULT_MAX_HOLD = 52
    DEFAULT_TP_MULT  = 4.0
    DEFAULT_SL_MULT  = 2.0

TARGET_SCHEMA_VERSION = "tbm_first_touch_v2"

EVENT_INVALID   = 0
EVENT_LONG_TP   = 1
EVENT_SHORT_TP  = 2
EVENT_LONG_SL   = 3
EVENT_SHORT_SL  = 4
EVENT_BOTH_TP   = 5
EVENT_BOTH_SL   = 6
EVENT_AMBIGUOUS = 7
EVENT_TIMEOUT   = 8

# BUG-4 FIX: New sentinel for wide-candle single-bar Both-SL
EVENT_WIDE_CANDLE = 9

EVENT_NAME_MAP: Dict[int, str] = {
    EVENT_INVALID:    "INVALID",
    EVENT_LONG_TP:    "LONG_TP",
    EVENT_SHORT_TP:   "SHORT_TP",
    EVENT_LONG_SL:    "LONG_SL",
    EVENT_SHORT_SL:   "SHORT_SL",
    EVENT_BOTH_TP:    "BOTH_TP",
    EVENT_BOTH_SL:    "BOTH_SL",
    EVENT_AMBIGUOUS:  "AMBIGUOUS",
    EVENT_TIMEOUT:    "TIMEOUT",
    EVENT_WIDE_CANDLE:"WIDE_CANDLE",
}


def _compute_wilder_atr(df_raw: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder ATR using float64 for numerical stability.
    BUG-2 FIX: Use a proper warmup — first `period` bars are NaN,
    then switch to recursive Wilder smoothing to avoid EWM underestimation.
    """
    high_low    = df_raw["high"] - df_raw["low"]
    high_close  = (df_raw["high"] - df_raw["close"].shift(1)).abs()
    low_close   = (df_raw["low"]  - df_raw["close"].shift(1)).abs()
    true_range  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Wilder smoothing: seed with SMA of first `period` TRs, then recurse
    # Use an explicit writable float64 buffer — `Series.values` can be a
    # read-only view depending on the pandas/numpy build, which breaks the
    # in-place loop below ("assignment destination is read-only").
    alpha = 1.0 / period
    values = true_range.to_numpy(dtype=float, copy=True)
    atr_values = np.empty(len(values), dtype=float)
    atr_values[:period] = np.nan

    # Seed value: simple mean of first `period` true ranges
    atr_values[period - 1] = true_range.iloc[:period].mean()

    for k in range(period, len(values)):
        atr_values[k] = atr_values[k - 1] * (1.0 - alpha) + values[k] * alpha

    return pd.Series(atr_values, index=df_raw.index)


@numba.njit(cache=True, fastmath=True)
def run_first_touch_triple_barrier(
    open_arr, high_arr, low_arr, close_arr,
    atr_arr, max_hold, tp_mult, sl_mult,
    exclude_both_tp,
):
    """
    Strict first-touch asymmetric triple-barrier labeling.

    BUG-1 FIX: Wide-candle Both-SL (single bar hitting both SL walls) is
    now classified as EVENT_WIDE_CANDLE (label=-99) instead of EVENT_BOTH_SL,
    allowing the pipeline to drop these structurally uninformative rows.

    BUG-5 FIX: Both-SL exit price is set to the first SL barrier touched,
    not entry_price.

    FIX #8: When exclude_both_tp=True, BOTH_TP events (both TP barriers hit
    within the same bar) are labelled -99 instead of +1. These events are
    directionally ambiguous (~50/50 with symmetric wicks), so hardcoding +1
    injected a systematic long bias into the training labels.

    Labels:
      +1.0  → Long TP first
      -1.0  → Short TP first
        0.0  → Timeout / ambiguous / SL-only events
      -99.0  → Wide-candle / Both-SL / Both-TP (excluded from training)
    """
    n = len(close_arr)

    targets        = np.empty(n, dtype=np.float32)
    targets[:]     = np.nan
    event_type     = np.zeros(n, dtype=np.int8)
    event_bar      = np.full(n, -1, dtype=np.int64)
    entry_price_out= np.full(n, np.nan, dtype=np.float32)
    exit_price_out = np.full(n, np.nan, dtype=np.float32)
    atr_out        = np.full(n, np.nan, dtype=np.float32)
    is_valid       = np.zeros(n, dtype=np.uint8)

    if max_hold <= 0:
        return targets, event_type, event_bar, entry_price_out, exit_price_out, atr_out, is_valid

    last_decision = n - max_hold
    for i in range(last_decision):
        entry_idx    = i + 1
        entry_price  = open_arr[entry_idx]
        atr_val      = atr_arr[i]

        if not np.isfinite(entry_price) or not np.isfinite(atr_val):
            continue
        if entry_price <= 0.0 or atr_val <= 0.0:
            continue

        tp_dist = atr_val * tp_mult
        sl_dist = atr_val * sl_mult

        if tp_dist <= 0.0 or sl_dist <= 0.0:
            continue

        l_tp = entry_price + tp_dist
        l_sl = entry_price - sl_dist
        s_tp = entry_price - tp_dist
        s_sl = entry_price + sl_dist

        # ── BUG-1 FIX: Check first bar (entry bar) for wide-candle Both-SL ──
        # If the entry bar itself spans >= 2×sl_dist, both SL barriers are
        # triggered simultaneously on bar i+1. This is not a real trade —
        # it is a structural data artifact (large-wick candle, auction gap, etc.)
        # Label as WIDE_CANDLE and exclude from training.
        entry_bar_range = high_arr[entry_idx] - low_arr[entry_idx]
        if entry_bar_range >= 2.0 * sl_dist:
            is_valid[i]        = 1
            targets[i]         = np.float32(-99.0)
            event_type[i]      = EVENT_WIDE_CANDLE
            event_bar[i]       = entry_idx
            entry_price_out[i] = np.float32(entry_price)
            atr_out[i]         = np.float32(atr_val)
            continue

        res_l = 0; res_s = 0
        res_l_bar = -1; res_s_bar = -1
        invalid_path = False
        end_idx = i + max_hold

        for idx in range(entry_idx, end_idx + 1):
            c_high = high_arr[idx]
            c_low  = low_arr[idx]

            if not np.isfinite(c_high) or not np.isfinite(c_low):
                invalid_path = True
                break

            if res_l == 0:
                long_hit_tp = c_high >= l_tp
                long_hit_sl = c_low  <= l_sl
                if long_hit_tp and long_hit_sl:
                    res_l = -2; res_l_bar = idx      # ambiguous bar for long
                elif long_hit_tp:
                    res_l = 1;  res_l_bar = idx
                elif long_hit_sl:
                    res_l = -1; res_l_bar = idx

            if res_s == 0:
                short_hit_tp = c_low  <= s_tp
                short_hit_sl = c_high >= s_sl
                if short_hit_tp and short_hit_sl:
                    res_s = -2; res_s_bar = idx      # ambiguous bar for short
                elif short_hit_tp:
                    res_s = 1;  res_s_bar = idx
                elif short_hit_sl:
                    res_s = -1; res_s_bar = idx

            if res_l != 0 and res_s != 0:
                break

        if invalid_path:
            continue

        is_valid[i]        = 1
        entry_price_out[i] = np.float32(entry_price)
        atr_out[i]         = np.float32(atr_val)

        if res_l == 1 and res_s != 1:
            targets[i]        = np.float32(1.0)
            event_type[i]     = EVENT_LONG_TP
            event_bar[i]      = res_l_bar
            exit_price_out[i] = np.float32(l_tp)

        elif res_s == 1 and res_l != 1:
            targets[i]        = np.float32(-1.0)
            event_type[i]     = EVENT_SHORT_TP
            event_bar[i]      = res_s_bar
            exit_price_out[i] = np.float32(s_tp)

        elif res_l == 1 and res_s == 1:
            # Both TP levels hit on the same bar: asymmetric barrier breakout.
            # FIX #8: directionally ambiguous (~50/50 with symmetric wicks).
            # When exclude_both_tp=True label as -99 so the pipeline drops
            # these rows; otherwise fall back to the legacy Long (+1) label.
            if exclude_both_tp:
                targets[i]        = np.float32(-99.0)
            else:
                targets[i]        = np.float32(1.0)
            event_type[i]     = EVENT_BOTH_TP
            event_bar[i]      = min(res_l_bar, res_s_bar) if res_l_bar >= 0 and res_s_bar >= 0 else -1

        elif res_l == -2 or res_s == -2:
            targets[i]        = np.float32(0.0)
            event_type[i]     = EVENT_AMBIGUOUS
            if res_l_bar >= 0 and res_s_bar >= 0:
                event_bar[i]  = min(res_l_bar, res_s_bar)
            elif res_l_bar >= 0:
                event_bar[i]  = res_l_bar
            else:
                event_bar[i]  = res_s_bar

        elif res_l == -1 and res_s == -1:
            # BUG-1 FIX: multi-bar Both-SL (whipsaw over multiple bars)
            # Kept as EVENT_BOTH_SL label=-99 to allow pipeline exclusion
            targets[i]        = np.float32(-99.0)
            event_type[i]     = EVENT_BOTH_SL
            event_bar[i]      = min(res_l_bar, res_s_bar) if res_l_bar >= 0 and res_s_bar >= 0 else -1
            # BUG-5 FIX: exit price = first SL level touched (not entry_price)
            if res_l_bar <= res_s_bar:
                exit_price_out[i] = np.float32(l_sl)   # long SL hit first
            else:
                exit_price_out[i] = np.float32(s_sl)   # short SL hit first

        elif res_l == -1 and res_s == 0:
            targets[i]        = np.float32(0.0)
            event_type[i]     = EVENT_LONG_SL
            event_bar[i]      = res_l_bar
            exit_price_out[i] = np.float32(l_sl)

        elif res_s == -1 and res_l == 0:
            targets[i]        = np.float32(0.0)
            event_type[i]     = EVENT_SHORT_SL
            event_bar[i]      = res_s_bar
            exit_price_out[i] = np.float32(s_sl)

        else:
            targets[i]        = np.float32(0.0)
            event_type[i]     = EVENT_TIMEOUT
            event_bar[i]      = end_idx
            final_close = close_arr[end_idx]
            if np.isfinite(final_close):
                exit_price_out[i] = np.float32(final_close)

    return targets, event_type, event_bar, entry_price_out, exit_price_out, atr_out, is_valid


def generate_tbm_targets(
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    max_hold: int   = DEFAULT_MAX_HOLD,
    atr_period: int = 14,
    tp_mult: float  = DEFAULT_TP_MULT,
    sl_mult: float  = DEFAULT_SL_MULT,
    target_mode: str        = "first_touch_class",
    timeout_policy: str     = "neutral",
    ambiguity_policy: str   = "neutral",
    drop_invalid: bool      = True,
    drop_both_sl: bool      = True,    # Filter out -99
    drop_neutral: bool      = False,   # New: filter out 0.0
    exclude_both_tp: bool   = False,   # FIX #8: exclude BOTH_TP rows (-99)
    return_metadata: bool   = False,
):
    if max_hold <= 0:        raise ValueError("max_hold must be > 0")
    if atr_period <= 0:      raise ValueError("atr_period must be > 0")
    if tp_mult <= 0 or sl_mult <= 0: raise ValueError("Multipliers must be > 0")
    if target_mode != "first_touch_class":
        raise NotImplementedError("Only target_mode='first_touch_class' is supported.")

    print(
        f"Generating Asymmetric TBM targets\n"
        f"  [schema={TARGET_SCHEMA_VERSION}, mode={target_mode}, "
        f"ATR({atr_period}), TP_mult={tp_mult}, SL_mult={sl_mult}, max_hold={max_hold}]"
    )

    atr = _compute_wilder_atr(df_raw, period=atr_period)

    valid_mask    = atr.notna()
    df_raw_valid  = df_raw.loc[valid_mask]
    atr_valid     = atr.loc[valid_mask]

    # BUG-7 FIX: warmup is `period` bars, not `period-1`
    warmup_dropped = int((~valid_mask).sum())
    print(f"  ATR warmup rows dropped: {warmup_dropped} (first {atr_period} bars)")

    (targets, event_type, event_bar,
     entry_price_arr, exit_price_arr,
     atr_used_arr, is_valid_arr) = run_first_touch_triple_barrier(
        df_raw_valid["open"].to_numpy(dtype=np.float32),
        df_raw_valid["high"].to_numpy(dtype=np.float32),
        df_raw_valid["low"].to_numpy(dtype=np.float32),
        df_raw_valid["close"].to_numpy(dtype=np.float32),
        atr_valid.to_numpy(dtype=np.float32),
        np.int64(max_hold),
        np.float32(tp_mult),
        np.float32(sl_mult),
        bool(exclude_both_tp),
    )

    meta = pd.DataFrame({
        "target":         pd.Series(targets,               index=df_raw_valid.index, dtype=np.float32),
        "event_code":     pd.Series(event_type,            index=df_raw_valid.index, dtype=np.int8),
        "event_bar_pos":  pd.Series(event_bar,             index=df_raw_valid.index, dtype=np.int64),
        "entry_price":    pd.Series(entry_price_arr,       index=df_raw_valid.index, dtype=np.float32),
        "exit_price":     pd.Series(exit_price_arr,        index=df_raw_valid.index, dtype=np.float32),
        "atr_value":      pd.Series(atr_used_arr,          index=df_raw_valid.index, dtype=np.float32),
        "is_valid":       pd.Series(is_valid_arr.astype(bool), index=df_raw_valid.index),
    })
    meta["event_type"]           = meta["event_code"].map(EVENT_NAME_MAP).fillna("UNKNOWN").astype("string")
    meta["target_schema_version"]= TARGET_SCHEMA_VERSION

    common_index   = df_features.index.intersection(meta.index)
    if len(common_index) <= max_hold:
        raise ValueError("Not enough aligned rows.")

    # BUG-3 FIX: kernel already NaN-marks last max_hold rows via last_decision.
    # Only drop the final max_hold rows once, not twice.
    decision_index       = common_index[:-max_hold]
    df_features_aligned  = df_features.loc[decision_index]
    meta_aligned         = meta.loc[decision_index].copy()

    # Standard invalid drop
    if drop_invalid:
        keep_mask      = meta_aligned["is_valid"] & meta_aligned["target"].notna()
        dropped_invalid = int((~keep_mask).sum())
        if dropped_invalid > 0:
            print(f"  Dropping invalid decision rows: {dropped_invalid}")
        df_features_aligned = df_features_aligned.loc[keep_mask]
        meta_aligned        = meta_aligned.loc[keep_mask]

    # BUG-1/4 FIX: Drop Both-SL and Wide-Candle rows (-99) from training.
    # FIX #8: also covers BOTH_TP rows when exclude_both_tp=True (same -99
    # sentinel, dropped by this mask).
    if drop_both_sl:
        whipsaw_mask        = meta_aligned["target"] != -99.0
        n_whipsaw_dropped   = int((~whipsaw_mask).sum())
        if n_whipsaw_dropped > 0:
            print(f"  Dropping excluded rows (-99: Both-SL + Wide-Candle"
                  f"{' + Both-TP' if exclude_both_tp else ''}): {n_whipsaw_dropped}")
        df_features_aligned = df_features_aligned.loc[whipsaw_mask]
        meta_aligned        = meta_aligned.loc[whipsaw_mask]

    # New: Drop Neutral rows (0.0) from training if requested
    if drop_neutral:
        neutral_mask        = meta_aligned["target"] != 0.0
        n_neutral_dropped   = int((~neutral_mask).sum())
        if n_neutral_dropped > 0:
            print(f"  Dropping neutral rows (0.0): {n_neutral_dropped}")
        df_features_aligned = df_features_aligned.loc[neutral_mask]
        meta_aligned        = meta_aligned.loc[neutral_mask]

    y_targets_aligned      = meta_aligned["target"].astype(np.float32)
    y_targets_aligned.name = "target"

    if len(df_features_aligned) == 0:
        raise ValueError("All aligned rows were invalid after filtering.")

    n_valid   = int(meta_aligned["is_valid"].sum())
    n_long    = int((y_targets_aligned ==  1.0).sum())
    n_short   = int((y_targets_aligned == -1.0).sum())
    n_neutral = int((y_targets_aligned ==  0.0).sum())

    event_counts     = meta_aligned["event_type"].value_counts(dropna=False).to_dict()
    timeout_count    = int(event_counts.get("TIMEOUT",     0))
    ambiguous_count  = int(event_counts.get("AMBIGUOUS",   0))
    both_tp_count    = int(event_counts.get("BOTH_TP",     0))
    both_sl_count    = int(event_counts.get("BOTH_SL",     0))
    wide_candle_count= int(event_counts.get("WIDE_CANDLE", 0))
    long_tp_count    = int(event_counts.get("LONG_TP",     0))
    short_tp_count   = int(event_counts.get("SHORT_TP",    0))
    long_sl_count    = int(event_counts.get("LONG_SL",     0))
    short_sl_count   = int(event_counts.get("SHORT_SL",    0))

    # Full event breakdown (all event types, not just the rare edge cases).
    full_event_breakdown = " | ".join(
        f"{k}: {v}" for k, v in sorted(event_counts.items(), key=lambda kv: -kv[1])
    )

    print(
        f"Target generation complete.\n"
        f"  Scored rows: {len(meta_aligned)} | Valid: {n_valid}\n"
        f"  Label distribution → Long: {n_long} | Short: {n_short} | Neutral: {n_neutral}\n"
        f"  Event distribution → {full_event_breakdown}"
    )

    # Sanity assertion
    both_sl_raw_rate = (both_sl_count + wide_candle_count) / max(len(meta_aligned), 1)
    if both_sl_raw_rate > 0.15:
        import warnings
        warnings.warn(
            f"Both-SL + Wide-Candle rate={both_sl_raw_rate:.1%} > 15%%. "
            f"Consider reducing ORACLE_MAX_HOLD or SL_ATR_MULT.",
            RuntimeWarning, stacklevel=2,
        )

    if return_metadata:
        return df_features_aligned, y_targets_aligned, meta_aligned
    return df_features_aligned, y_targets_aligned
