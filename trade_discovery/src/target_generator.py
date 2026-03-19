import importlib
from typing import Dict

import numba
import numpy as np
import pandas as pd

try:
    config = importlib.import_module("src.config")
    DEFAULT_MAX_HOLD = int(config.ORACLE_MAX_HOLD)
    DEFAULT_TP_MULT = float(config.TP_ATR_MULT)
    DEFAULT_SL_MULT = float(config.SL_ATR_MULT)
    DEFAULT_ATR_PERIOD = int(getattr(config, "ATR_PERIOD", 14))
except (ImportError, AttributeError):
    DEFAULT_MAX_HOLD = 52
    DEFAULT_TP_MULT = 4.0
    DEFAULT_SL_MULT = 1.7
    DEFAULT_ATR_PERIOD = 14


TARGET_SCHEMA_VERSION = "tbm_first_touch_v1"

EVENT_INVALID = 0
EVENT_LONG_TP = 1
EVENT_SHORT_TP = 2
EVENT_LONG_SL = 3
EVENT_SHORT_SL = 4
EVENT_BOTH_TP = 5
EVENT_BOTH_SL = 6
EVENT_AMBIGUOUS = 7
EVENT_TIMEOUT = 8

EVENT_NAME_MAP: Dict[int, str] = {
    EVENT_INVALID: "INVALID",
    EVENT_LONG_TP: "LONG_TP",
    EVENT_SHORT_TP: "SHORT_TP",
    EVENT_LONG_SL: "LONG_SL",
    EVENT_SHORT_SL: "SHORT_SL",
    EVENT_BOTH_TP: "BOTH_TP",
    EVENT_BOTH_SL: "BOTH_SL",
    EVENT_AMBIGUOUS: "AMBIGUOUS",
    EVENT_TIMEOUT: "TIMEOUT",
}


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE UTILITY: Wilder's ATR (RMA)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_wilder_atr(df_raw: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Wilder ATR using float64 arithmetic for numerical stability.
    """
    high_low = df_raw["high"] - df_raw["low"]
    high_close = (df_raw["high"] - df_raw["close"].shift(1)).abs()
    low_close = (df_raw["low"] - df_raw["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr = true_range.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr


# ─────────────────────────────────────────────────────────────────────────────
# NUMBA CORE - STRICT FIRST-TOUCH ASYMMETRIC TRIPLE BARRIER
# ─────────────────────────────────────────────────────────────────────────────

@numba.njit(cache=True, fastmath=True)
def run_first_touch_triple_barrier(
    open_arr,
    high_arr,
    low_arr,
    close_arr,
    atr_arr,
    max_hold,
    tp_mult,
    sl_mult,
):
    """
    Strict first-touch asymmetric triple-barrier labeling.

    Decision time: bar i
    Entry anchor:  next bar open, open_arr[i + 1]

    Output target policy:
      +1.0 -> Long TP wins and short TP does not also succeed
      -1.0 -> Short TP wins and long TP does not also succeed
       0.0 -> Timeout, ambiguity, or stop-only outcomes
       NaN -> Invalid sample
    """
    n = len(close_arr)

    targets = np.empty(n, dtype=np.float32)
    targets[:] = np.nan

    event_type = np.zeros(n, dtype=np.int8)
    event_bar = np.full(n, -1, dtype=np.int64)
    entry_price_out = np.full(n, np.nan, dtype=np.float32)
    exit_price_out = np.full(n, np.nan, dtype=np.float32)
    atr_out = np.full(n, np.nan, dtype=np.float32)
    is_valid = np.zeros(n, dtype=np.uint8)

    if max_hold <= 0:
        return targets, event_type, event_bar, entry_price_out, exit_price_out, atr_out, is_valid

    last_decision = n - max_hold
    for i in range(last_decision):
        entry_idx = i + 1
        entry_price = open_arr[entry_idx]
        atr_val = atr_arr[i]

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

        res_l = 0
        res_s = 0
        res_l_bar = -1
        res_s_bar = -1

        invalid_path = False
        end_idx = i + max_hold

        for idx in range(entry_idx, end_idx + 1):
            c_high = high_arr[idx]
            c_low = low_arr[idx]

            if not np.isfinite(c_high) or not np.isfinite(c_low):
                invalid_path = True
                break

            if res_l == 0:
                long_hit_tp = c_high >= l_tp
                long_hit_sl = c_low <= l_sl

                if long_hit_tp and long_hit_sl:
                    res_l = -2
                    res_l_bar = idx
                elif long_hit_tp:
                    res_l = 1
                    res_l_bar = idx
                elif long_hit_sl:
                    res_l = -1
                    res_l_bar = idx

            if res_s == 0:
                short_hit_tp = c_low <= s_tp
                short_hit_sl = c_high >= s_sl

                if short_hit_tp and short_hit_sl:
                    res_s = -2
                    res_s_bar = idx
                elif short_hit_tp:
                    res_s = 1
                    res_s_bar = idx
                elif short_hit_sl:
                    res_s = -1
                    res_s_bar = idx

            if res_l != 0 and res_s != 0:
                break

        if invalid_path:
            continue

        is_valid[i] = 1
        entry_price_out[i] = np.float32(entry_price)
        atr_out[i] = np.float32(atr_val)

        if res_l == 1 and res_s != 1:
            targets[i] = np.float32(1.0)
            event_type[i] = EVENT_LONG_TP
            event_bar[i] = res_l_bar
            exit_price_out[i] = np.float32(l_tp)

        elif res_s == 1 and res_l != 1:
            targets[i] = np.float32(-1.0)
            event_type[i] = EVENT_SHORT_TP
            event_bar[i] = res_s_bar
            exit_price_out[i] = np.float32(s_tp)

        elif res_l == 1 and res_s == 1:
            targets[i] = np.float32(0.0)
            event_type[i] = EVENT_BOTH_TP
            if res_l_bar >= 0 and res_s_bar >= 0:
                event_bar[i] = min(res_l_bar, res_s_bar)

        elif res_l == -2 or res_s == -2:
            targets[i] = np.float32(0.0)
            event_type[i] = EVENT_AMBIGUOUS
            if res_l_bar >= 0 and res_s_bar >= 0:
                event_bar[i] = min(res_l_bar, res_s_bar)
            elif res_l_bar >= 0:
                event_bar[i] = res_l_bar
            else:
                event_bar[i] = res_s_bar

        elif res_l == -1 and res_s == -1:
            targets[i] = np.float32(0.0)
            event_type[i] = EVENT_BOTH_SL
            if res_l_bar >= 0 and res_s_bar >= 0:
                event_bar[i] = min(res_l_bar, res_s_bar)
            exit_price_out[i] = np.float32(entry_price)

        elif res_l == -1 and res_s == 0:
            targets[i] = np.float32(0.0)
            event_type[i] = EVENT_LONG_SL
            event_bar[i] = res_l_bar
            exit_price_out[i] = np.float32(l_sl)

        elif res_s == -1 and res_l == 0:
            targets[i] = np.float32(0.0)
            event_type[i] = EVENT_SHORT_SL
            event_bar[i] = res_s_bar
            exit_price_out[i] = np.float32(s_sl)

        else:
            targets[i] = np.float32(0.0)
            event_type[i] = EVENT_TIMEOUT
            event_bar[i] = end_idx
            final_close = close_arr[end_idx]
            if np.isfinite(final_close):
                exit_price_out[i] = np.float32(final_close)

    return targets, event_type, event_bar, entry_price_out, exit_price_out, atr_out, is_valid


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_tbm_targets(
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    max_hold: int = DEFAULT_MAX_HOLD,
    atr_period: int = DEFAULT_ATR_PERIOD,
    tp_mult: float = DEFAULT_TP_MULT,
    sl_mult: float = DEFAULT_SL_MULT,
    target_mode: str = "first_touch_class",
    timeout_policy: str = "neutral",
    ambiguity_policy: str = "neutral",
    drop_invalid: bool = True,
    return_metadata: bool = False,
):
    """
    Generate asymmetric next-open TBM targets aligned to df_features.

    Current supported target_mode:
      - "first_touch_class"

    Returns:
      - (X_aligned, y_aligned) by default
      - (X_aligned, y_aligned, meta_aligned) when return_metadata=True
    """
    if max_hold <= 0:
        raise ValueError("max_hold must be > 0")
    if atr_period <= 0:
        raise ValueError("atr_period must be > 0")
    if tp_mult <= 0 or sl_mult <= 0:
        raise ValueError("Multipliers must be > 0")
    if target_mode != "first_touch_class":
        raise NotImplementedError(
            "Only target_mode='first_touch_class' is currently supported."
        )
    if timeout_policy != "neutral":
        raise NotImplementedError(
            "Only timeout_policy='neutral' is currently supported."
        )
    if ambiguity_policy != "neutral":
        raise NotImplementedError(
            "Only ambiguity_policy='neutral' is currently supported."
        )
    if not {"open", "high", "low", "close"}.issubset(set(df_raw.columns)):
        raise ValueError("df_raw must contain open, high, low, close columns")

    print(
        "Generating Asymmetric TBM targets\n"
        f"  [schema={TARGET_SCHEMA_VERSION}, mode={target_mode}, "
        f"ATR({atr_period}), TP_mult={tp_mult}, SL_mult={sl_mult}, max_hold={max_hold}]"
    )

    atr = _compute_wilder_atr(df_raw, period=atr_period)

    valid_mask = atr.notna()
    df_raw_valid = df_raw.loc[valid_mask]
    atr_valid = atr.loc[valid_mask]

    warmup_dropped = int((~valid_mask).sum())
    print(f"  ATR warmup rows dropped: {warmup_dropped} (first {atr_period - 1} bars)")

    (
        targets,
        event_type,
        event_bar,
        entry_price_arr,
        exit_price_arr,
        atr_used_arr,
        is_valid_arr,
    ) = run_first_touch_triple_barrier(
        df_raw_valid["open"].to_numpy(dtype=np.float32),
        df_raw_valid["high"].to_numpy(dtype=np.float32),
        df_raw_valid["low"].to_numpy(dtype=np.float32),
        df_raw_valid["close"].to_numpy(dtype=np.float32),
        atr_valid.to_numpy(dtype=np.float32),
        np.int64(max_hold),
        np.float32(tp_mult),
        np.float32(sl_mult),
    )

    meta = pd.DataFrame(
        {
            "target": pd.Series(targets, index=df_raw_valid.index, dtype=np.float32),
            "event_code": pd.Series(event_type, index=df_raw_valid.index, dtype=np.int8),
            "event_bar_pos": pd.Series(event_bar, index=df_raw_valid.index, dtype=np.int64),
            "entry_price": pd.Series(entry_price_arr, index=df_raw_valid.index, dtype=np.float32),
            "exit_price": pd.Series(exit_price_arr, index=df_raw_valid.index, dtype=np.float32),
            "atr_value": pd.Series(atr_used_arr, index=df_raw_valid.index, dtype=np.float32),
            "is_valid": pd.Series(is_valid_arr.astype(bool), index=df_raw_valid.index),
        }
    )

    meta["event_type"] = (
        meta["event_code"]
        .map(EVENT_NAME_MAP)
        .fillna("UNKNOWN")
        .astype("string")
    )
    meta["target_schema_version"] = TARGET_SCHEMA_VERSION

    common_index = df_features.index.intersection(meta.index)
    if len(common_index) <= max_hold:
        raise ValueError("Not enough aligned rows.")

    decision_index = common_index[:-max_hold]
    df_features_aligned = df_features.loc[decision_index]
    meta_aligned = meta.loc[decision_index].copy()

    if drop_invalid:
        keep_mask = meta_aligned["is_valid"] & meta_aligned["target"].notna()
        dropped_invalid = int((~keep_mask).sum())
        if dropped_invalid > 0:
            print(f"  Dropping invalid decision rows: {dropped_invalid}")
        df_features_aligned = df_features_aligned.loc[keep_mask]
        meta_aligned = meta_aligned.loc[keep_mask]
    else:
        dropped_invalid = int((~meta_aligned["is_valid"]).sum())

    y_targets_aligned = meta_aligned["target"].astype(np.float32)
    y_targets_aligned.name = "target"

    if len(df_features_aligned) == 0:
        raise ValueError("All aligned rows were invalid after filtering.")

    n_valid = int(meta_aligned["is_valid"].sum())
    n_invalid = dropped_invalid if drop_invalid else int((~meta_aligned["is_valid"]).sum())
    n_long = int((y_targets_aligned == 1.0).sum())
    n_short = int((y_targets_aligned == -1.0).sum())
    n_neutral = int((y_targets_aligned == 0.0).sum())

    event_counts = meta_aligned["event_type"].value_counts(dropna=False).to_dict()
    timeout_count = int(event_counts.get("TIMEOUT", 0))
    ambiguous_count = int(event_counts.get("AMBIGUOUS", 0))
    both_tp_count = int(event_counts.get("BOTH_TP", 0))
    both_sl_count = int(event_counts.get("BOTH_SL", 0))

    print(
        "Target generation complete.\n"
        f"  Scored rows: {len(meta_aligned)} | Valid: {n_valid} | Invalid dropped: {n_invalid}\n"
        f"  Label distribution -> Long: {n_long} | Short: {n_short} | Neutral: {n_neutral}\n"
        f"  Event distribution -> Timeout: {timeout_count} | Ambiguous: {ambiguous_count} | "
        f"Both TP: {both_tp_count} | Both SL: {both_sl_count}"
    )

    if return_metadata:
        return df_features_aligned, y_targets_aligned, meta_aligned

    return df_features_aligned, y_targets_aligned


# Backward-compatible alias
def generate_oracle_targets(
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    max_hold: int = DEFAULT_MAX_HOLD,
    atr_period: int = DEFAULT_ATR_PERIOD,
    atr_mult: float = DEFAULT_TP_MULT,
    target_mode: str = "first_touch_class",
    timeout_policy: str = "neutral",
    ambiguity_policy: str = "neutral",
    drop_invalid: bool = True,
    return_metadata: bool = False,
):
    return generate_tbm_targets(
        df_raw=df_raw,
        df_features=df_features,
        max_hold=max_hold,
        atr_period=atr_period,
        tp_mult=atr_mult,
        sl_mult=atr_mult,
        target_mode=target_mode,
        timeout_policy=timeout_policy,
        ambiguity_policy=ambiguity_policy,
        drop_invalid=drop_invalid,
        return_metadata=return_metadata,
    )
