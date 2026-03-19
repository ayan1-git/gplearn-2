from __future__ import annotations

import gc
import importlib
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    config             = importlib.import_module("src.config")
    OB_ATR_MULT        = float(config.OB_ATR_MULT)
    OB_INTERNAL_LB     = int(config.OB_INTERNAL_LB)
    OB_SWING_LB        = int(config.OB_SWING_LB)
    OB_MAX_OBS         = int(config.OB_MAX_OBS)
    OB_IOU_THRESHOLD   = float(config.OB_IOU_THRESHOLD)
    FE_MISSING_FILL    = float(config.FE_MISSING_FILL)

    FE_MOMENTUM_PERIOD = int(config.FE_MOMENTUM_PERIOD)
    FE_VOL_SHORT_PERIOD = int(config.FE_VOL_SHORT_PERIOD)
    FE_VOL_LONG_PERIOD  = int(config.FE_VOL_LONG_PERIOD)
    FE_SKEW_PERIOD      = int(config.FE_SKEW_PERIOD)
    FE_ZSCORE_PERIOD    = int(config.FE_ZSCORE_PERIOD)
    FE_ICP_PERIOD       = int(config.FE_ICP_PERIOD)
    FE_MDS_FAST_WINDOW  = int(config.FE_MDS_FAST_WINDOW)
    FE_MDS_SLOW_WINDOW  = int(config.FE_MDS_SLOW_WINDOW)
    FE_VOL_ASYM_WINDOW  = int(config.FE_VOL_ASYM_WINDOW)
    FE_STOCH_PERIOD     = int(config.FE_STOCH_PERIOD)
    FE_ADX_PERIOD       = int(config.FE_ADX_PERIOD)
    FE_BAR_PER_DAY      = int(config.FE_BAR_PER_DAY)
except (ImportError, AttributeError):
    OB_ATR_MULT        = 0.5
    OB_INTERNAL_LB     = 5
    OB_SWING_LB        = 20
    OB_MAX_OBS         = 5
    OB_IOU_THRESHOLD   = 0.85
    FE_MISSING_FILL    = 5.0

    FE_MOMENTUM_PERIOD = 14
    FE_VOL_SHORT_PERIOD = 6
    FE_VOL_LONG_PERIOD  = 100
    FE_SKEW_PERIOD      = 28
    FE_ZSCORE_PERIOD    = 50
    FE_ICP_PERIOD       = 14
    FE_MDS_FAST_WINDOW  = 5
    FE_MDS_SLOW_WINDOW  = 30
    FE_VOL_ASYM_WINDOW  = 20
    FE_STOCH_PERIOD     = 14
    FE_ADX_PERIOD       = 14
    FE_BAR_PER_DAY      = 13

EPS = 1e-12

# --- FEATURE CONTRACTS FOR WFO SCALING ---

PASSTHROUGH_FEATURES = [
    "feat_ob_supp_active",
    "feat_ob_res_active",
    "feat_session_sin",
    "feat_session_cos",
    "feat_icp",
    "feat_efficiency",
    "feat_ob_supp_touches",
    "feat_ob_res_touches",
    "feat_momentum_rsi",
    "feat_rejection_upper",
    "feat_rejection_lower",
    "feat_local_structure",
    "feat_session_gap",
]

SCALE_FEATURES = [
    "feat_volatility_regime",
    "feat_dist_skew",
    "feat_zscore",
    "feat_momentum_mds",
    "feat_vol_asymmetry",
    "feat_ob_dist_supp",
    "feat_ob_dist_res",
    "feat_vol_squeeze",
]


def clip_scale(series: pd.Series, bound: float = 1.0) -> pd.Series:
    if bound <= 0:
        raise ValueError("clip_scale bound must be > 0")
    return series.clip(-bound, bound)


try:
    config = importlib.import_module("src.config")
    DEFAULT_OPEN = str(config.SESSION_OPEN)
    DEFAULT_CLOSE = str(config.SESSION_CLOSE)
    DEFAULT_TZ = str(config.SESSION_TZ)
except (ImportError, AttributeError):
    DEFAULT_OPEN = "09:15"
    DEFAULT_CLOSE = "15:30"
    DEFAULT_TZ = "Asia/Kolkata"

@dataclass(frozen=True)
class SessionConfig:
    open_time: str = DEFAULT_OPEN
    close_time: str = DEFAULT_CLOSE
    tz: str = DEFAULT_TZ


def parse_hhmm(hhmm: str) -> Tuple[int, int]:
    parts = hhmm.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid hhmm time string: {hhmm}")

    h = int(parts[0])
    m = int(parts[1])

    if not (0 <= h <= 23) or not (0 <= m <= 59):
        raise ValueError(f"Invalid time: {hhmm}")

    return h, m


def session_cyclic_position(
    index: pd.DatetimeIndex,
    session: SessionConfig,
    clip_outside_session: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex for session features")

    idx = index
    if idx.tz is not None:
        idx = idx.tz_convert(session.tz)

    open_h, open_m = parse_hhmm(session.open_time)
    close_h, close_m = parse_hhmm(session.close_time)

    open_minutes = open_h * 60 + open_m
    close_minutes = close_h * 60 + close_m
    session_len = close_minutes - open_minutes
    if session_len <= 0:
        raise ValueError("Session close must be after session open")

    minutes = pd.Series(
        idx.hour.astype(np.int32) * 60 + idx.minute.astype(np.int32),
        index=index,
        name="minutes",
    )

    if clip_outside_session:
        minutes = minutes.clip(lower=open_minutes, upper=close_minutes)
    else:
        minutes = minutes.where(
            (minutes >= open_minutes) & (minutes <= close_minutes),
            np.nan,
        )

    pos = (minutes - open_minutes) / float(session_len)
    angle = 2.0 * np.pi * pos

    feat_sin = np.sin(angle).astype(np.float32).rename("feat_session_sin")
    feat_cos = np.cos(angle).astype(np.float32).rename("feat_session_cos")
    return feat_sin, feat_cos


def momentum_divergence_score(
    log_ret: pd.Series,
    fast_window: int,
    slow_window: int,
) -> pd.Series:
    if fast_window < 1 or slow_window < 2:
        raise ValueError("fast_window must be >= 1 and slow_window must be >= 2")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be < slow_window")

    fast_sum = log_ret.rolling(fast_window, min_periods=fast_window).sum()
    slow_sum = log_ret.rolling(slow_window, min_periods=slow_window).sum()
    slow_vol = log_ret.rolling(slow_window, min_periods=slow_window).std() + EPS

    mds = (fast_sum - slow_sum) / slow_vol
    return mds.rename("feat_momentum_mds")


def directional_vol_asymmetry(
    log_ret: pd.Series,
    window: int,
) -> pd.Series:
    if window < 2:
        raise ValueError("window must be >= 2")

    up = log_ret.clip(lower=0.0)
    down = log_ret.clip(upper=0.0)

    up_vol = up.rolling(window, min_periods=window).std()
    down_vol = down.rolling(window, min_periods=window).std()

    raw = (up_vol - down_vol) / (up_vol + down_vol + EPS)
    return raw.rename("feat_vol_asymmetry")


class OptimizedOrderBlockEngine:
    """
    Policy:
    - Zone touch is wick-based.
    - Zone invalidation is close-through-based.
    - Pivot confirmation occurs only after the full left/right lookback window exists.
    """

    def __init__(
        self,
        internal_lookback: int = OB_INTERNAL_LB,
        swing_lookback: int = OB_SWING_LB,
        atr_multiplier: float = OB_ATR_MULT,
        max_obs: int = OB_MAX_OBS,
        iou_threshold: float = OB_IOU_THRESHOLD,
        missing_value_fill: float = FE_MISSING_FILL,
    ):
        if internal_lookback < 1 or swing_lookback < 1:
            raise ValueError("Lookbacks must be >= 1")
        if internal_lookback >= swing_lookback:
            raise ValueError("internal_lookback must be < swing_lookback")
        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be > 0")
        if max_obs < 1:
            raise ValueError("max_obs must be >= 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")

        self.int_lb = int(internal_lookback)
        self.swg_lb = int(swing_lookback)
        self.atr_mult = float(atr_multiplier)
        self.max_obs = int(max_obs)
        self.iou_threshold = float(iou_threshold)
        self.missing_fill = float(missing_value_fill)
        self.max_window = (self.swg_lb * 2) + 1

    def get_pivot_flags(self, series: pd.Series, lookback: int, is_high: bool) -> np.ndarray:
        """
        Return detection-time flags.

        If a pivot occurs at origin index j, the flag is emitted at j + lookback,
        which preserves the original engine's confirmation lag behavior.
        """
        if lookback < 1:
            raise ValueError("lookback must be >= 1")

        values = np.asarray(series.to_numpy(dtype=np.float64))
        n = len(values)
        flags = np.zeros(n, dtype=bool)

        if n < (2 * lookback + 1):
            return flags

        for origin_idx in range(lookback, n - lookback):
            candidate = values[origin_idx]
            if not np.isfinite(candidate):
                continue

            window = values[origin_idx - lookback: origin_idx + lookback + 1]
            if not np.isfinite(window).all():
                continue

            if is_high:
                extreme = np.max(window)
                is_strict_pivot = (candidate == extreme) and (np.sum(window == extreme) == 1)
            else:
                extreme = np.min(window)
                is_strict_pivot = (candidate == extreme) and (np.sum(window == extreme) == 1)

            if is_strict_pivot:
                detect_idx = origin_idx + lookback
                if detect_idx < n:
                    flags[detect_idx] = True

        return flags

    @staticmethod
    def iou_1d(bot_a: float, top_a: float, bot_b: float, top_b: float) -> float:
        intersect_top = min(top_a, top_b)
        intersect_bot = max(bot_a, bot_b)
        if intersect_top <= intersect_bot:
            return 0.0
        return (intersect_top - intersect_bot) / (max(top_a, top_b) - min(bot_a, bot_b))

    def is_duplicate_spatial(self, new_ob: dict, queue: deque) -> bool:
        return any(
            self.iou_1d(new_ob["bot"], new_ob["top"], ob["bot"], ob["top"]) > self.iou_threshold
            for ob in queue
        )

    @staticmethod
    def _zone_touched(ob: dict, curr_h: float, curr_l: float) -> bool:
        return curr_l <= ob["top"] and curr_h >= ob["bot"]

    @staticmethod
    def _zone_invalidated(ob: dict, curr_c: float, is_bull: bool) -> bool:
        if is_bull:
            return curr_c < ob["bot"]
        return curr_c > ob["top"]

    def create_ob(
        self,
        origin_idx: int,
        current_idx: int,
        is_high: bool,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atrs: np.ndarray,
    ):
        atr_val = atrs[origin_idx]
        if not np.isfinite(atr_val) or atr_val <= 0.0:
            return None

        atr_val *= self.atr_mult
        base = highs[origin_idx] if is_high else lows[origin_idx]
        if not np.isfinite(base):
            return None

        top, bot = (base, base - atr_val) if is_high else (base + atr_val, base)
        if bot > top:
            top, bot = bot, top

        phantom_closes = closes[origin_idx + 1:current_idx]
        phantom_highs = highs[origin_idx + 1:current_idx]
        phantom_lows = lows[origin_idx + 1:current_idx]

        if phantom_closes.size > 0:
            if is_high and np.any(phantom_closes > top):
                return None
            if (not is_high) and np.any(phantom_closes < bot):
                return None

        touches = (
            int(np.sum((phantom_lows <= top) & (phantom_highs >= bot)))
            if phantom_closes.size > 0
            else 0
        )

        return {
            "idx": origin_idx,
            "top": float(top),
            "bot": float(bot),
            "touches": int(touches),
        }

    def promote_or_create(
        self,
        origin_idx: int,
        current_idx: int,
        is_high: bool,
        internal_q: deque,
        swing_q: deque,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atrs: np.ndarray,
    ) -> None:
        promoted_ob = next((ob for ob in internal_q if ob["idx"] == origin_idx), None)
        if promoted_ob is not None:
            if not self.is_duplicate_spatial(promoted_ob, swing_q):
                internal_q.remove(promoted_ob)
                swing_q.append(promoted_ob)
            return

        new_ob = self.create_ob(origin_idx, current_idx, is_high, highs, lows, closes, atrs)
        if new_ob is not None and not self.is_duplicate_spatial(new_ob, swing_q):
            swing_q.append(new_ob)

    def _update_touch_counts(self, queue: deque, curr_h: float, curr_l: float) -> None:
        for ob in queue:
            if self._zone_touched(ob, curr_h, curr_l):
                ob["touches"] += 1

    def _evict_invalidated(self, queue: deque, curr_c: float, is_bull: bool) -> None:
        to_evict = [ob for ob in queue if self._zone_invalidated(ob, curr_c, is_bull)]
        for ob in to_evict:
            queue.remove(ob)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        n = len(result)

        highs = np.ascontiguousarray(result["high"].to_numpy(), dtype=np.float64)
        lows = np.ascontiguousarray(result["low"].to_numpy(), dtype=np.float64)
        closes = np.ascontiguousarray(result["close"].to_numpy(), dtype=np.float64)
        atrs = np.ascontiguousarray(result["ATR"].to_numpy(), dtype=np.float64)

        int_ph = self.get_pivot_flags(result["high"], self.int_lb, True)
        int_pl = self.get_pivot_flags(result["low"], self.int_lb, False)
        swg_ph = self.get_pivot_flags(result["high"], self.swg_lb, True)
        swg_pl = self.get_pivot_flags(result["low"], self.swg_lb, False)

        swg_bull, swg_bear, int_bull, int_bear = [deque(maxlen=self.max_obs) for _ in range(4)]

        out_swg_supp = np.full(n, np.nan, dtype=np.float64)
        out_swg_res = np.full(n, np.nan, dtype=np.float64)
        out_int_supp = np.full(n, np.nan, dtype=np.float64)
        out_int_res = np.full(n, np.nan, dtype=np.float64)
        out_swg_supp_touches = np.zeros(n, dtype=np.float32)
        out_swg_res_touches = np.zeros(n, dtype=np.float32)
        mask_swg_supp = np.zeros(n, dtype=bool)
        mask_swg_res = np.zeros(n, dtype=bool)

        for i in range(self.max_window, n):
            curr_h = highs[i]
            curr_l = lows[i]
            curr_c = closes[i]

            if not (np.isfinite(curr_h) and np.isfinite(curr_l) and np.isfinite(curr_c)):
                continue

            if swg_ph[i]:
                self.promote_or_create(
                    i - self.swg_lb,
                    i,
                    True,
                    int_bear,
                    swg_bear,
                    highs,
                    lows,
                    closes,
                    atrs,
                )

            if swg_pl[i]:
                self.promote_or_create(
                    i - self.swg_lb,
                    i,
                    False,
                    int_bull,
                    swg_bull,
                    highs,
                    lows,
                    closes,
                    atrs,
                )

            if int_ph[i]:
                new_ob = self.create_ob(i - self.int_lb, i, True, highs, lows, closes, atrs)
                if (
                    new_ob is not None
                    and not self.is_duplicate_spatial(new_ob, swg_bear)
                    and not self.is_duplicate_spatial(new_ob, int_bear)
                ):
                    int_bear.append(new_ob)

            if int_pl[i]:
                new_ob = self.create_ob(i - self.int_lb, i, False, highs, lows, closes, atrs)
                if (
                    new_ob is not None
                    and not self.is_duplicate_spatial(new_ob, swg_bull)
                    and not self.is_duplicate_spatial(new_ob, int_bull)
                ):
                    int_bull.append(new_ob)

            self._update_touch_counts(swg_bull, curr_h, curr_l)
            self._update_touch_counts(int_bull, curr_h, curr_l)
            self._update_touch_counts(swg_bear, curr_h, curr_l)
            self._update_touch_counts(int_bear, curr_h, curr_l)

            self._evict_invalidated(swg_bull, curr_c, is_bull=True)
            self._evict_invalidated(int_bull, curr_c, is_bull=True)
            self._evict_invalidated(swg_bear, curr_c, is_bull=False)
            self._evict_invalidated(int_bear, curr_c, is_bull=False)

            if swg_bull:
                closest = max(swg_bull, key=lambda x: x["top"])
                out_swg_supp[i] = closest["top"]
                out_swg_supp_touches[i] = closest["touches"]
                mask_swg_supp[i] = True

            if swg_bear:
                closest = min(swg_bear, key=lambda x: x["bot"])
                out_swg_res[i] = closest["bot"]
                out_swg_res_touches[i] = closest["touches"]
                mask_swg_res[i] = True

            if int_bull:
                out_int_supp[i] = max(int_bull, key=lambda x: x["top"])["top"]

            if int_bear:
                out_int_res[i] = min(int_bear, key=lambda x: x["bot"])["bot"]

        result["SwingSupportTop"] = out_swg_supp
        result["SwingSupportTouches"] = out_swg_supp_touches
        result["ActiveSwgSupportMask"] = mask_swg_supp.astype(np.int8)

        result["SwingResistanceBot"] = out_swg_res
        result["SwingResistanceTouches"] = out_swg_res_touches
        result["ActiveSwgResistanceMask"] = mask_swg_res.astype(np.int8)

        dist_supp = (result["close"] - result["SwingSupportTop"]) / (result["close"] + EPS)
        dist_res = (result["SwingResistanceBot"] - result["close"]) / (result["close"] + EPS)

        result["DistSwingSuppPct"] = np.where(mask_swg_supp, dist_supp, self.missing_fill)
        result["DistSwingResPct"] = np.where(mask_swg_res, dist_res, self.missing_fill)

        return result


def calculate_features(
    df_raw: pd.DataFrame,
    momentum_period: int = FE_MOMENTUM_PERIOD,
    vol_short_period: int = FE_VOL_SHORT_PERIOD,
    vol_long_period: int = FE_VOL_LONG_PERIOD,
    skew_period: int = FE_SKEW_PERIOD,
    zscore_period: int = FE_ZSCORE_PERIOD,
    icp_period: int = FE_ICP_PERIOD,
    mds_fast_window: int = FE_MDS_FAST_WINDOW,
    mds_slow_window: int = FE_MDS_SLOW_WINDOW,
    vol_asym_window: int = FE_VOL_ASYM_WINDOW,
    stoch_period: int = FE_STOCH_PERIOD,
    adx_period: int = FE_ADX_PERIOD,
    ob_atr_mult: Optional[float] = None,
    bars_per_day: int = FE_BAR_PER_DAY,
    add_session_features: bool = True,
    session: SessionConfig = SessionConfig(),
    clip_outside_session: bool = True,
    dtype: np.dtype = np.float32,
) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    cols = {c.lower() for c in df_raw.columns}
    missing = sorted(list(required - cols))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if momentum_period < 2:
        raise ValueError("momentum_period must be >= 2")
    if vol_short_period < 2 or vol_long_period <= vol_short_period:
        raise ValueError("Require 2 <= vol_short_period < vol_long_period")
    if skew_period < 3:
        raise ValueError("skew_period must be >= 3")
    if zscore_period < 3:
        raise ValueError("zscore_period must be >= 3")
    if icp_period < 2:
        raise ValueError("icp_period must be >= 2")
    if bars_per_day < 1:
        raise ValueError("bars_per_day must be >= 1")
    if stoch_period < 1 or adx_period < 1:
        raise ValueError("stoch_period and adx_period must be >= 1")

    df = df_raw.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close"]].astype(np.float64)

    if (df <= 0.0).any().any():
        raise ValueError("OHLC values must be strictly positive for log-return features.")

    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    log_ret = np.log(c / c.shift(1))

    out = pd.DataFrame(index=df.index)

    net_move = c.diff(momentum_period)
    path_len = c.diff().abs().rolling(momentum_period, min_periods=momentum_period).sum()
    ker_signed = (net_move.abs() / (path_len + EPS)) * np.sign(net_move)
    out["feat_efficiency"] = clip_scale(ker_signed, bound=1.0)

    v_short = log_ret.rolling(vol_short_period, min_periods=vol_short_period).std()
    v_long = log_ret.rolling(vol_long_period, min_periods=vol_long_period).std()
    out["feat_volatility_regime"] = np.log((v_short + EPS) / (v_long + EPS))

    raw_icp = (c - l) / (h - l + EPS)
    scaled_icp = (raw_icp * 2.0) - 1.0
    icp_smooth = scaled_icp.rolling(icp_period, min_periods=icp_period).mean()
    out["feat_icp"] = clip_scale(icp_smooth, bound=1.0)

    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    alpha = 1.0 / float(momentum_period)
    avg_up = up.ewm(alpha=alpha, adjust=False, min_periods=momentum_period).mean()
    avg_down = down.ewm(alpha=alpha, adjust=False, min_periods=momentum_period).mean()

    rs = avg_up / (avg_down + EPS)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_centered = (rsi - 50.0) / 50.0
    out["feat_momentum_rsi"] = clip_scale(rsi_centered, bound=1.0)

    out["feat_dist_skew"] = log_ret.rolling(skew_period, min_periods=skew_period).skew()

    ret_mean = log_ret.rolling(zscore_period, min_periods=zscore_period).mean()
    ret_std = log_ret.rolling(zscore_period, min_periods=zscore_period).std()
    out["feat_zscore"] = (log_ret - ret_mean) / (ret_std + EPS)

    out["feat_momentum_mds"] = momentum_divergence_score(
        log_ret=log_ret,
        fast_window=mds_fast_window,
        slow_window=mds_slow_window,
    )

    out["feat_vol_asymmetry"] = directional_vol_asymmetry(
        log_ret=log_ret,
        window=vol_asym_window,
    )

    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()

    atr_m = ob_atr_mult if ob_atr_mult is not None else OB_ATR_MULT
    ob_engine = OptimizedOrderBlockEngine(
        internal_lookback=OB_INTERNAL_LB,
        swing_lookback=OB_SWING_LB,
        atr_multiplier=atr_m,
        max_obs=OB_MAX_OBS,
        iou_threshold=OB_IOU_THRESHOLD,
        missing_value_fill=FE_MISSING_FILL,
    )
    ob_df = ob_engine.generate_features(df)

    out["feat_ob_dist_supp"] = ob_df["DistSwingSuppPct"]
    out["feat_ob_dist_res"] = ob_df["DistSwingResPct"]
    out["feat_ob_supp_touches"] = np.tanh(ob_df["SwingSupportTouches"] / 3.0)
    out["feat_ob_res_touches"] = np.tanh(ob_df["SwingResistanceTouches"] / 3.0)
    out["feat_ob_supp_active"] = ob_df["ActiveSwgSupportMask"].astype(np.float32)
    out["feat_ob_res_active"] = ob_df["ActiveSwgResistanceMask"].astype(np.float32)

    # FIX-FE-1: Explicit NaN-sentinel guard for order-block distance features.
    # missing_fill=5.0 is a large-value sentinel used when no OB zone is active.
    # Without this, GP can learn rules that fire on *absence* of zones rather than
    # genuine proximity — a semantically incorrect but numerically valid pattern.
    #
    # feat_ob_supp_active and feat_ob_res_active (already in PASSTHROUGH_FEATURES)
    # serve as the correct binary gates. We add an assertion to ensure the contract
    # is enforced and log a warning if sentinel values dominate the feature.

    _supp_sentinel_rate = (out["feat_ob_dist_supp"] >= 4.9).mean()
    _res_sentinel_rate  = (out["feat_ob_dist_res"]  >= 4.9).mean()

    if _supp_sentinel_rate > 0.5:
        import warnings
        warnings.warn(
            f"feat_ob_dist_supp: {_supp_sentinel_rate:.1%} of rows are sentinel "
            f"(missing_fill=5.0). Consider increasing OB lookback or reducing "
            f"swing_lookback to generate more active zones.",
            RuntimeWarning, stacklevel=2,
        )
    if _res_sentinel_rate > 0.5:
        import warnings
        warnings.warn(
            f"feat_ob_dist_res: {_res_sentinel_rate:.1%} of rows are sentinel "
            f"(missing_fill=5.0). GP rules on this feature may be learning "
            f"zone-absence, not zone-proximity.",
            RuntimeWarning, stacklevel=2,
        )

    hl_range = h - l + EPS
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

    out["feat_rejection_upper"] = (upper_wick / hl_range).clip(0.0, 1.0)
    out["feat_rejection_lower"] = (lower_wick / hl_range).clip(0.0, 1.0)

    local_structure_lookback = max(20, int(bars_per_day) * 5)
    rolling_high = h.rolling(
        local_structure_lookback,
        min_periods=max(5, local_structure_lookback // 2),
    ).max()
    rolling_low = l.rolling(
        local_structure_lookback,
        min_periods=max(5, local_structure_lookback // 2),
    ).min()

    out["feat_local_structure"] = (
        ((c - rolling_low) / (rolling_high - rolling_low + EPS)) * 2.0
    ) - 1.0
    out["feat_local_structure"] = clip_scale(out["feat_local_structure"], bound=1.0)

    session_marker = pd.Series(df.index.normalize(), index=df.index)
    is_new_session = session_marker.ne(session_marker.shift(1))
    prev_close = c.shift(1)
    raw_gap = (o - prev_close) / (prev_close + EPS)
    out["feat_session_gap"] = np.where(is_new_session, raw_gap, 0.0)
    if len(out) > 0:
        out.iloc[0, out.columns.get_loc("feat_session_gap")] = 0.0

    atr_fast = tr.rolling(5, min_periods=5).mean()
    atr_slow = tr.rolling(20, min_periods=20).mean()
    out["feat_vol_squeeze"] = atr_fast / (atr_slow + EPS)

    if add_session_features:
        s_sin, s_cos = session_cyclic_position(
            index=out.index,
            session=session,
            clip_outside_session=clip_outside_session,
        )
        out["feat_session_sin"] = s_sin.astype(np.float32)
        out["feat_session_cos"] = s_cos.astype(np.float32)

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.astype(dtype, copy=False)

    missing_contract_cols = [
        c for c in (PASSTHROUGH_FEATURES + SCALE_FEATURES) if c not in out.columns
    ]
    if missing_contract_cols:
        raise ValueError(f"Feature contract columns missing after generation: {missing_contract_cols}")

    duplicate_contract_cols = set(PASSTHROUGH_FEATURES).intersection(SCALE_FEATURES)
    if duplicate_contract_cols:
        raise ValueError(
            f"Feature contract overlap detected between passthrough and scale features: "
            f"{sorted(duplicate_contract_cols)}"
        )

    del df
    del ob_df
    gc.collect()

    return out
