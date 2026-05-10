"""
talib_features.py
=================
TA-Lib feature expansion for NIFTY50 30-min OHLCV-free pipeline.
All features are OHLC-only (no volume), normalized to [-1,+1] or [0,1].

Feature buckets:
  TALIB_PASSTHROUGH — already bounded, no scaling needed
  TALIB_SCALE       — unbounded/skewed, needs tanh scaling in pipeline
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-8

# ── TA-Lib import guard ───────────────────────────────────────────────────────
try:
    import talib
    _TALIB_AVAILABLE = True
except ImportError:
    _TALIB_AVAILABLE = False
    logger.warning(
        "TA-Lib not installed. talib_features will return empty DataFrame. "
        "Install with: pip install TA-Lib"
    )


def _safe(arr: np.ndarray) -> np.ndarray:
    """Cast to float64 and replace inf with nan."""
    out = np.array(arr, dtype=np.float64)
    out[~np.isfinite(out)] = np.nan
    return out


def build_talib_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TA-Lib feature expansion from OHLC data (no volume required).

    Parameters
    ----------
    df_raw : pd.DataFrame
        Must have lowercase columns: open, high, low, close.
        Index must be a DatetimeIndex (same as pipeline convention).

    Returns
    -------
    pd.DataFrame
        All float64. NaN rows where warmup is insufficient.
        Columns defined in TALIB_PASSTHROUGH + TALIB_SCALE lists.
    """
    if not _TALIB_AVAILABLE:
        return pd.DataFrame(index=df_raw.index)

    o = _safe(df_raw["open"].values)
    h = _safe(df_raw["high"].values)
    l = _safe(df_raw["low"].values)
    c = _safe(df_raw["close"].values)
    idx = df_raw.index

    feats: dict[str, np.ndarray] = {}

    # ── MOMENTUM ─────────────────────────────────────────────────────────────

    # RSI (14) — centered [-1, +1]
    # Note: your existing feat_momentum_rsi uses Wilder EWM from scratch.
    # This RSI uses talib's C implementation at different periods for diversity.
    feats["talib_rsi_7"]  = (_safe(talib.RSI(c, 7))  - 50) / 50
    feats["talib_rsi_21"] = (_safe(talib.RSI(c, 21)) - 50) / 50

    # Stochastic %D (smoothed) — centered [-1, +1]
    _, stoch_d = talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowd_period=3)
    feats["talib_stoch_d"] = (_safe(stoch_d) - 50) / 50

    # Stochastic RSI — centered [-1, +1]
    fastk, fastd = talib.STOCHRSI(c, timeperiod=14, fastk_period=5, fastd_period=3)
    feats["talib_stochrsi_k"] = (_safe(fastk) - 50) / 50
    feats["talib_stochrsi_d"] = (_safe(fastd) - 50) / 50

    # Williams %R — centered [-1, +1]
    # talib returns [-100, 0]; transform to [-1, +1]: val/50 + 1
    feats["talib_willr_14"] = _safe(talib.WILLR(h, l, c, 14)) / 50 + 1.0

    # CCI (20) — tanh-normalize (unbounded) → SCALE bucket
    feats["talib_cci_20"] = np.tanh(_safe(talib.CCI(h, l, c, 20)) / 100.0)

    # Rate of Change — tanh-normalize (unbounded) → SCALE bucket
    feats["talib_roc_5"]  = np.tanh(_safe(talib.ROC(c,  5)) / 3.0)
    feats["talib_roc_13"] = np.tanh(_safe(talib.ROC(c, 13)) / 5.0)
    feats["talib_roc_26"] = np.tanh(_safe(talib.ROC(c, 26)) / 8.0)

    # Momentum (raw close diff) — tanh-normalize → SCALE bucket
    # Different from ROC: absolute price change not percentage
    feats["talib_mom_10"] = np.tanh(_safe(talib.MOM(c, 10)) / (np.nanstd(c) * 0.1 + _EPS))

    # ── TREND ─────────────────────────────────────────────────────────────────

    # ADX — trend strength [0, 100] → normalize to [0, 1]
    feats["talib_adx_14"] = _safe(talib.ADX(h, l, c, 14)) / 100.0

    # DI+ minus DI- — direction × strength [-1, +1]
    diplus  = _safe(talib.PLUS_DI(h, l, c, 14))
    diminus = _safe(talib.MINUS_DI(h, l, c, 14))
    feats["talib_di_diff"] = np.tanh((diplus - diminus) / 50.0)

    # Aroon oscillator — already [-100, +100] → normalize to [-1, +1]
    feats["talib_aroon_osc_25"] = _safe(talib.AROONOSC(h, l, timeperiod=25)) / 100.0

    # MACD histogram — tanh-normalize → SCALE bucket
    # Different spans from your existing MACDs (8/24, 26/78, 52/156)
    _, _, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    price_scale = np.nanstd(c) * 0.01 + _EPS
    feats["talib_macd_hist"] = np.tanh(_safe(macd_hist) / price_scale)

    # TEMA (Triple EMA) vs close — normalized relative deviation → SCALE bucket
    tema_20 = _safe(talib.TEMA(c, 20))
    feats["talib_tema_dev"] = np.tanh((c - tema_20) / (tema_20 * 0.01 + _EPS))

    # ── VOLATILITY ────────────────────────────────────────────────────────────

    # ATR ratio at multiple scales (fast/slow) — SCALE bucket
    atr_5  = _safe(talib.ATR(h, l, c,  5))
    atr_14 = _safe(talib.ATR(h, l, c, 14))
    atr_28 = _safe(talib.ATR(h, l, c, 28))
    feats["talib_atr_5_14"]  = np.log(atr_5  / (atr_14 + _EPS) + _EPS)   # log ratio
    feats["talib_atr_14_28"] = np.log(atr_14 / (atr_28 + _EPS) + _EPS)

    # Normalized ATR (ATR / close) — relative bar range → SCALE bucket
    feats["talib_natr_14"] = np.tanh(_safe(talib.NATR(h, l, c, 14)) / 2.0)

    # Bollinger Band %B — position within bands [-1, +1]
    # %B = (price - lower) / (upper - lower) → mapped to [-1, +1]
    bb_upper, bb_mid, bb_lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
    bb_upper, bb_mid, bb_lower = _safe(bb_upper), _safe(bb_mid), _safe(bb_lower)
    bb_range = bb_upper - bb_lower
    bb_pct_b = (c - bb_lower) / (bb_range + _EPS)  # [0, 1] approx
    feats["talib_bb_pctb"]  = (bb_pct_b * 2.0 - 1.0).clip(-2.0, 2.0)   # passthrough
    feats["talib_bb_width"] = np.tanh(bb_range / (bb_mid + _EPS) / 0.02) # SCALE

    # ── STRUCTURE / PRICE POSITION ────────────────────────────────────────────

    # DPO (De-trended Price Oscillator) — tanh → SCALE bucket
    feats["talib_dpo_20"] = np.tanh(_safe(talib.DX(h, l, c, 20)) / 25.0 - 1.0)

    # Midpoint price position within rolling range — passthrough
    midpoint = _safe(talib.MIDPRICE(h, l, 14))
    feats["talib_midprice_dev"] = np.tanh((c - midpoint) / (atr_14 + _EPS))

    # Highest high / lowest low distance — normalized [-1, +1]
    hh_26 = _safe(talib.MAX(h, 26))
    ll_26 = _safe(talib.MIN(l, 26))
    rng_26 = hh_26 - ll_26
    feats["talib_price_pos_26"] = ((c - ll_26) / (rng_26 + _EPS) * 2.0 - 1.0).clip(-1.0, 1.0)

    hh_65 = _safe(talib.MAX(h, 65))
    ll_65 = _safe(talib.MIN(l, 65))
    rng_65 = hh_65 - ll_65
    feats["talib_price_pos_65"] = ((c - ll_65) / (rng_65 + _EPS) * 2.0 - 1.0).clip(-1.0, 1.0)

    # ── CANDLESTICK PATTERNS (binary signals) ─────────────────────────────────
    # TA-Lib returns -100, 0, +100 → divide by 100 → {-1, 0, +1}
    # These are passthrough (already bounded, discrete)
    candle_funcs = {
        "talib_cdl_doji":        talib.CDLDOJI,
        "talib_cdl_hammer":      talib.CDLHAMMER,
        "talib_cdl_invhammer":   talib.CDLINVERTEDHAMMER,
        "talib_cdl_engulf":      talib.CDLENGULFING,
        "talib_cdl_harami":      talib.CDLHARAMI,
        "talib_cdl_morningstar": talib.CDLMORNINGSTAR,
        "talib_cdl_eveningstar": talib.CDLEVENINGSTAR,
        "talib_cdl_3whitesol":   talib.CDL3WHITESOLDIERS,
        "talib_cdl_3blackcrows": talib.CDL3BLACKCROWS,
        "talib_cdl_shootingstar":talib.CDLSHOOTINGSTAR,
    }
    for name, func in candle_funcs.items():
        feats[name] = _safe(func(o, h, l, c)) / 100.0

    # ── ASSEMBLE ──────────────────────────────────────────────────────────────
    result = pd.DataFrame(feats, index=idx, dtype=np.float64)

    nan_counts = result.isna().sum()
    logger.info(
        "talib_features built: shape=%s | NaN counts (top 5):\n%s",
        result.shape,
        nan_counts[nan_counts > 0].sort_values(ascending=False).head(5)
    )
    return result


# ── Feature bucket lists (for tanh-scaling decisions in main_pipeline) ────────

TALIB_PASSTHROUGH = [
    "talib_rsi_7", "talib_rsi_21",
    "talib_stoch_d", "talib_stochrsi_k", "talib_stochrsi_d",
    "talib_willr_14",
    "talib_adx_14",
    "talib_aroon_osc_25",
    "talib_di_diff",
    "talib_bb_pctb",
    "talib_price_pos_26", "talib_price_pos_65",
    "talib_cdl_doji", "talib_cdl_hammer", "talib_cdl_invhammer",
    "talib_cdl_engulf", "talib_cdl_harami", "talib_cdl_morningstar",
    "talib_cdl_eveningstar", "talib_cdl_3whitesol", "talib_cdl_3blackcrows",
    "talib_cdl_shootingstar",
]

TALIB_SCALE = [
    "talib_cci_20",
    "talib_roc_5", "talib_roc_13", "talib_roc_26",
    "talib_mom_10",
    "talib_macd_hist",
    "talib_tema_dev",
    "talib_atr_5_14", "talib_atr_14_28",
    "talib_natr_14",
    "talib_bb_width",
    "talib_dpo_20",
    "talib_midprice_dev",
]