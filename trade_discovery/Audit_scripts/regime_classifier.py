import importlib
import numpy as np
import pandas as pd

try:
    config = importlib.import_module("src.config")
    HURST_TREND_THRESH  = float(config.HURST_TREND_THRESH)
    HURST_MR_THRESH     = float(config.HURST_MR_THRESH)
    ADX_TREND_THRESH    = float(config.ADX_TREND_THRESH)
    CHOP_TREND_THRESH   = float(config.CHOP_TREND_THRESH)
    CHOP_CHOPPY_THRESH  = float(config.CHOP_CHOPPY_THRESH)
    CHOP_PERIOD         = int(config.CHOP_PERIOD)
    ATR_PERIOD          = int(config.ATR_PERIOD)
except (ImportError, AttributeError):
    HURST_TREND_THRESH  = 0.55
    HURST_MR_THRESH     = 0.45
    ADX_TREND_THRESH    = 25.0
    CHOP_TREND_THRESH   = 38.2
    CHOP_CHOPPY_THRESH  = 61.8
    CHOP_PERIOD         = 28
    ATR_PERIOD          = 14

def hurst_exponent(series: pd.Series, max_lag: int = 50) -> float:
    """R/S analysis Hurst exponent. H>0.55=trend, H<0.45=MR, else noise."""
    lags   = range(2, min(max_lag, len(series) // 2))
    tau    = [np.std(series.diff(lag).dropna()) for lag in lags]
    with np.errstate(divide='ignore', invalid='ignore'):
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0]   # slope ≈ Hurst


def adx_value(high: pd.Series, low: pd.Series, close: pd.Series,
              period: int = ATR_PERIOD) -> float:
    """Wilder's ADX. >25 = directional, <20 = non-directional."""
    tr   = pd.concat([high - low,
                      (high - close.shift()).abs(),
                      (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr  = tr.ewm(span=period, min_periods=period).mean()

    up   = (high - high.shift()).clip(lower=0)
    down = (low.shift() - low).clip(lower=0)
    pdi  = 100 * (up.ewm(span=period).mean()  / atr.replace(0, np.nan))
    ndi  = 100 * (down.ewm(span=period).mean() / atr.replace(0, np.nan))

    dx   = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).fillna(0)
    adx  = dx.ewm(span=period).mean()
    return float(adx.iloc[-1])


def choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = CHOP_PERIOD) -> float:
    """
    Standard Choppiness Index using the last `period` bars only.
    >61.8 = choppy/range-bound, <38.2 = trending.
    """
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)

    # KEY FIX: slice to last `period` bars — not full fold
    tr_window  = tr.iloc[-period:]
    hi_window  = high.iloc[-period:]
    lo_window  = low.iloc[-period:]

    atr_sum   = tr_window.sum()
    val_range = hi_window.max() - lo_window.min()

    if val_range == 0 or len(tr_window) < period // 2:
        return 100.0   # max chop if no movement or insufficient data

    chop = 100 * np.log10(atr_sum / val_range) / np.log10(period)
    return float(np.clip(chop, 0.0, 200.0))  # safety clip


def classify_regime(df_raw: pd.DataFrame,
                    hurst_trend_thresh: float  = HURST_TREND_THRESH,
                    hurst_mr_thresh:    float  = HURST_MR_THRESH,
                    adx_trend_thresh:   float  = ADX_TREND_THRESH,
                    chop_trend_thresh:  float  = CHOP_TREND_THRESH,
                    chop_choppy_thresh: float  = CHOP_CHOPPY_THRESH) -> str:
    """
    Extended regime classifier: Hurst + ADX + Choppiness Index.
    
    Returns: 
      - 'trending' (strong confirmation)
      - 'mean_reverting' (low Hurst)
      - 'choppy_random_walk' (mid Hurst, high Chop)
      - 'trending_random_walk' (mid Hurst, low Chop or high ADX)
      - 'random_walk' (uncertain)
    """
    h    = hurst_exponent(df_raw['close'])
    adx  = adx_value(df_raw['high'], df_raw['low'], df_raw['close'])
    chop = choppiness_index(df_raw['high'], df_raw['low'], df_raw['close'])

    if h > hurst_trend_thresh and adx > adx_trend_thresh:
        return 'trending'
    elif h < hurst_mr_thresh:
        return 'mean_reverting'
    
    # Random walk disambiguation
    if chop > chop_choppy_thresh:
        return 'choppy_random_walk'
    elif chop < chop_trend_thresh or adx > adx_trend_thresh:
        return 'trending_random_walk'
    else:
        return 'random_walk'
