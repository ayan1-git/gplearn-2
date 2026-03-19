"""
regime_classifier.py — Hurst + ADX regime labeler for WFO folds.
Returns: 'trending' | 'mean_reverting' | 'random_walk'
"""
import numpy as np
import pandas as pd

def hurst_exponent(series: pd.Series, max_lag: int = 50) -> float:
    """R/S analysis Hurst exponent. H>0.55=trend, H<0.45=MR, else noise."""
    lags   = range(2, min(max_lag, len(series) // 2))
    tau    = [np.std(series.diff(lag).dropna()) for lag in lags]
    with np.errstate(divide='ignore', invalid='ignore'):
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0]   # slope ≈ Hurst


def adx_value(high: pd.Series, low: pd.Series, close: pd.Series,
              period: int = 14) -> float:
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
                     period: int = 28) -> float:
    """
    Choppiness Index. >61.8 = choppy/range-bound, <38.2 = trending.
    Calculated across the entire series (fold).
    """
    tr   = pd.concat([high - low,
                      (high - close.shift()).abs(),
                      (low  - close.shift()).abs()], axis=1).max(axis=1)
    
    atr_sum = tr.sum()
    val_range = high.max() - low.min()
    
    if val_range == 0:
        return 100.0   # maximum chop if no price movement
        
    chop = 100 * np.log10(atr_sum / val_range) / np.log10(len(close))
    return float(chop)


def classify_regime(df_raw: pd.DataFrame,
                    hurst_trend_thresh: float  = 0.55,
                    hurst_mr_thresh:    float  = 0.45,
                    adx_trend_thresh:   float  = 25.0,
                    chop_trend_thresh:  float  = 38.2,
                    chop_choppy_thresh: float  = 61.8) -> str:
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
