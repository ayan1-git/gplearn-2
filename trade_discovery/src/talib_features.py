"""
talib_features.py
=================
Automated TA-Lib feature expansion for OHLC data.
Pre-calculates feature buckets at module load time to ensure compatibility with 
the pipeline's static registration system.
"""
import logging
import numpy as np
import pandas as pd
import src.config as cfg

logger = logging.getLogger(__name__)

_EPS = 1e-8

# ── TA-Lib import guard ───────────────────────────────────────────────────────
try:
    import talib
    _TALIB_AVAILABLE = True
except ImportError:
    _TALIB_AVAILABLE = False
    logger.warning("TA-Lib not installed. Features will be empty.")

# ── Module-level Feature Discovery ───────────────────────────────────────────
TALIB_PASSTHROUGH = []
TALIB_SCALE = []
_FUNC_REGISTRY = [] # List of (func_name, group, is_candle, output_count)

if _TALIB_AVAILABLE:
    _groups = talib.get_function_groups()
    _ignore_groups = ["Volume Indicators", "Math Operators", "Math Transform"]
    
    for _group, _funcs in _groups.items():
        if _group in _ignore_groups:
            continue
            
        for _f_name in _funcs:
            # Candlestick Patterns
            if _group == "Pattern Recognition":
                _name = f"talib_{_f_name.lower()}"
                TALIB_PASSTHROUGH.append(_name)
                _FUNC_REGISTRY.append((_f_name, _group, True, 1))
                continue
            
            # Multi-output function mapping
            _multi = {
                "BBANDS": 3, "MACD": 3, "MACDEXT": 3, "MACDFIX": 3, 
                "STOCH": 2, "STOCHF": 2, "STOCHRSI": 2, "MAMA": 2, "AROON": 2,
                "MINMAX": 2, "PHASOR": 2, "SINE": 2
            }
            _count = _multi.get(_f_name, 1)
            
            # Heuristic for bucket assignment
            _is_osc = any(x in _f_name for x in ["RSI", "MFI", "ADX", "STOCH", "WILLR", "AROON", "ULTOSC", "CCI"])
            
            for _i in range(_count):
                _suffix = f"_{_i}" if _count > 1 else ""
                _name = f"talib_{_f_name.lower()}{_suffix}"
                if _is_osc:
                    TALIB_PASSTHROUGH.append(_name)
                else:
                    TALIB_SCALE.append(_name)
            
            _FUNC_REGISTRY.append((_f_name, _group, False, _count))

def _safe(arr, expected_len: int = None) -> np.ndarray:
    """Cast to float64 and replace inf with nan. Ensures 1D of correct length."""
    if arr is None: return np.array([], dtype=np.float64)
    out = np.array(arr, dtype=np.float64)
    
    # If it's multi-dimensional, try to find a 1D slice that matches expected_len
    if out.ndim > 1:
        if expected_len is not None:
            # Check if any dimension matches
            for axis in range(out.ndim):
                if out.shape[axis] == expected_len:
                    # Take first slice along other dimensions
                    if axis == 0: out = out[0]
                    else: out = out[:, 0] # simplistic
                    break
            else:
                # No match, take first anyway but it might still fail length check
                out = out.reshape(-1)[:expected_len]
        else:
            out = out.flatten()
            
    out[~np.isfinite(out)] = np.nan
    return out

def build_talib_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ALL pre-registered TA-Lib features using OHLC data.
    """
    if not _TALIB_AVAILABLE:
        return pd.DataFrame(index=df_raw.index)

    # Standardize inputs
    o = np.array(df_raw["open"].values, dtype=np.float64)
    h = np.array(df_raw["high"].values, dtype=np.float64)
    l = np.array(df_raw["low"].values, dtype=np.float64)
    c = np.array(df_raw["close"].values, dtype=np.float64)
    idx = df_raw.index
    n_rows = len(c)
    # ATR (price-scale volatility) used to make price-level features dimensionless.
    atr = talib.ATR(h, l, c, timeperiod=int(getattr(cfg, 'ATR_PERIOD', 20))) if _TALIB_AVAILABLE else None

    # Pre-calculate reference scale (approx volatility) for Tanh normalization
    price_std = np.nanstd(c) + _EPS
    
    feats: dict[str, np.ndarray] = {}
    
    for f_name, group, is_candle, count in _FUNC_REGISTRY:
        try:
            func = getattr(talib, f_name)
            
            if is_candle:
                val = _safe(func(o, h, l, c), n_rows)
                if len(val) == n_rows:
                    feats[f"talib_{f_name.lower()}"] = val / 100.0
                continue

            # ── 1. Determine Inputs & Call ──
            outputs = None
            try:
                outputs = func(h, l, c)
            except Exception:
                try:
                    outputs = func(c)
                except Exception:
                    try:
                        outputs = func(h, l)
                    except Exception:
                        continue

            # ── 2. Handle Outputs ──
            if count > 1:
                # Tuple of arrays
                if isinstance(outputs, (list, tuple)):
                    for i in range(count):
                        name = f"talib_{f_name.lower()}_{i}"
                        if i < len(outputs):
                            val = _safe(outputs[i], n_rows)
                            if len(val) == n_rows:
                                feats[name] = _apply_smart_normalization(_talib_kind(group, f_name), val, c, atr)
                # Single 2D array
                elif isinstance(outputs, np.ndarray) and outputs.ndim == 2:
                    for i in range(min(count, outputs.shape[0])):
                        name = f"talib_{f_name.lower()}_{i}"
                        val = _safe(outputs[i], n_rows)
                        if len(val) == n_rows:
                            feats[name] = _apply_smart_normalization(_talib_kind(group, f_name), val, c, atr)
            else:
                # Single output
                val = _safe(outputs, n_rows)
                if len(val) == n_rows:
                    feats[f"talib_{f_name.lower()}"] = _apply_smart_normalization(_talib_kind(group, f_name), val, c, atr)
                        
        except Exception:
            continue

    # Final length check before DataFrame creation
    final_feats = {}
    for k, v in feats.items():
        if len(v) == n_rows:
            final_feats[k] = v
        else:
            logger.debug("Dropping %s: length mismatch (%d vs %d)", k, len(v), n_rows)

    return pd.DataFrame(final_feats, index=idx, dtype=np.float64)

def _zscore(arr: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """Scale-invariant z-score with NaN handling and clipping (never saturates)."""
    arr = np.asarray(arr, dtype=np.float64)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(arr)
    return np.clip((arr - mu) / sd, -clip, clip)


def _talib_kind(group: str, f_name: str) -> str:
    """Map a TA-Lib function to a normalization kind (see _apply_smart_normalization)."""
    if group == "Pattern Recognition":
        return "pattern"
    if group in ("Overlap Studies", "Price Transform"):
        return "price_level"
    _OSC = ["RSI", "MFI", "ADX", "ADXR", "CCI", "CMO", "DX", "MINUS_DI", "PLUS_DI",
            "PPO", "APO", "AROON", "WILLR", "ULTOSC", "STOCH", "BETA", "CORREL"]
    if any(x in f_name for x in _OSC):
        return "oscillator"
    if f_name in ("ROC", "ROCP", "ROCR", "ROCR100"):
        return "return_ratio"
    if f_name == "MOM":
        return "price_level"
    if any(x in f_name for x in ["LINEARREG", "TSF"]):
        return "price_level"
    if any(x in f_name for x in ["STDDEV", "VAR"]):
        return "volatility_raw"
    if f_name in ("NATR", "TRANGE", "ATR"):
        return "volatility_raw"
    return "zscore"  # safe scale-invariant fallback


def _apply_smart_normalization(kind: str, arr: np.ndarray, close: np.ndarray, atr) -> np.ndarray:
    """Kind-aware, scale-invariant normalization.

    Uses dimensionless transforms (z-score, deviation-from-close-in-ATR z-scored,
    log-z) so large-magnitude price/level features can never saturate to a constant ±1.
    """
    arr = np.asarray(arr, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    atr = np.asarray(atr, dtype=np.float64) if atr is not None else np.zeros_like(arr)

    if kind == "pattern":
        return arr  # already scaled by /100 at the call site
    if kind == "mask":
        return arr
    if kind == "oscillator":
        amin = np.nanmin(arr) if np.isfinite(arr).any() else 0.0
        if amin < -50:  # e.g. CCI spans ~[-200, 200]
            return np.clip((arr + 50.0) / 50.0, -1.0, 1.0)
        return np.clip((arr - 50.0) / 50.0, -1.0, 1.0)
    if kind in ("return_ratio", "volatility_raw"):
        return _zscore(arr)
    if kind == "skewed_positive":
        a = np.clip(arr, _EPS, None)
        return _zscore(np.log(a))
    if kind == "price_level":
        # Dimensionless deviation of the level from price, in ATR units, then
        # z-scored so it can never saturate even in strong trends.
        dev = (arr - close) / (atr + _EPS)
        return _zscore(dev, clip=6.0)
    # default: scale-invariant z-score (never saturates, regardless of raw scale)
    return _zscore(arr)