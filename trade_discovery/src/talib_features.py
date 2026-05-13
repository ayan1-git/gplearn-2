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
                                feats[name] = _apply_smart_normalization(f_name, val, price_std, group, c)
                # Single 2D array
                elif isinstance(outputs, np.ndarray) and outputs.ndim == 2:
                    for i in range(min(count, outputs.shape[0])):
                        name = f"talib_{f_name.lower()}_{i}"
                        val = _safe(outputs[i], n_rows)
                        if len(val) == n_rows:
                            feats[name] = _apply_smart_normalization(f_name, val, price_std, group, c)
            else:
                # Single output
                val = _safe(outputs, n_rows)
                if len(val) == n_rows:
                    feats[f"talib_{f_name.lower()}"] = _apply_smart_normalization(f_name, val, price_std, group, c)
                        
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

def _apply_smart_normalization(f_name: str, arr: np.ndarray, price_std: float, group: str, close: np.ndarray) -> np.ndarray:
    # Overlap Studies (Price levels) -> Deviation from close
    if group == "Overlap Studies":
        return np.tanh((close - arr) / (arr * 0.01 + _EPS))

    # Bounded Oscillators [0, 100] -> [-1, 1]
    if any(x in f_name for x in ["RSI", "MFI", "ADX", "STOCH", "WILLR", "AROON", "ULTOSC", "CCI"]):
        # Safe check for range
        try:
            amin = np.nanmin(arr)
            if amin < -50: # Likely -100 to 0
                return (arr + 50) / 50.0
            return (arr - 50) / 50.0
        except Exception:
            return arr # Fallback

    if "NATR" in f_name or "ROC" in f_name:
        return np.tanh(arr / 5.0)

    # General Tanh scaling
    return np.tanh(arr / (price_std * 0.1 + _EPS))