"""
talib_features.py
=================
Automated TA-Lib feature expansion for OHLC data.
Pre-calculates feature buckets at module load to ensure compatibility with 
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
            
            # Others: determine output count via a dummy call or meta-analysis
            # For simplicity, we use a known list of multi-output functions
            _multi = {
                "BBANDS": 3, "MACD": 3, "MACDEXT": 3, "MACDFIX": 3, 
                "STOCH": 2, "STOCHF": 2, "STOCHRSI": 2, "MAMA": 2, "AROON": 2
            }
            _count = _multi.get(_f_name, 1)
            
            # Heuristic for bucket assignment
            # Oscillators -> Passthrough (we will normalize to [-1, 1])
            # Price/Volatility -> Scale (we will use Tanh)
            _is_osc = any(x in _f_name for x in ["RSI", "MFI", "ADX", "STOCH", "WILLR", "AROON", "ULTOSC", "CCI"])
            
            for _i in range(_count):
                _suffix = f"_{_i}" if _count > 1 else ""
                _name = f"talib_{_f_name.lower()}{_suffix}"
                if _is_osc:
                    TALIB_PASSTHROUGH.append(_name)
                else:
                    TALIB_SCALE.append(_name)
            
            _FUNC_REGISTRY.append((_f_name, _group, False, _count))

def _safe(arr: np.ndarray) -> np.ndarray:
    """Cast to float64 and replace inf with nan."""
    out = np.array(arr, dtype=np.float64)
    out[~np.isfinite(out)] = np.nan
    return out

def build_talib_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ALL pre-registered TA-Lib features using OHLC data.
    """
    if not _TALIB_AVAILABLE:
        return pd.DataFrame(index=df_raw.index)

    o = _safe(df_raw["open"].values)
    h = _safe(df_raw["high"].values)
    l = _safe(df_raw["low"].values)
    c = _safe(df_raw["close"].values)
    idx = df_raw.index

    # Pre-calculate reference scale (approx volatility) for Tanh normalization
    price_std = np.nanstd(c) + _EPS
    
    feats: dict[str, np.ndarray] = {}
    
    for f_name, group, is_candle, count in _FUNC_REGISTRY:
        try:
            func = getattr(talib, f_name)
            
            if is_candle:
                outputs = func(o, h, l, c)
                feats[f"talib_{f_name.lower()}"] = _safe(outputs) / 100.0
                continue

            # Overlap Studies (Price levels)
            if group == "Overlap Studies":
                if f_name in ["SAR", "SAREXT"]:
                    val = _safe(func(h, l))
                    feats[f"talib_{f_name.lower()}"] = np.tanh((c - val) / (val * 0.01 + _EPS))
                elif count > 1:
                    res = func(c)
                    for i, arr in enumerate(res):
                        val = _safe(arr)
                        feats[f"talib_{f_name.lower()}_{i}"] = np.tanh((c - val) / (val * 0.01 + _EPS))
                else:
                    val = _safe(func(c))
                    feats[f"talib_{f_name.lower()}"] = np.tanh((c - val) / (val * 0.01 + _EPS))
                continue

            # General Indicators
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

            if count > 1 and isinstance(outputs, tuple):
                for i, out_arr in enumerate(outputs):
                    name = f"talib_{f_name.lower()}_{i}"
                    feats[name] = _apply_smart_normalization(f_name, _safe(out_arr), price_std)
            else:
                name = f"talib_{f_name.lower()}"
                feats[name] = _apply_smart_normalization(f_name, _safe(outputs), price_std)
                        
        except Exception:
            continue

    return pd.DataFrame(feats, index=idx, dtype=np.float64)

def _apply_smart_normalization(f_name: str, arr: np.ndarray, price_std: float) -> np.ndarray:
    # Bounded Oscillators [0, 100] -> [-1, 1]
    if any(x in f_name for x in ["RSI", "MFI", "ADX", "STOCH", "WILLR", "AROON", "ULTOSC", "CCI"]):
        if np.nanmin(arr) < -50: 
            return (arr + 50) / 50.0
        return (arr - 50) / 50.0

    if "NATR" in f_name or "ROC" in f_name:
        return np.tanh(arr / 5.0)

    # General Tanh scaling
    return np.tanh(arr / (price_std * 0.1 + _EPS))