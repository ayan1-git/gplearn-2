# Feature Normalization in Trade Discovery Pipeline

## Overview

Normalization occurs at **three distinct levels** in this project and is designed to make raw market data stationary, bounded, and comparable across features with different natural scales. All normalization is hand-rolled — no third-party sklearn scalers are used.

---

## Level 1 — Construction-Time Normalization

Location: `src/feature_engineering.py`, `src/talib_features.py`

### 1.1 Bounded Features (no pipeline scaling)

These features are self-normalized during construction and pass through the pipeline untouched.

#### 1.1.1 Internal Close Position (ICP)
- **File:** `src/feature_engineering.py` lines 657–701
- **Formula:**
  ```
  raw_icp = (close - low) / (high - low + _EPS)          ∈ [0, 1]
  scaled  = raw_icp * 2 - 1                               ∈ [-1, +1]
  icp     = rolling_mean(scaled, period)
  ```
- **Nuances:**
  - Zero-range bars (doji) are detected with `_DOJI_THRESH = 1e-6` and set to `0.5` (→ `0.0` after scaling).
  - A safety `_EPS = 1e-10` guards against remaining edge cases.
  - Result is final-clipped to `[-1, +1]`.

#### 1.1.2 Kaufman Efficiency Ratio (KER)
- **File:** `src/feature_engineering.py` lines 623–649
- **Formula:**
  ```
  net_move = close_t - close_{t-period}
  path_len = Σ|close_i - close_{i-1}|  over period bars
  ker_raw  = |net_move| / (path_len + _EPS)
  ker      = ker_raw * sign(net_move)                    → clip to [-1, +1]
  ```
- **Range:** `[-1, +1]`. Perfect trend = ±1, random walk = 0.

#### 1.1.3 Centered RSI
- **File:** `src/feature_engineering.py` lines 708–754
- **Formula:**
  ```
  RSI_std = 100 - 100/(1 + RS)                          ∈ [0, 100]
  output  = (RSI_std - 50) / 50                          ∈ [-1, +1]
  ```
- Uses Wilder-smoothed EWMA for `avg_gain` and `avg_loss`.

#### 1.1.4 Directional Volatility Asymmetry
- **File:** `src/feature_engineering.py` lines 761–802
- **Formula:**
  ```
  up_vol = rolling_std_of(upside_moves)
  dn_vol = rolling_std_of(downside_moves)
  output = (up_vol - dn_vol) / (up_vol + dn_vol + _EPS)
  ```
- Final result clipped to `[-1, +1]`.

#### 1.1.5 Local Structure Position
- **File:** `src/feature_engineering.py` lines 809–842
- **Formula:**
  ```
  raw = (close - roll_low) / (roll_high - roll_low + _EPS) * 2 - 1
  ```
- After a rolling window (Donchian-style).
- Clipped to `[-1, +1]`.

#### 1.1.6 Session Cyclic Features
- **File:** `src/feature_engineering.py` lines 859–933
- **Formula:**
  ```
  angle   = (session_position / session_length) * 2 * π
  sin_enc = sin(angle)
  cos_enc = cos(angle)
  ```
- **Range:** `[-1, +1]` naturally.

#### 1.1.7 Price Rejection Features
- **File:** `src/feature_engineering.py` lines 980–1006
- **Formula:**
  ```
  upper = (high - max(open, close)) / (range + _EPS)
  lower = (max(open, close) - low) / (range + _EPS)
  ```
- Clipped `upper → [0, 1]`, `lower → [0, 1]`.

#### 1.1.8 TA-Lib Smart Normalization
- **File:** `src/talib_features.py` lines 169–189
- Applied to every TA-Lib indicator output before storage.

| Ta-Lib Group | Rule |
|---|---|
| `Overlap Studies` | `tanh((close - arr) / (arr * 0.01 + _EPS))` |
| Oscillators (RSI, MFI, ADX, STOCH, WILLR, AROON, ULTOSC, CCI) | `(arr - 50) / 50` or `(arr + 50) / 50` if range is `[-100, 0]` |
| NATR, ROC | `tanh(arr / 5.0)` |
| Everything else | `tanh(arr / (price_std * 0.1 + _EPS))` |

- **Oscillator branch nuance:** Uses `np.nanmin(arr)` to detect whether the range is `[-100, 0]` (uses `+50`) or `[0, 100]` (uses `-50`), gracefully falls back to raw array on exception.

### 1.2 Volatility-Normalized Features (no pipeline scaling)

#### 1.2.1 Normalized Returns
- **File:** `src/feature_engineering.py` lines 463–485
- **Formula:**
  ```
  r_norm(t, h) = r(t, h) / (σ_t · √h)
  ```
- Renders returns dimensionless and horizon-invariant. Concentrated in `[-2, +2]`.
- NaN divisions are suppressed via `np.errstate(invalid="ignore")` and replaced with `np.nan` where denominator < `_EPS`.

#### 1.2.2 Multi-Scale MACD Signal
- **File:** `src/feature_engineering.py` lines 492–537
- Three-step normalization:
  1. `macd_raw = EWMA_short(price) - EWMA_long(price)`
  2. `q = macd_raw / rolling_std(price, window=63)`
  3. `signal = q / rolling_std(q, window=252)`
- Result is approximately unit variance. Concentrated in `[-4, +4]`.
- NaN-safe: denominator zero replaced with `np.nan`.

#### 1.2.3 Volatility Scaling Factor
- **File:** `src/feature_engineering.py` lines 566–575
- **Formula:** `1.0 / σ_t`
- Passed through as-is (ROBUST bucket — no additional scaling). This feature is naturally large when volatility is low and vice versa.

### 1.3 Unbounded Robust Features (sandwiched into pipeline SCALE_FEATURES)

#### 1.3.1 Ichimoku Distances
- **File:** `src/feature_engineering.py` lines 1009–1052
- **Formula:** `(close - component) / (close + _EPS)`
- Unbounded, percentage-style distances. These go into `SCALE_FEATURES`.

#### 1.3.2 Optimized Order Block Distances
- **File:** `src/feature_engineering.py` lines 1235–1360 (via `OptimizedOrderBlockEngine`)
- **Formula:**
  ```
  dist_supp = (close - supp_level) / close
  dist_res  = (res_level - close) / close
  ```
- Unbounded. These are also `SCALE_FEATURES`.

---

## Level 2 — Pipeline-Time Tanh Scaling

Location: `scripts/main_pipeline.py` lines 101–133

```python
def tanh_scale_train_apply_test(
    X_train, X_test, scale_cols, passthrough_cols,
):
    scale_factors = np.maximum(
        np.percentile(np.abs(train_view), 75, axis=0), 1e-8
    )
    scaled_train = np.tanh(train_view / scale_factors).astype(np.float32)
    scaled_test  = np.tanh(X_test[scale_cols]  / scale_factors).astype(np.float32)
```

### Mathematical Properties

```
scaled_value = tanh(value / P75(|train_feature|))
```

| Property | Detail |
|---|---|
| **Scale factor** | 75th percentile of absolute training values, floored at `1e-8` |
| **Output range** | `(-1, +1)` (open interval — never exactly ±1) |
| **Zero preservation** | `tanh(0) = 0` — neutral inputs stay neutral |
| **Outlier robustness** | Asymptotes at ±1; extreme values don't distort the scale |
| **Linearity near zero** | For `value << scale_factor`: `tanh(x) ≈ x` |

### Why 75th percentile?

- More outlier-resistant than mean, median, or max.
- Ensures ~75% of training data is in the "linear" regime (`tanh ≈ identity`), while the top 25% gets squashed.

### Bucket Assignment

**PASSTHROUGH_FEATURES** (already bounded, no scaling):
```
ewma_vol_span{span}, ret_norm_{h}d, macd_{short}_{long},
feat_efficiency, feat_icp, feat_momentum_rsi, feat_vol_asymmetry,
feat_local_structure, feat_session_sin, feat_session_cos,
feat_ob_supp_touches, feat_ob_res_touches,
feat_rejection_upper, feat_rejection_lower,
feat_ob_supp_mask, feat_ob_res_mask,
feat_ob_supp_level, feat_ob_res_level
# + TA-Lib passthrough variants (if Talib available)
```

**SCALE_FEATURES** (unbounded, receive tanh treatment):
```
vs_factor_span{span}, feat_vol_squeeze,
feat_ichimoku_dist_tenkan, feat_ichimoku_dist_kijun,
feat_ichimoku_dist_span_a, feat_ichimoku_dist_span_b,
feat_ob_supp_dist, feat_ob_res_dist
# + TA-Lib scale variants (if Talib available)
```

---

## Level 3 — Score-Level Rank Normalization

Location: `src/vectorbt_evaluator.py` lines 117–123

This occurs **after** the GP model produces raw scores on OOS features that were already construction-time + pipeline normalized.

```python
rolling_ranks   = scores.rolling(window=500, min_periods=50).rank(pct=True)
expanding_ranks = scores.expanding(min_periods=1).rank(pct=True)
causal_ranks    = rolling_ranks.fillna(expanding_ranks)

long_entries  = causal_ranks >= (ENTRY_PCT / 100.0)   # 85th percentile
short_entries = causal_ranks <= (EXIT_PCT / 100.0)    # 15th percentile
```

**Nuances:**
- **Rolling first, expanding fallback:** During the warm-up period (`min_periods=50`), expanding ranks are used.
- **Causal only:** Uses `.rolling()` and `.expanding()` (Wikipedia "causal" mode) — no look-ahead in percentile rank estimation.
- **Absolute edge floor:** Entries also require `|score| >= absolute_edge_floor` to avoid noise peaks.
- **Result range:** `[0, 1]`. Thresholds convert to binary trading signals.

---

## Level 4 — GP Internal Normalization

Location: `src/gp_engine.py` lines 125–141

### 4.1 Soft Comparison Operators
GP models learn boolean-like logic but with soft tanh gates:

```python
GT_SOFT_SCALE = 3.0

def _gt_soft(x1, x2):  return np.tanh((x1 - x2) * GT_SOFT_SCALE)
def _lt_soft(x1, x2):  return np.tanh((x2 - x1) * GT_SOFT_SCALE)
def _and(x1, x2):      return np.minimum(x1, x2)
def _or(x1, x2):       return np.maximum(x1, x2)
def _if_then(c, t, f): return np.where(c > 0.0, t, f)
```

| Operator | Range | Notes |
|---|---|---|
| `gt_soft(x1, x2)` | `(-1, +1)` | Returns positive when `x1 > x2` |
| `lt_soft(x1, x2)` | `(-1, +1)` | Returns positive when `x2 > x1` |
| `and` | `(-∞, min(a,b))` | Element-wise min |
| `or` | `(-∞, max(a,b))` | Element-wise max |
| `if_then` | `∈{-∞, +∞}` | Branching, not differentiable |

### 4.2 Protected Division
```python
def _protected_div(x1, x2):
    return np.where(np.abs(x2) > 0.001, x1 / x2, np.ones_like(x1))
```

Denominator floor at `0.001`. Returns `1.0` (neutral element) when denominator is near-zero.

### 4.3 Mutation/Crossover Probability Normalization
```python
fixed_p   = HOIST_MUTATION + POINT_MUTATION
total_p   = hp["p_crossover"] + subtree_mut + fixed_p
if total_p > 0.9999:
    scale = (0.998 - fixed_p) / (hp["p_crossover"] + subtree_mut)
    hp["p_crossover"] *= scale
    subtree_mut       *= scale
```

Ensures GP evolutionary probabilities sum to **≤ 0.998** (reserving 0.002 headroom for hoist + point mutations).

---

## Data Leakage Prevention

Every normalization step respects causal ordering:

| Stage | Where Scale Is Fit | Where Applied |
|---|---|---|
| Tanh scaling | `X_train` only (P75 of abs values) | `X_train` and `X_hold` identically |
| Rank normalization | Expanding window up to `t` | Window `[t-500, t]` |
| MACD/return norm | Rolling window ending at `t` | At `t` only |

### Embargo During Train/Holdout Split
- `ORACLE_MAX_HOLD` bars are removed from the end of `X_train` before tanh scaling.
- Prevents TBM (Tension Breakout Model) forward-looking multi-bar targets from leaking.

---

## Summary Table

| Layer | Location | Method | Range | Fit |
|---|---|---|---|---|
| Construction | feature_engineering.py | clip, ratio, division by σ | varies | Per-window |
| Construction | talib_features.py | smart normalization | varies | Per-indicator |
| Pipeline     | main_pipeline.py    | `tanh(x / P75(|train|))` | `(-1, +1)` | Train only |
| Signal Gen.  | vectorbt_evaluator.py | rolling `rank(pct=True)` | `[0, 1]` | Expanding causal |
| GP Operator  | gp_engine.py | `tanh((x1-x2)*3.0)` | `(-1, +1)` | N/A |
| GP Evolution | gp_engine.py | probability mass renormalization | `≤0.998` | N/A |
| Target       | feature_engineering.py | `clip(r_{t+1}/σ, ±20)` | `[-20, +20]` | Train only (inference error) |
