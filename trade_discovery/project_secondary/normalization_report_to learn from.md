# Feature Normalization Report — Lpatchtst

**Date:** 2026-07-04  
**Project:** Lpatchtst (`ayanmarvin124/Lpatchtst`)  
**Scope:** End-to-end normalization of financial time-series features from raw instrument data to model input.

---

## 1. Executive Summary

This project uses a **two-layer normalization strategy**:

1. **Feature-level normalization** (`features.py`): Most features are engineered to be already dimensionless and approximately zero-mean/unit-variance at the source (volatility-scaled returns, 3-step normalized MACD, bounded ratio features).
2. **Global data-level normalization** (`data_loader.py`): Only highly skewed / unbounded features pass through a **column-selective `RobustScaler`** fit on the training split. Pre-normalized features are routed to a **NO_SCALE** identity stream.

A previous approach (per-window z-score inside `__getitem__`) was deliberately **abandoned** after audits proved it destroys 33-67 % of predictive signal by subtracting the window mean, which carries regime-level DC information.

---

## 2. Normalization Layers

| Layer | File | Mechanism | Scope |
|-------|------|-----------|-------|
| Feature engineering | `features.py` | Analytical normalization inside each feature formula | Per-column, per-timestep |
| Global scaler | `data_loader.py` (`ColumnSelectiveScaler`) | `RobustScaler` fit on **train only** | Whole training split per split-type |
| Tokenizer pre-norm | `data_loader.py` (`tokenize_full_series`) | Per-window z-score of raw OHLC windows before VQ encoding | Tokenizer input only |
| Model internal | `model.py`, `tokenizer.py` | `LayerNorm` / `RMSNorm` inside blocks | Per-block activations |

Only Layers 1 and 2 affect the feature vector fed to the model backbone in `features_only` / `combined` mode.

---

## 3. Feature-Level Normalization (`features.py`)

Each feature is designed with a published analytical normalization so that it arrives at the global scaler already on a sensible scale.

### 3.1 Close-Only Features (NO_SCALE bucket)

| Feature | Formula / Logic | Resulting Scale | Bucket |
|---------|----------------|-----------------|--------|
| `ewma_vol_span{N}` | `σ_t = sqrt(EWMA_var(r_t - μ_t))` | ~0.003, tight band | NO_SCALE |
| `ret_norm_{h}d` | `r(t,h) / (σ_t · √h)` | p1/p99 ≈ [-2.5, +2.5] | NO_SCALE |
| `macd_{s}_{l}` | 3-step: raw → `MACD_raw / price_std` → `q / q_std` | std ≈ 1.05, [-3, +3] | NO_SCALE |

**Why NO_SCALE?**
- `ewma_vol`: Already a tiny dimensionless fraction (~0.003). Centering to zero destroys the DC component (σ = 0 is the natural origin).
- `ret_norm_*`: By construction a volatility-adjusted z-score. `RobustScaler` would re-center an already-centered signal.
- `macd_*`: Produced by Eqs. 19–21 in the paper. Empirically std ≈ 1.05. Re-scaling adds noise.

### 3.2 OHLC-Based Features (NO_SCALE or ROBUST)

| Feature | Analytical Normalization | Bucket |
|---------|--------------------------|--------|
| `feat_efficiency` (KER) | `net / path_len`, clipped to [-1, +1] | NO_SCALE |
| `feat_icp` | `(close - low) / (high - low + ε) * 2 - 1` → [-1, +1] | NO_SCALE |
| `feat_momentum_rsi` | `(RSI - 50) / 50` → [-1, +1] | NO_SCALE |
| `feat_vol_asymmetry` | `(up_vol - dn_vol) / (up_vol + dn_vol + ε)` → [-1, +1] | NO_SCALE |
| `feat_local_structure` | `(close - roll_low) / (roll_high - roll_low + ε) * 2 - 1` | NO_SCALE |
| `feat_session_sin` | `sin(2π · pos_in_session)` | NO_SCALE |
| `feat_session_cos` | `cos(2π · pos_in_session)` | NO_SCALE |
| `feat_vol_squeeze` | `ATR_fast / (ATR_slow + ε)`, right-skewed, unbounded | **ROBUST** |
| `vs_factor_span{N}` | `1 / σ_t`, mean ~346, skew ~24, spikes to 3000+ | **ROBUST** |

### 3.3 Nuances in Feature-Level Normalization

- **NaN propagation**: Features compute with `np.errstate(divide='ignore', invalid='ignore')` and guard against zero-division via `_EPS = 1e-10`. NaN gaps are carried forward, never injected as zeros.
- **OHLC validation** (`_validate_ohlc`): Enforces `high >= low`, strictly positive prices, and warns (not errors) when `close` lands outside `[low, high]` due to vendor adjustment artifacts.
- ** `feat_icp` doji handling**: Zero-range bars (`high - low < 1e-6`) are set to neutral `0.5` before scaling.
- **Warm-up handling**: All rolling features respect a `min_periods=period` warm-up. Before warm-up, values are NaN.
- **EWMA seeding**: `_ewm_wilder_seeded` uses a Wilder-style SMA seed over the first `seed_period` valid bars and carries forward across NaN gaps instead of restarting.

---

## 4. Global Data-Level Normalization (`data_loader.py`)

### 4.1 The Two-Bucket Routing System

Normalization routing is **prefix-based** (not a hardcoded frozenset) so it survives `FeatureConfig` span changes.

```python
def _col_bucket(col: str) -> str:
    if col.startswith("ewma_vol_span"):           return "no_scale"
    if col.startswith("ret_norm_"):               return "no_scale"
    if col.startswith("macd_"):                   return "no_scale"
    if col.startswith("vs_factor_span"):          return "robust"
    if col.startswith("feat_session_"):           return "no_scale"
    if col == "feat_vol_squeeze":                 return "robust"
    if col.startswith("feat_"):                   return "no_scale"
    if col.startswith("talib_"):                  return "no_scale"
    return "robust"  # safest default for unknown columns
```

### 4.2 `ColumnSelectiveScaler`

```python
class ColumnSelectiveScaler:
    def __init__(self, feature_cols, clip_bounds=None, default_clip_bound=3.0):
        # Routes each column to NO_SCALE (identity passthrough)
        # or ROBUST (sklearn RobustScaler + per-column IQR clipping)

    def fit(self, X):       # fit ONLY on training split
    def transform(self, X): # apply to any split
    def fit_transform(self, X):
```

**Why `RobustScaler`?**
- Centers by **median** (robust to spikes, not mean).
- Scales by **IQR** (robust to fat tails, not standard deviation).
- Prevents heavily right-skewed columns (`vs_factor`, `feat_vol_squeeze`) from dominating gradients.

### 4.3 Per-Column Clip Bounds (IQR units)

Calibrated via `clip_audit.py` on training data:

```python
ROBUST_CLIP_BOUNDS = {
    "open":               3.0,
    "high":               3.0,
    "low":                3.0,
    "close":              3.0,
    "feat_vol_squeeze":   3.0,
    "vs_factor_span":     2.0,   # prefix match
}
ROBUST_CLIP_BOUND_DEFAULT = 3.0  # ~0.3 % clip rate for any other robust column
```

Inside `transform()`:
```python
for local_j, (global_i, bound) in enumerate(zip(self._robust_idx, self._robust_clip_bounds)):
    col_data  = transformed[:, local_j]
    clip_rate = (np.abs(col_data) > bound).mean()
    if clip_rate > 0.02:
        print(f"[ColumnSelectiveScaler] WARNING: '{col_name}' clip rate {clip_rate:.2%} > 2% ...")
    transformed[:, local_j] = np.clip(col_data, -bound, bound)
```

### 4.4 Key Constraints

- `fit_scaler()` **rejects non-finite input** (NaN / Inf). Caller must strip warmup rows first.
- The scaler is **fit on the training split only**. Passing val/test data is documented as a data-leakage bug.
- `FinancialDataset.__init__` applies the fitted scaler on construction. `__getitem__` returns the stored, already-scaled slice (no per-window transform in `features_only` / `combined`).

---

## 5. Why Per-Window Z-Score Was Rejected

Multiple audit scripts (`double_norm_audit.py`, `scale_analysis_and_fix.py`, `signal_destruction_audit.py`) empirically demonstrated that per-window z-score:

1. **Subtracts the window mean**, destroying the DC component which carries regime signal.
2. **Shrinks near-constant features to noise** (e.g. `feat_icp`, `feat_vol_asymmetry`).
3. Breaks the 1-horizon → N-horizon predictive consistency.

Mutual-information counterfactual experiments showed:
- Per-window z-score destroys **33-67 % of MI**.
- For pre-normalized features, 50 %+ of signal lives in the DC component (window mean), which is subtracted away.

The current `__getitem__` comment is explicit:
```python
# Per-window z-score is NOT applied because it destroys the DC
# component (window mean) which carries the regime signal.
# See: signal_destruction_audit.py, double_norm_audit.py
```

---

## 6. Tokenizer Normalization (`tokenizer.py`, `data_loader.py`)

The `KronosTokenizer` path has its own normalization, completely separate from the feature path:

```python
# Inside tokenize_full_series(), per-window over the raw OHLC input:
w_mean = batch.mean(dim=1, keepdim=True)
w_std  = batch.std(dim=1, keepdim=True) + 1e-5

batch = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
batch = (batch - w_mean) / w_std
batch = torch.clamp(batch, -5.0, 5.0)
```

**Important:** This per-window z-score applies only to the *tokenizer input* (raw OHLC windows). It does **not** apply to the engineered features path. The tokenizer is frozen pre-training, so its internal normalization behavior is fixed.

---

## 7. Model-Internal Normalization

- **`model.py`**: Uses `nn.LayerNorm` (pre-LN) in every encoder and decoder block. `LayerNorm` normalizes over the feature dimension at each timestep.
- **`tokenizer.py`** (Kronos): Uses custom `RMSNorm` in its `TransformerBlock`:
  ```python
  def _norm(self, x):
      return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
  ```
- **`tokenizer.py`** (VQ/BSQ): `F.normalize(z, dim=-1)` is applied *inside* `BSQuantizer.forward()` before quantization, but this is disabled during inference encoding:
  ```python
  # Do NOT apply_normalize here — it collapses variance during inference.
  ```

---

## 8. Data Leakage Prevention

The project has deliberate guard-rails against leakage:

| Splits | Global Scaler | Tokenizer |
|--------|---------------|-----------|
| Train | Fit on `features[:train_end]` | Fit on `features[:te]` (train slice) → transform full series |
| Val | Transform with fitted scaler | Sliced from full-series tokens |
| Test | Transform with fitted scaler | Sliced from full-series tokens |

Calls like `create_multi_index_dataloaders()` collect train chunks first, fit a `__global__` scaler, then apply it to all splits. `test_grad_norm.py` and `pre-train` / `finetune` scripts respect this split.

---

## 9. Config Mapping

`config.py` defines feature-engineering hyperparameters. `train.py::_make_feature_config()` maps them to `FeatureConfig`. The list of columns produced by `FeatureEngineer.build()` drives `ColumnSelectiveScaler` automatically because routing is prefix-based.

Any change to `FE_*` config keys (e.g. `FE_RETURN_HORIZONS`, `FE_MACD_PAIRS`, `ewma_span`) changes:
- Produced column names,
- Scalability bucket (NO_SCALE vs ROBUST),
- Model `input_dim` passed at `_build_model()`.

---

## 10. Summary of Normalization Philosophy

1. **Normalize as close to the source as possible** — features carry their own analytical normalization in `features.py`.
2. **Global, train-only, robust scaling** — `ColumnSelectiveScaler` + `RobustScaler` is the *only* data-level transform applied to the feature matrix.
3. **No per-window z-score on features** — preserves DC (regime) signal.
4. **Audit-driven parameters** — clip bounds calibrated by `clip_audit.py`; distribution checked by `diagnose_distribution.py` and `signal_destruction_audit.py`.
5. **Fail fast on leakage** — explicit error in `fit_scaler()` for non-finite training data.
