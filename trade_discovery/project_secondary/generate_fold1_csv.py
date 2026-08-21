import pandas as pd
import numpy as np
import importlib
import os
import sys

# Ensure we can import from src (src/ lives in project_core/ after the restructure)
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "project_core")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

fe = importlib.import_module('src.feature_engineering')
tg = importlib.import_module('src.target_generator')

DATAPATH = 'data/NIFTYNEXT50_30min_4Y.csv'
ORACLE_ATR_MULT = 4.0  # From Discovery Run Parameters
ORACLE_MAX_HOLD = 96
OB_ATR_MULT = 0.5      # From src/config.py

# --- 1. Load Data ---
print(f"Loading data from {DATAPATH}...")
df_raw = pd.read_csv(DATAPATH, parse_dates=['datetime'], index_col='datetime')
df_raw.sort_index(inplace=True)
df_raw.columns = [col.lower() for col in df_raw.columns]

# --- 2. Calculate Features ---
print("Calculating features...")
feature_kwargs = dict(
    add_session_features=True,
    session=fe.SessionConfig(open_time='09:15', close_time='15:30', tz='Asia/Kolkata'),
    clip_outside_session=True,
    mds_fast_window=5,
    mds_slow_window=30,
    vol_asym_window=20,
    ob_atr_mult=OB_ATR_MULT,
)

# We need the OB details for raw columns later
c = df_raw['close']
h = df_raw['high']
l = df_raw['low']

# Re-run logic for specific components needed for CSV
tr1 = h - l
tr2 = (h - c.shift(1)).abs()
tr3 = (l - c.shift(1)).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df_raw_with_atr = df_raw.copy()
df_raw_with_atr['ATR'] = tr.ewm(alpha=1.0/14, adjust=False, min_periods=14).mean()

ob_engine = fe.OptimizedOrderBlockEngine(
    internal_lookback=5, swing_lookback=20, atr_multiplier=OB_ATR_MULT, missing_value_fill=5.0
)
ob_df = ob_engine.generate_features(df_raw_with_atr)

df_features = fe.calculate_features(df_raw, **feature_kwargs)

# Align DataFrames
common_index = df_features.index.intersection(ob_df.index)
df_features = df_features.loc[common_index]
df_raw = df_raw.loc[common_index]
ob_df = ob_df.loc[common_index]

# --- 3. Identify Fold 1 Window ---
start_date = df_features.index.min()
train_months = 30
test_months = 6

train_end = start_date + pd.DateOffset(months=train_months)
test_end = train_end + pd.DateOffset(months=test_months)
train_end_inclusive = train_end - pd.Timedelta(nanoseconds=1)

# Purge logic from main_pipeline.py
X_train_raw = df_features.loc[start_date:train_end_inclusive]
if len(X_train_raw) > ORACLE_MAX_HOLD:
    X_train = X_train_raw.iloc[:-ORACLE_MAX_HOLD]
else:
    raise ValueError("Training set too short")

# --- 4. Tanh Scaling Parameters from Fold 1 Train ---
SCALE_FEATURES = fe.SCALE_FEATURES
PASSTHROUGH_FEATURES = fe.PASSTHROUGH_FEATURES

train_scale_view = X_train[SCALE_FEATURES]
EPS_VAL = 1e-8
scale_factors = np.percentile(np.abs(train_scale_view), 75, axis=0)
scale_factors = np.maximum(scale_factors, EPS_VAL)

# Map scale factors for easy access
scale_map = dict(zip(SCALE_FEATURES, scale_factors))

# --- 5. Apply Scaling to Full History ---
X_full_scaled = pd.DataFrame(index=df_features.index)
for col in SCALE_FEATURES:
    X_full_scaled[col] = np.tanh(df_features[col] / scale_map[col]).astype(np.float32)
for col in PASSTHROUGH_FEATURES:
    X_full_scaled[col] = df_features[col].astype(np.float32)

# --- 6. Evaluate GP Signal ---
# Logic: min(if_then(feat_ichimoku_dist_kijun, feat_ob_dist_supp, feat_icp), mul(feat_trend_adx, feat_ob_supp_touches))

def if_then(cond, t, f):
    return np.where(cond > 0.0, t, f)

feat_kijun = X_full_scaled['feat_ichimoku_dist_kijun']
feat_ob_supp = X_full_scaled['feat_ob_dist_supp']
feat_icp = X_full_scaled['feat_icp']
feat_adx = X_full_scaled['feat_trend_adx']
feat_touches = X_full_scaled['feat_ob_supp_touches']

part1 = if_then(feat_kijun, feat_ob_supp, feat_icp)
part2 = feat_adx * feat_touches
gp_signal_raw = np.minimum(part1, part2)

# --- 7. Causal Rolling Rank ---
signals_series = pd.Series(gp_signal_raw, index=df_features.index)
rolling_window = 500
rolling_ranks = signals_series.rolling(window=rolling_window, min_periods=50).rank(pct=True)
expanding_ranks = signals_series.expanding(min_periods=1).rank(pct=True)
causal_ranks = rolling_ranks.fillna(expanding_ranks)

# --- 8. Entries and ATR ---
long_entry = causal_ranks > 0.80
short_entry = causal_ranks < 0.20

# ATR calculation for trailing stop
atr_series = df_raw_with_atr['ATR'].reindex(df_features.index)
atr_pct_series = (atr_series / df_raw['close']) * ORACLE_ATR_MULT
atr_pct_series = atr_pct_series.ffill().fillna(0.01).clip(lower=0.001)

# --- 9. Final DataFrame ---
out_csv = pd.DataFrame(index=df_features.index)
out_csv['close'] = df_raw['close']
out_csv['high'] = df_raw['high']
out_csv['low'] = df_raw['low']
out_csv['feat_ichimoku_dist_kijun_raw'] = df_features['feat_ichimoku_dist_kijun']
out_csv['feat_icp'] = df_features['feat_icp']
out_csv['feat_trend_adx'] = df_features['feat_trend_adx']
out_csv['feat_ob_dist_supp_raw'] = ob_df['DistSwingSuppPct']
out_csv['feat_ob_supp_touches'] = df_features['feat_ob_supp_touches']
out_csv['feat_ob_supp_active'] = df_features['feat_ob_supp_active']
out_csv['gp_signal_raw'] = gp_signal_raw
out_csv['causal_rank'] = causal_ranks
out_csv['long_entry'] = long_entry.astype(int)
out_csv['short_entry'] = short_entry.astype(int)
out_csv['atr_pct_series'] = atr_pct_series

# --- Output CSV ---
output_filename = 'outputs/fold1_validation_data.csv'
out_csv.to_csv(output_filename)
print(f"CSV written to {output_filename}")

# Print window boundaries for reference
print(f"\n--- Boundaries for Fold 1 ---")
print(f"fold1_train_start: {start_date}")
print(f"fold1_train_end: {train_end}")
print(f"fold1_test_end: {test_end}")
