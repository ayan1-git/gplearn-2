#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import src.config as cfg
from src.feature_engineering import calculate_features, SessionConfig
from src.target_generator import generate_tbm_targets

# Point the audit at the SAME data file the pipeline trains on (src/config.DATAPATH).
# This guarantees the audit checks the exact targets the model is trained on.
cfg.DATAPATH = getattr(cfg, 'DATAPATH', 'data/NIFTY MID SELECT_15minute.csv')
_resolved_datapath = cfg.DATAPATH if os.path.isabs(cfg.DATAPATH) else os.path.join(REPO_ROOT, cfg.DATAPATH)
cfg.DATA_DIR = os.path.dirname(_resolved_datapath) or 'data'
cfg.DATA_FILE = [_resolved_datapath]


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_config(data_dir_override=None):
    data_dir = _resolve_path(data_dir_override) if data_dir_override else cfg.DATA_DIR
    cfg.DATA_DIR = data_dir
    cfg.DATA_FILE = [_resolve_path(p) for p in cfg.DATA_FILE]
    return cfg


def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in df.columns:
        cl = c.lower().strip()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def load_ohlc_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = find_col(df, ['datetime', 'date', 'time', 'timestamp'])
    open_col = find_col(df, ['open'])
    high_col = find_col(df, ['high'])
    low_col = find_col(df, ['low'])
    close_col = find_col(df, ['close'])
    vol_col = find_col(df, ['volume', 'vol'])

    missing = [name for name, col in [('open', open_col), ('high', high_col), ('low', low_col), ('close', close_col)] if col is None]
    if missing:
        raise ValueError(f'{path}: missing required columns {missing}; columns={list(df.columns)}')

    out = pd.DataFrame()
    if ts_col is not None:
        out['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce')
    else:
        out['timestamp'] = pd.RangeIndex(len(df))
    out['open'] = pd.to_numeric(df[open_col], errors='coerce')
    out['high'] = pd.to_numeric(df[high_col], errors='coerce')
    out['low'] = pd.to_numeric(df[low_col], errors='coerce')
    out['close'] = pd.to_numeric(df[close_col], errors='coerce')
    if vol_col is not None:
        out['volume'] = pd.to_numeric(df[vol_col], errors='coerce')
    return out


def basic_ohlc_checks(df: pd.DataFrame) -> Dict:
    out = {}
    out['rows'] = int(len(df))
    out['null_counts'] = {c: int(df[c].isna().sum()) for c in df.columns}
    out['duplicated_rows'] = int(df.duplicated().sum())
    out['duplicated_timestamps'] = int(df['timestamp'].duplicated().sum()) if 'timestamp' in df.columns else None
    bad = {}
    bad['high_lt_low'] = int((df['high'] < df['low']).sum())
    bad['open_outside_hl'] = int(((df['open'] > df['high']) | (df['open'] < df['low'])).sum())
    bad['close_outside_hl'] = int(((df['close'] > df['high']) | (df['close'] < df['low'])).sum())
    bad['nonpositive_prices'] = int(((df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)).sum())
    out['ohlc_violations'] = bad
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        d = df['timestamp'].diff().dropna()
        out['non_monotonic_timestamps'] = int((d <= pd.Timedelta(0)).sum())
        vc = d.value_counts().head(5)
        out['top_time_deltas'] = {str(k): int(v) for k, v in vc.items()}
    else:
        out['non_monotonic_timestamps'] = None
        out['top_time_deltas'] = {}
    return out


def target_distribution(targets: np.ndarray, sampler_threshold: float) -> Dict:
    a = np.abs(targets)
    exact_zero = targets == 0.0
    eps_zero = np.isclose(targets, 0.0, atol=1e-12)
    out = {
        'n': int(len(targets)),
        'mean': float(np.nanmean(targets)),
        'std': float(np.nanstd(targets)),
        'min': float(np.nanmin(targets)),
        'max': float(np.nanmax(targets)),
        'exact_zero_count': int(exact_zero.sum()),
        'exact_zero_pct': float(exact_zero.mean()),
        'eps_zero_count': int(eps_zero.sum()),
        'eps_zero_pct': float(eps_zero.mean()),
        'tiny_abs_lt_1e8_count': int((a < 1e-8).sum()),
        'tiny_abs_lt_1e6_count': int((a < 1e-6).sum()),
        'abs_0_to_0p01': int(((a > 0) & (a < 0.01)).sum()),
        'abs_0p01_to_0p03': int(((a >= 0.01) & (a < 0.03)).sum()),
        'abs_0p03_to_0p05': int(((a >= 0.03) & (a < 0.05)).sum()),
        'abs_0p05_to_0p10': int(((a >= 0.05) & (a < 0.10)).sum()),
        'abs_ge_0p10': int((a >= 0.10).sum()),
        'sampler_flat_count': int((a <= sampler_threshold).sum()),
        'sampler_flat_pct': float((a <= sampler_threshold).mean()),
        'positive_count': int((targets > 0).sum()),
        'negative_count': int((targets < 0).sum()),
    }
    return out


def sample_rows(df: pd.DataFrame, targets: np.ndarray, mask: np.ndarray, n: int = 10) -> List[Dict]:
    idxs = np.where(mask)[0][:n]
    rows = []
    for i in idxs:
        rows.append({
            'idx': int(i),
            'timestamp': None if 'timestamp' not in df.columns else str(df.iloc[i]['timestamp']),
            'open': float(df.iloc[i]['open']),
            'high': float(df.iloc[i]['high']),
            'low': float(df.iloc[i]['low']),
            'close': float(df.iloc[i]['close']),
            'target': float(targets[i]),
        })
    return rows


def label_return_correlations(df: pd.DataFrame, targets: np.ndarray, max_lag: int = 3):
    """Correlate targets with 1-bar returns at various lags.

    - forward[L] : target[i] vs return[i+L]  (future return, inside the label window).
                   This is expected to correlate — it is just the label's own outcome.
    - past[L]    : target[i] vs return[i-L]  (past return) and same_bar.
                   These SHOULD be ~0. A large value means the label is explained by
                   information unavailable at decision time → a real leakage red flag.
    """
    close = df['close'].to_numpy(dtype=np.float64)
    ret1 = np.full_like(close, np.nan)
    ret1[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-12)

    n = len(targets)
    forward, past = {}, {}
    for L in range(1, max_lag + 1):
        xf, yf = ret1[L:n], targets[:n - L]
        mf = np.isfinite(xf) & np.isfinite(yf)
        forward[f'fwd_ret+{L}'] = float(np.corrcoef(xf[mf], yf[mf])[0, 1]) if mf.sum() > 10 else None
        xp, yp = ret1[:n - L], targets[L:n]
        mp = np.isfinite(xp) & np.isfinite(yp)
        past[f'past_ret-{L}'] = float(np.corrcoef(xp[mp], yp[mp])[0, 1]) if mp.sum() > 10 else None
    ms = np.isfinite(ret1) & np.isfinite(targets)
    past['same_bar'] = float(np.corrcoef(ret1[ms], targets[ms])[0, 1]) if ms.sum() > 10 else None

    leakage_risk = max((abs(v) for v in past.values() if v is not None), default=0.0)
    return {'forward': forward, 'past': past, 'leakage_risk': leakage_risk}


def oracle_behavior(event_counts: Dict, n: int) -> Dict:
    """Summarise the raw triple-barrier outcome mix (before any dropping)."""
    tp = (event_counts.get('LONG_TP', 0) + event_counts.get('SHORT_TP', 0)
          + event_counts.get('BOTH_TP', 0))
    sl = (event_counts.get('LONG_SL', 0) + event_counts.get('SHORT_SL', 0)
          + event_counts.get('BOTH_SL', 0))
    timeout = event_counts.get('TIMEOUT', 0)
    neutral = event_counts.get('AMBIGUOUS', 0) + event_counts.get('WIDE_CANDLE', 0)
    denom = max(n, 1)
    return {
        'tp_hit_rate': tp / denom,
        'sl_first_rate': sl / denom,
        'timeout_rate': timeout / denom,
        'neutral_rate': neutral / denom,
        'anomaly': (sl == 0 and tp > 0),  # SL level is closer than TP, so SL-first
                                           # should normally occur; 0 is suspicious.
    }


def training_readiness(X, y, df_raw, df_features, max_hold, holdout_fraction):
    """Pre-training gate: alignment, feature health, in-feature leakage, split integrity.

    X : aligned feature matrix actually handed to the model (df_features_aligned)
    y : aligned targets (y_targets_aligned), same index as X
    """
    out = {}
    flags = []

    # (a) Feature <-> target alignment (the #1 look-ahead risk)
    out['index_equal'] = bool(X.index.equals(y.index))
    out['index_monotonic'] = bool(X.index.is_monotonic_increasing)
    out['features_subset_of_raw'] = bool(set(X.index).issubset(set(df_raw.index)))
    n_feat, n_y = len(df_features), len(y)
    out['trailing_rows_trimmed'] = int(n_feat - n_y)
    if (n_feat - n_y) < (max_hold - 1):
        flags.append('ALIGNMENT: too few rows trimmed at tail; labels may lack a full forward window')
    if not out['index_equal']:
        flags.append('ALIGNMENT: feature/target indexes NOT identical (off-by-one / look-ahead risk)')
    if not out['features_subset_of_raw']:
        flags.append('ALIGNMENT: feature index not a subset of raw-data index')
    out['tz_aware'] = str(df_raw.index.tz)
    if df_raw.index.tz is None:
        flags.append('TIMEZONE: DatetimeIndex is tz-naive; confirm data is already in Asia/Kolkata')

    # (b) Per-feature NaN / constant (dead) features
    nan_pct = X.isna().mean()
    out['feature_nan_pct'] = {k: round(float(v), 4) for k, v in nan_pct.items()}
    dead = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    out['constant_features'] = dead
    if dead:
        flags.append(f'FEATURES: {len(dead)} constant/dead feature(s): {dead[:10]}')
    high_nan = [c for c, v in nan_pct.items() if v > 0.05]
    out['high_nan_features'] = high_nan
    if high_nan:
        flags.append(f'FEATURES: {len(high_nan)} feature(s) >5% NaN (drive row drops): {high_nan[:10]}')

    # (c) In-feature leakage screen: same-bar and next-bar correlation with target
    corr0, corr_fwd = {}, {}
    y_next = y.shift(-1)
    for c in X.columns:
        cc0 = X[c].corr(y)
        ccf = X[c].corr(y_next)
        corr0[c] = None if pd.isna(cc0) else round(float(cc0), 4)
        corr_fwd[c] = None if pd.isna(ccf) else round(float(ccf), 4)
    out['feature_target_corr_lag0'] = corr0
    out['feature_target_corr_lag+1'] = corr_fwd
    leaky = [c for c, v in corr0.items() if v is not None and abs(v) > 0.95]
    if leaky:
        flags.append(f'LEAKAGE: {len(leaky)} feature(s) |corr|>0.95 with target (too good / possible leakage): {leaky[:10]}')

    # (d) Train/holdout split integrity (mirrors split_train_holdout)
    n = len(X)
    split_idx = int(n * (1 - holdout_fraction))
    embargo = max_hold
    train_end_pos = max(split_idx - embargo, 0)
    train_idx = X.index[:train_end_pos]
    hold_idx = X.index[split_idx:]
    overlap = set(train_idx).intersection(set(hold_idx))
    out['split_train_rows'] = int(len(train_idx))
    out['split_holdout_rows'] = int(len(hold_idx))
    out['split_overlap'] = int(len(overlap))
    out['split_train_range'] = [str(X.index[0]), str(X.index[train_end_pos - 1])] if train_end_pos > 0 else [None, None]
    out['split_holdout_range'] = [str(X.index[split_idx]), str(X.index[-1])]
    if overlap:
        flags.append(f'SPLIT: {len(overlap)} overlapping rows between train and holdout (LEAKAGE)')
    if split_idx <= embargo:
        flags.append('SPLIT: train set too short after embargo; reduce holdout fraction or add data')

    out['flags'] = flags
    return out


def main():
    ap = argparse.ArgumentParser(
        description='Audit data and the ACTUAL labels the pipeline trains on.')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--out', default='output/data_label_audit')
    ap.add_argument('--limit-files', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.data_dir)

    files = list(cfg.DATA_FILE)
    if args.limit_files and args.limit_files > 0:
        files = files[:args.limit_files]

    # Real pipeline parameters (mirrors main_pipeline.load_and_prepare_data)
    max_hold   = int(cfg.ORACLE_MAX_HOLD)
    tp_mult    = float(cfg.TP_ATR_MULT)
    sl_mult    = float(cfg.SL_ATR_MULT)
    atr_period = int(cfg.ATR_PERIOD)
    drop_both_sl = bool(getattr(cfg, 'DROP_WHIPSAW', True))
    drop_neutral  = bool(getattr(cfg, 'DROP_NEUTRAL', False))

    all_summaries = []
    for path in files:
        df = load_ohlc_csv(path)
        ohlc = basic_ohlc_checks(df)

        # --- Replicate the pipeline's exact preprocessing + target generation ---
        # load_and_prepare_data(): lowercase cols, datetime index named 'datetime', sort.
        df_raw = df.copy()
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], errors='coerce')
        df_raw = df_raw.set_index('timestamp')
        df_raw.index.name = 'datetime'
        df_raw = df_raw.sort_index()

        df_features = calculate_features(
            df_raw,
            add_session_features=True,
            session=SessionConfig(),
            clip_outside_session=True,
            mds_fast_window=cfg.FE_MDS_FAST_WINDOW,
            mds_slow_window=cfg.FE_MDS_SLOW_WINDOW,
        )

        # Same call main_pipeline uses; return_metadata gives us the event breakdown.
        df_features_aligned, y_targets_aligned, meta_aligned = generate_tbm_targets(
            df_raw, df_features,
            max_hold=max_hold,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            atr_period=atr_period,
            drop_both_sl=drop_both_sl,
            drop_neutral=drop_neutral,
            return_metadata=True,
        )
        X = df_features_aligned
        y = y_targets_aligned

        # Training-readiness gate: alignment, feature health, in-feature leakage, split.
        holdout_fraction = float(getattr(cfg, 'GEL_HOLDOUT_FRACTION', 0.20))
        readiness = training_readiness(X, y, df_raw, df_features, max_hold, holdout_fraction)

        # Raw event distribution (no dropping) — this is what the oracle actually
        # produced before the pipeline removes whipsaw/neutral rows. Needed to judge
        # oracle behaviour (TP vs SL vs timeout mix), which the dropped set hides.
        _, _, meta_raw = generate_tbm_targets(
            df_raw, df_features,
            max_hold=max_hold,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            atr_period=atr_period,
            drop_both_sl=False,
            drop_neutral=False,
            return_metadata=True,
        )
        raw_reasons = meta_raw['event_type'].value_counts(dropna=False).to_dict()
        behavior = oracle_behavior(raw_reasons, len(meta_raw))

        targets = y_targets_aligned.to_numpy(dtype=np.float32)

        # Display frame aligned row-for-row with the generated targets.
        display = df_raw.loc[meta_aligned.index].reset_index()

        # Reason/event breakdown uses the real pipeline event codes.
        oracle_audit = pd.DataFrame({
            'idx': np.arange(len(meta_aligned)),
            'target': meta_aligned['target'].to_numpy(),
            'event_type': meta_aligned['event_type'].to_numpy(),
            'reason': meta_aligned['event_type'].to_numpy(),
        })
        sampler_threshold = float(getattr(cfg, 'SIGNAL_SEP_FLOOR', 0.05))
        dist = target_distribution(targets, sampler_threshold)
        reasons = oracle_audit['reason'].value_counts(dropna=False).to_dict()

        flat_examples = sample_rows(display, targets, np.abs(targets) <= sampler_threshold, 12)
        zero_examples = sample_rows(display, targets, targets == 0.0, 12)
        tiny_examples = sample_rows(display, targets, (np.abs(targets) > 0) & (np.abs(targets) < 0.05), 12)
        small_examples = sample_rows(display, targets, (np.abs(targets) >= 0.05) & (np.abs(targets) < 0.10), 12)
        pos_tail_examples = sample_rows(display, targets, targets > 0.5, 12)
        neg_tail_examples = sample_rows(display, targets, targets < -0.5, 12)

        # Window examples: locate each decision bar in the raw frame, then show the
        # forward max_hold window the triple-barrier method actually evaluated.
        raw_positions = df_raw.index.get_indexer(meta_aligned.index)
        oracle_window_examples = []
        for k, pos in enumerate(raw_positions[:8]):
            if pos < 0:
                continue
            end = min(pos + max_hold, len(df_raw) - 1)
            window = df_raw.iloc[pos:end + 1][['open', 'high', 'low', 'close']].copy()
            window = window.reset_index()
            window['target_at_entry'] = np.nan
            window.iloc[0, window.columns.get_loc('target_at_entry')] = float(meta_aligned['target'].iloc[k])
            oracle_window_examples.append({
                'idx': int(pos),
                'reason': meta_aligned['event_type'].iloc[k],
                'target': float(meta_aligned['target'].iloc[k]),
                'rows': window.astype(str).to_dict(orient='records'),
            })

        leak = label_return_correlations(display, targets, max_lag=3)

        base = os.path.splitext(os.path.basename(path))[0]
        oracle_audit.to_csv(os.path.join(args.out, f'{base}_oracle_audit.csv'), index=False)
        pd.DataFrame(flat_examples).to_csv(os.path.join(args.out, f'{base}_flat_sampler_examples.csv'), index=False)
        pd.DataFrame(zero_examples).to_csv(os.path.join(args.out, f'{base}_exact_zero_examples.csv'), index=False)
        pd.DataFrame(tiny_examples).to_csv(os.path.join(args.out, f'{base}_tiny_signal_examples.csv'), index=False)
        pd.DataFrame(small_examples).to_csv(os.path.join(args.out, f'{base}_small_signal_examples.csv'), index=False)
        pd.DataFrame(pos_tail_examples).to_csv(os.path.join(args.out, f'{base}_positive_tail_examples.csv'), index=False)
        pd.DataFrame(neg_tail_examples).to_csv(os.path.join(args.out, f'{base}_negative_tail_examples.csv'), index=False)

        summary = {
            'file': path,
            'ohlc_checks': ohlc,
            'target_distribution': dist,
            'trained_label_counts': {k: int(v) for k, v in reasons.items()},
            'raw_event_counts': {k: int(v) for k, v in raw_reasons.items()},
            'oracle_behavior': {k: (float(v) if isinstance(v, float) else v) for k, v in behavior.items()},
            'flat_sampler_examples_preview': flat_examples[:5],
            'zero_examples_preview': zero_examples[:5],
            'tiny_signal_examples_preview': tiny_examples[:5],
            'small_signal_examples_preview': small_examples[:5],
            'label_return_correlations': leak,
            'training_readiness': readiness,
            'notes': [
                'Targets are generated by src/target_generator.generate_tbm_targets — the SAME',
                'function main_pipeline.py uses to build training labels. This audit therefore',
                'checks the actual labels the model trains on, not an independent reimplementation.',
                'raw_event_counts = oracle output BEFORE dropping whipsaw/neutral rows;',
                'trained_label_counts = what the model actually trains on after DROP_WHIPSAW/DROP_NEUTRAL.',
                'oracle_behavior.anomaly=True means ZERO SL-first events despite SL(1.9) < TP(2.4):',
                'the stop is never hit first, which is suspicious for trending-vs-labeling and worth investigating.',
                'label_return_correlations.past/same_bar should be ~0 (real leakage if large);',
                'forward lags are expected to correlate (they are just the label\'s own window).',
                'Review oracle_audit.csv for the per-row event-type breakdown.',
            ],
            'oracle_window_examples_preview': oracle_window_examples,
        }
        with open(os.path.join(args.out, f'{base}_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        all_summaries.append(summary)

        # ── Console audit report (so the findings are visible, not just written) ──
        print("\n" + "=" * 70)
        print(f"AUDIT REPORT: {os.path.basename(path)}")
        print("=" * 70)
        print(f"  Raw rows              : {ohlc['rows']}")
        print(f"  Trained label rows    : {dist['n']}")
        print(f"  OHLC violations       : {ohlc['ohlc_violations']}")
        print(f"  Duplicate timestamps  : {ohlc['duplicated_timestamps']}")
        print(f"  Non-monotonic ts      : {ohlc['non_monotonic_timestamps']}")
        print(f"  Long / Short / Neutral: {dist['positive_count']} / "
              f"{dist['negative_count']} / {dist['exact_zero_count']}")
        print(f"  RAW oracle event mix  : {raw_reasons}")
        print(f"  Trained label mix     : {reasons}")
        print(f"  Oracle behaviour      : TP-hit={behavior['tp_hit_rate']:.1%} "
              f"SL-first={behavior['sl_first_rate']:.1%} timeout={behavior['timeout_rate']:.1%}")
        if behavior['anomaly']:
            print("  !! ANOMALY: 0 SL-first events though SL(1.9) < TP(2.4) — investigate.")
        fwd = ", ".join(f"{k}={v:.3f}" for k, v in leak['forward'].items() if v is not None)
        past = ", ".join(f"{k}={v:.3f}" for k, v in leak['past'].items() if v is not None)
        print(f"  Label↔FUTURE return   : {fwd}   (expected, not leakage)")
        print(f"  Label↔PAST return     : {past}   (leakage_risk={leak['leakage_risk']:.3f}, want ~0)")

        # Training-readiness gate
        print(f"  --- TRAINING READINESS {'OK' if not readiness['flags'] else 'FLAGS (' + str(len(readiness['flags'])) + ')'}")
        print(f"  Align: idx_equal={readiness['index_equal']} monotonic={readiness['index_monotonic']} "
              f"tz={readiness['tz_aware']} trailing_trimmed={readiness['trailing_rows_trimmed']}")
        print(f"  Features: n={len(X.columns)} constant={len(readiness['constant_features'])} "
              f"high_nan(>5%)={len(readiness['high_nan_features'])}")
        if readiness['constant_features']:
            print(f"    constant: {readiness['constant_features']}")
        if readiness['high_nan_features']:
            print(f"    high_nan: {readiness['high_nan_features']}")
        print(f"  Split: train={readiness['split_train_rows']} holdout={readiness['split_holdout_rows']} "
              f"overlap={readiness['split_overlap']}")
        print(f"    train   : {readiness['split_train_range'][0]} -> {readiness['split_train_range'][1]}")
        print(f"    holdout : {readiness['split_holdout_range'][0]} -> {readiness['split_holdout_range'][1]}")
        for fl in readiness['flags']:
            print(f"    !! {fl}")
        print(f"  Reports written to    : {args.out}")
        print("=" * 70)

    aggregate = {
        'config_snapshot': {
            'DATA_FILE': files,
            'DATAPATH': cfg.DATAPATH,
            'ORACLE_MAX_HOLD': max_hold,
            'TP_ATR_MULT': tp_mult,
            'SL_ATR_MULT': sl_mult,
            'ATR_PERIOD': atr_period,
            'DROP_WHIPSAW': drop_both_sl,
            'DROP_NEUTRAL': drop_neutral,
            'TARGET_GENERATOR': 'src.target_generator.generate_tbm_targets',
            'FE_MDS_FAST_WINDOW': cfg.FE_MDS_FAST_WINDOW,
            'FE_MDS_SLOW_WINDOW': cfg.FE_MDS_SLOW_WINDOW,
        },
        'files': all_summaries,
        'checklist_covered': [
            'Data integrity: OHLC violations, nulls, duplicate/non-monotonic timestamps',
            'Actual pipeline labels (long/short/neutral) via generate_tbm_targets',
            'RAW oracle outcome mix (TP/SL/timeout) BEFORE dropping — the real label behaviour',
            'Trained label mix after DROP_WHIPSAW / DROP_NEUTRAL',
            'Anomaly flag: zero SL-first events despite SL < TP',
            'Leakage: label vs PAST/same-bar returns (~0 expected) separated from expected forward consistency',
            'Forward-window inspection of how each label was decided',
        ],
    }
    with open(os.path.join(args.out, 'aggregate_summary.json'), 'w') as f:
        json.dump(aggregate, f, indent=2)

    md = ['# Data and Label Audit', '']
    md.append('Targets are generated by the SAME generate_tbm_targets the training pipeline uses.')
    md.append('')
    for s in all_summaries:
        md.append(f"## {os.path.basename(s['file'])}")
        td = s['target_distribution']
        md.append(f"- Rows (raw): {s['ohlc_checks']['rows']}")
        md.append(f"- Trained label rows: {td['n']}")
        md.append(f"- Long / Short / Neutral: {td['positive_count']} / {td['negative_count']} / {td['exact_zero_count']}")
        md.append(f"- OHLC violations: {json.dumps(s['ohlc_checks']['ohlc_violations'])}")
        md.append(f"- RAW oracle event mix (before dropping): {json.dumps(s['raw_event_counts'])}")
        md.append(f"- Trained label mix (after dropping): {json.dumps(s['trained_label_counts'])}")
        b = s['oracle_behavior']
        md.append(f"- Oracle behaviour: TP-hit={b['tp_hit_rate']:.1%} SL-first={b['sl_first_rate']:.1%} "
                  f"timeout={b['timeout_rate']:.1%} anomaly={b['anomaly']}")
        lr = s['label_return_correlations']
        fwd = ", ".join(f"{k}={v:.3f}" for k, v in lr['forward'].items() if v is not None)
        past = ", ".join(f"{k}={v:.3f}" for k, v in lr['past'].items() if v is not None)
        md.append(f"- Label↔FUTURE return (expected): {fwd}")
        md.append(f"- Label↔PAST return (leakage, want ~0): {past}  risk={lr['leakage_risk']:.3f}")
        r = s['training_readiness']
        md.append(f"- READINESS: {'OK' if not r['flags'] else str(len(r['flags'])) + ' flag(s)'}")
        md.append(f"-   alignment: idx_equal={r['index_equal']} monotonic={r['index_monotonic']} "
                  f"tz={r['tz_aware']} trailing_trimmed={r['trailing_rows_trimmed']}")
        md.append(f"-   features: n={len(s['training_readiness']['feature_nan_pct'])} "
                  f"constant={r['constant_features']} high_nan={r['high_nan_features']}")
        md.append(f"-   split: train={r['split_train_rows']} holdout={r['split_holdout_rows']} "
                  f"overlap={r['split_overlap']}")
        md.append(f"-   train   : {r['split_train_range'][0]} -> {r['split_train_range'][1]}")
        md.append(f"-   holdout : {r['split_holdout_range'][0]} -> {r['split_holdout_range'][1]}")
        for fl in r['flags']:
            md.append(f"-   !! {fl}")
        md.append('')
    with open(os.path.join(args.out, 'README.md'), 'w') as f:
        f.write('\n'.join(md))


if __name__ == '__main__':
    main()
