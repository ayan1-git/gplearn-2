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


def leakage_surrogates(df: pd.DataFrame, targets: np.ndarray, max_lag: int = 3) -> Dict:
    close = df['close'].to_numpy(dtype=np.float64)
    ret1 = np.full_like(close, np.nan)
    ret1[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-12)
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = ret1[-lag:]
            y = targets[:len(targets) + lag]
        elif lag > 0:
            x = ret1[:-lag]
            y = targets[lag:]
        else:
            x = ret1
            y = targets
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 10:
            out[f'ret1_vs_target_lag_{lag}'] = float(np.corrcoef(x[m], y[m])[0, 1])
        else:
            out[f'ret1_vs_target_lag_{lag}'] = None
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

        leak = leakage_surrogates(display, targets, max_lag=3)

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
            'oracle_reason_counts': {k: int(v) for k, v in reasons.items()},
            'flat_sampler_examples_preview': flat_examples[:5],
            'zero_examples_preview': zero_examples[:5],
            'tiny_signal_examples_preview': tiny_examples[:5],
            'small_signal_examples_preview': small_examples[:5],
            'leakage_surrogates': leak,
            'notes': [
                'Targets are generated by src/target_generator.generate_tbm_targets — the SAME',
                'function main_pipeline.py uses to build training labels. This audit therefore',
                'checks the actual labels the model trains on, not an independent reimplementation.',
                'exact_zero (target==0) corresponds to neutral labels; positive/negative to long/short.',
                'Large ret1_vs_target correlations at negative lags can indicate suspicious alignment or leakage proxies.',
                'Review oracle_audit.csv for the real event-type breakdown (TIMEOUT, BOTH_TP, LONG_SL, ...).',
            ],
            'oracle_window_examples_preview': oracle_window_examples,
        }
        with open(os.path.join(args.out, f'{base}_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        all_summaries.append(summary)

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
            'Target distribution sanity / exact-zero (neutral) share',
            'Real triple-barrier event breakdown (timeout, both-TP, SL-only, ...)',
            'Forward-window inspection of how each label was decided',
            'Simple leakage surrogate correlations across lags',
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
        md.append(f"- Label rows (post-filter): {td['n']}")
        md.append(f"- Neutral (exact zero) count: {td['exact_zero_count']} ({td['exact_zero_pct']:.2%})")
        md.append(f"- Long / Short counts: {td['positive_count']} / {td['negative_count']}")
        md.append(f"- OHLC violations: {json.dumps(s['ohlc_checks']['ohlc_violations'])}")
        md.append(f"- Top event reasons: {json.dumps(s['oracle_reason_counts'])}")
        md.append(f"- Leakage surrogates: {json.dumps(s['leakage_surrogates'])}")
        md.append('')
    with open(os.path.join(args.out, 'README.md'), 'w') as f:
        f.write('\n'.join(md))


if __name__ == '__main__':
    main()
