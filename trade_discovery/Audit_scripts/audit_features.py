#!/usr/bin/env python3
"""Audit feature normalization and dominance.

Builds the SAME feature matrix the pipeline uses (src.feature_engineering.calculate_features)
plus the aligned targets (src.target_generator.generate_tbm_targets), then checks:
  1. Normalization quality per feature: dead (constant), saturated (stuck at +/-1 boundary),
     extreme scale, NaN rate.
  2. Dominance:
       - scale dominance   : one feature's std orders of magnitude above the rest
       - predictive domin. : one feature's |corr| with the target dwarfs all others
       - collinearity      : pairs of features that are near-identical (redundant)

Writes a JSON report to output/feature_audit/ and prints a console summary.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "project_core")
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)  # resolve relative cfg.DATAPATH (data/) against project_core

import src.config as cfg
from src.feature_engineering import calculate_features, SessionConfig
from src.target_generator import generate_tbm_targets


def build_raw(df):
    df = df.copy()
    tscol = 'timestamp' if 'timestamp' in df.columns else 'date'
    df[tscol] = pd.to_datetime(df[tscol], errors='coerce')
    df = df.set_index(tscol)
    df.index.name = 'datetime'
    df = df.sort_index()
    if df.index.tz is None:
        df = df.tz_localize('Asia/Kolkata')
    return df


def normalization_report(X):
    """Per-feature normalization diagnostics."""
    rep = {}
    for c in X.columns:
        s = X[c]
        finite = s[np.isfinite(s)]
        nan_pct = float(s.isna().mean())
        if len(finite) == 0:
            rep[c] = {'nan_pct': 1.0, 'dead': True, 'saturated_pct': None,
                      'mean': None, 'std': None, 'min': None, 'max': None}
            continue
        std = float(finite.std())
        nuniq = int(finite.nunique())
        is_binary = nuniq <= 2  # 0/1 masks are legitimately at the boundary; not saturation
        rep[c] = {
            'nan_pct': round(nan_pct, 4),
            'dead': bool(nuniq <= 1),
            'std': round(std, 4),
            'mean': round(float(finite.mean()), 4),
            'min': round(float(finite.min()), 4),
            'max': round(float(finite.max()), 4),
            # fraction stuck at the tanh boundary (sign of broken normalization),
            # ignoring legitimate binary 0/1 mask features.
            'saturated_pct': None if is_binary else round(float((finite.abs() >= 0.999).mean()), 4),
        }
    return rep


def scale_dominance(X, norm):
    stds = {c: v['std'] for c, v in norm.items() if v['std'] is not None and v['std'] > 0}
    if not stds:
        return {'ok': True, 'max_over_median': None, 'top': []}
    arr = np.array(sorted(stds.values()))
    median = float(np.median(arr))
    mx = float(arr.max())
    ratio = mx / median if median > 0 else None
    top = sorted(stds.items(), key=lambda kv: -kv[1])[:5]
    return {
        'ok': (ratio is None) or (ratio <= 20),
        'max_over_median': round(ratio, 2) if ratio is not None else None,
        'min_std': round(float(arr.min()), 5),
        'max_std': round(mx, 4),
        'top': top,
    }


def predictive_dominance(X, y):
    corrs = {}
    for c in X.columns:
        cc = X[c].corr(y)
        if pd.notna(cc):
            corrs[c] = float(cc)
    ranked = sorted(corrs.items(), key=lambda kv: -abs(kv[1]))
    if not ranked:
        return {'ok': True, 'top': [], 'top_over_second': None}
    top_abs = [abs(v) for _, v in ranked[:2]]
    ratio = top_abs[0] / top_abs[1] if len(top_abs) > 1 and top_abs[1] > 0 else None
    return {
        'ok': (ratio is None) or (ratio <= 3),  # top feature not wildly dominant
        'top_over_second': round(ratio, 2) if ratio is not None else None,
        'top': [(c, round(v, 4)) for c, v in ranked[:5]],
    }


def collinearity(X, threshold=0.95, top_n=15):
    num = X.select_dtypes(include=[np.number]).dropna(axis=1, how='any')
    if num.shape[1] < 2:
        return {'pairs': []}
    corr = num.corr().abs()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v) and v >= threshold:
                pairs.append((cols[i], cols[j], round(float(v), 4)))
    pairs.sort(key=lambda t: -t[2])
    return {'pairs': pairs[:top_n], 'count': len(pairs)}


def main():
    ap = argparse.ArgumentParser(description='Audit feature normalization and dominance.')
    ap.add_argument('--out', default='output/feature_audit')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df_raw = build_raw(pd.read_csv(cfg.DATAPATH))
    df_features = calculate_features(
        df_raw, add_session_features=True, session=SessionConfig(),
        clip_outside_session=True,
        mds_fast_window=cfg.FE_MDS_FAST_WINDOW, mds_slow_window=cfg.FE_MDS_SLOW_WINDOW,
    )
    X_full = df_features
    # Align to the labels the pipeline actually trains on.
    X, y, _ = generate_tbm_targets(
        df_raw, df_features,
        max_hold=int(cfg.ORACLE_MAX_HOLD), tp_mult=float(cfg.TP_ATR_MULT),
        sl_mult=float(cfg.SL_ATR_MULT), atr_period=int(cfg.ATR_PERIOD),
        drop_both_sl=bool(getattr(cfg, 'DROP_WHIPSAW', True)),
        drop_neutral=bool(getattr(cfg, 'DROP_NEUTRAL', False)),
        return_metadata=True,
    )

    norm = normalization_report(X)
    dead = [c for c, v in norm.items() if v.get('dead')]
    saturated = [c for c, v in norm.items()
                 if v.get('saturated_pct') is not None and v['saturated_pct'] > 0.95]
    high_nan = [c for c, v in norm.items() if v['nan_pct'] > 0.05]
    sd = scale_dominance(X, norm)
    pd_dom = predictive_dominance(X, y)
    coll = collinearity(X)

    report = {
        'n_features': int(X.shape[1]),
        'n_rows': int(X.shape[0]),
        'dead_features': dead,
        'saturated_features': saturated,
        'high_nan_features': high_nan,
        'scale_dominance': sd,
        'predictive_dominance': pd_dom,
        'collinearity': coll,
        'per_feature': norm,
    }
    with open(os.path.join(args.out, 'feature_normalization.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # ── Console summary ──
    print("=" * 70)
    print(f"FEATURE NORMALIZATION & DOMINANCE AUDIT: {os.path.basename(cfg.DATAPATH)}")
    print("=" * 70)
    print(f"  Features: {X.shape[1]} | Rows: {X.shape[0]}")
    print(f"  Dead (constant)      : {len(dead)}  -> {dead[:10]}")
    print(f"  Saturated (|x|>=0.999): {len(saturated)} -> {saturated[:10]}")
    print(f"  High-NaN (>5%)        : {len(high_nan)} -> {high_nan[:10]}")
    print(f"  Scale dominance       : {'OK' if sd['ok'] else 'FLAG'} "
          f"(max/std over median = {sd['max_over_median']}, max_std={sd['max_std']})")
    for c, v in sd['top']:
        print(f"      top-std: {c} = {v}")
    print(f"  Predictive dominance  : {'OK' if pd_dom['ok'] else 'FLAG'} "
          f"(top/2nd |corr| = {pd_dom['top_over_second']})")
    for c, v in pd_dom['top']:
        print(f"      |corr w target|: {c} = {v}")
    print(f"  Collinear pairs (|r|>=0.95): {coll['count']}")
    for a, b, r in coll['pairs'][:8]:
        print(f"      {a}  ~  {b}  (r={r})")
    print("=" * 70)
    verdict = []
    if saturated:
        verdict.append("SATURATED features => broken normalization (fix _apply_smart_normalization)")
    if dead:
        verdict.append(f"{len(dead)} dead features (candlesticks expected; others investigate)")
    if not sd['ok']:
        verdict.append("SCALE DOMINANCE: one feature dwarfs others by std")
    if not pd_dom['ok']:
        verdict.append("PREDICTIVE DOMINANCE: a single feature explains the target far more than the rest")
    if coll['count'] > 20:
        verdict.append(f"HIGH COLLINEARITY: {coll['count']} redundant feature pairs")
    if not verdict:
        verdict.append("All checks passed: features are normalized and no single feature dominates.")
    print("  VERDICT:")
    for v in verdict:
        print(f"    - {v}")
    print(f"  Report written to: {args.out}/feature_normalization.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
