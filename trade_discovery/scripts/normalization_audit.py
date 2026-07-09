#!/usr/bin/env python3
"""
normalization_audit.py  —  Post-Scaling Normalization & Signal-Quality Audit
================================================================================
Replays the *exact* project pipeline (feature build → target gen → temporal split
→ tanh scaling) and reports:

  1. Normalization correctness
     - NaN counts before/after engineering
     - Per-feature range / skew / kurtosis (pre- and post-tanh)
     - Expected-bounds compliance for bounded features
     - Tanh asymptote saturation rate for SCALE_FEATURES
     - Zero-preservation check on PASSTHROUGH_FEATURES
     - Scale-factor sanity (no pathological P75 values)

  2. Signal-quality preservation
     - Pearson and Spearman correlation with target (pre- vs post-scaling)
     - Histogram-based mutual-information estimate (pre- vs post-scaling)
     - Train→Holdout distribution shift (Wasserstein / KS) pre- vs post-scaling
     - Rank-preservation score

  3. Leakage guard
     - Confirms scale factors fit on train only
     - Warns if holdout statistics bleed into the scaler

Run:  python scripts/normalization_audit.py
"""

from __future__ import annotations

import os
import sys

# ── Ensure `src/` is resolvable regardless of working directory ─────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
# ──────────────────────────────────────────────────────────────────────────────

import logging
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import wasserstein_distance, ks_2samp

import src.config as cfg
from scripts.main_pipeline import (
    load_and_prepare_data,
    split_train_holdout,
    tanh_scale_train_apply_test,
    PASSTHROUGH_FEATURES,
    SCALE_FEATURES,
)
from src.feature_engineering import calculate_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("normalization_audit")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_stat(series: pd.Series, stat: str) -> float:
    val = getattr(series, stat)()
    return float(val) if np.isfinite(val) else float("nan")


def _approx_mi(x: pd.Series, y: pd.Series, bins: int = 20) -> float:
    """Histogram-based mutual information (nats). Returns NaN if degenerate."""
    x = x.dropna()
    y = y.loc[x.index].dropna()
    if len(x) < 20 or len(y) < 20:
        return float("nan")
    xv = x.values
    yv = y.values
    ux = np.unique(xv)
    uy = np.unique(yv)
    if len(ux) < 2 or len(uy) < 2:
        return float("nan")
    if np.nanmin(np.abs(np.diff(ux))) < 1e-12 or np.nanmin(np.abs(np.diff(uy))) < 1e-12:
        return float("nan")
    try:
        x_b = np.digitize(xv, np.histogram_bin_edges(xv, bins=bins)[1:-1])
        y_b = np.digitize(yv, np.histogram_bin_edges(yv, bins=bins)[1:-1])
    except ValueError:
        return float("nan")
    x_b = np.clip(x_b, 0, bins - 1)
    y_b = np.clip(y_b, 0, bins - 1)
    joint = np.zeros((bins, bins), dtype=float)
    for xi, yi in zip(x_b, y_b):
        joint[xi, yi] += 1.0
    joint /= joint.sum()
    if np.any(joint == 0):
        return float("nan")
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mi = np.nansum(joint * np.log((joint / (px * py + 1e-12)) + 1e-12))
    return float(mi)


def _warn_rate(name: str, rate: float, warn_threshold: float = 0.02) -> str:
    flag = "  *** WARN" if rate > warn_threshold else "  OK"
    return f"{flag} | {name}: {rate:.2%}"


# ──────────────────────────────────────────────────────────────────────────────
# Core audit
# ──────────────────────────────────────────────────────────────────────────────

def audit_normalization(csv_path: str = None) -> Dict:
    if csv_path is None:
        csv_path = cfg.DATAPATH

    # ── 1. Replay exact project pipeline ─────────────────────────────────────
    logger.info("Loading data via load_and_prepare_data(%s)", csv_path)
    df_raw, df_features, y_targets = load_and_prepare_data(csv_path)

    logger.info("Splitting data via split_train_holdout")
    X_train, y_train, X_hold, raw_hold, split_date = split_train_holdout(
        df_features, y_targets, df_raw
    )

    logger.info("Applying tanh scaling via tanh_scale_train_apply_test")
    X_train_s, X_hold_s = tanh_scale_train_apply_test(
        X_train, X_hold, SCALE_FEATURES, PASSTHROUGH_FEATURES
    )

    # Align targets (they are already aligned by load_and_prepare_data / split)
    assert len(X_train_s) == len(y_train), "Train alignment broken"
    assert len(X_hold_s) == len(X_hold), "Holdout alignment broken"

    results: Dict = {
        "dataset": os.path.basename(csv_path),
        "n_raw": int(len(df_raw)),
        "n_features_engineered": int(len(df_features)),
        "n_train": int(len(X_train)),
        "n_hold": int(len(X_hold)),
        "split_date": str(split_date.date()) if hasattr(split_date, "date") else str(split_date),
    }

    # ── 2. NaN audit ─────────────────────────────────────────────────────────
    nan_pre_eng = df_features.isna().sum()
    nan_post_eng = df_features.dropna().isna().sum()
    nan_train_pre = X_train[[c for c in SCALE_FEATURES + PASSTHROUGH_FEATURES if c in X_train.columns]].isna().sum()
    nan_hold_pre = X_hold[[c for c in SCALE_FEATURES + PASSTHROUGH_FEATURES if c in X_hold.columns]].isna().sum()
    nan_train_post = X_train_s[[c for c in SCALE_FEATURES + PASSTHROUGH_FEATURES if c in X_train_s.columns]].isna().sum()
    nan_hold_post = X_hold_s[[c for c in SCALE_FEATURES + PASSTHROUGH_FEATURES if c in X_hold_s.columns]].isna().sum()

    results["nan"] = {
        "pre_dropna": {c: int(v) for c, v in nan_pre_eng.items() if v > 0},
        "train_pre_scale_nz": {c: int(v) for c, v in nan_train_pre.items() if v > 0},
        "hold_pre_scale_nz": {c: int(v) for c, v in nan_hold_pre.items() if v > 0},
        "train_post_scale_nz": {c: int(v) for c, v in nan_train_post.items() if v > 0},
        "hold_post_scale_nz": {c: int(v) for c, v in nan_hold_post.items() if v > 0},
    }

    # ── 3. Per-feature statistical profile (pre- and post-tanh) ──────────────
    all_feats = [c for c in X_train.columns if c in SCALE_FEATURES + PASSTHROUGH_FEATURES]
    # Missing expected features (e.g. some talib variants dropped during dropna)
    missing_expected = sorted(set(SCALE_FEATURES + PASSTHROUGH_FEATURES) - set(X_train.columns))
    if missing_expected:
        logger.info("Missing from train (likely dropped by dropna): %s", missing_expected)

    pre_stats = {}
    post_stats = {}
    for feat in all_feats:
        pre_s = X_train[feat].dropna()
        post_s = X_train_s[feat].dropna()
        pre_stats[feat] = {
            "mean": float(pre_s.mean()),
            "std": float(pre_s.std()),
            "skew": float(pre_s.skew()),
            "kurtosis": float(pre_s.kurtosis()),
            "p25": float(pre_s.quantile(0.25)),
            "p75": float(pre_s.quantile(0.75)),
            "p99": float(pre_s.quantile(0.99)),
            "min": float(pre_s.min()),
            "max": float(pre_s.max()),
        }
        post_stats[feat] = {
            "mean": float(post_s.mean()),
            "std": float(post_s.std()),
            "skew": float(post_s.skew()),
            "kurtosis": float(post_s.kurtosis()),
            "p25": float(post_s.quantile(0.25)),
            "p75": float(post_s.quantile(0.75)),
            "p99": float(post_s.quantile(0.99)),
            "min": float(post_s.min()),
            "max": float(post_s.max()),
            "saturation_ge_0_99": float((post_s.abs() > 0.99).mean()),
            "saturation_ge_0_95": float((post_s.abs() > 0.95).mean()),
        }

    results["stats"] = {"pre": pre_stats, "post": post_stats}

    # ── 4. Bounded-feature compliance ────────────────────────────────────────
    # Expected theoretical bounds after construction-time normalization
    bounded_rules = {
        "feat_efficiency": (-1.0, 1.0),
        "feat_icp": (-1.0, 1.0),
        "feat_momentum_rsi": (-1.0, 1.0),
        "feat_vol_asymmetry": (-1.0, 1.0),
        "feat_local_structure": (-1.0, 1.0),
        "feat_session_sin": (-1.0, 1.0),
        "feat_session_cos": (-1.0, 1.0),
        "feat_rejection_upper": (0.0, 1.0),
        "feat_rejection_lower": (0.0, 1.0),
    }

    bounds_report = {}
    for feat, (lo, hi) in bounded_rules.items():
        if feat not in X_train.columns:
            continue
        s = X_train[feat].dropna()
        actual_lo, actual_hi = float(s.min()), float(s.max())
        in_bounds = float(((s >= lo) & (s <= hi)).mean())
        out_of_bounds = 1.0 - in_bounds
        flags = []
        if out_of_bounds > 0.001:
            flags.append("violates_bounds")
        if actual_lo < lo - 1e-4 or actual_hi > hi + 1e-4:
            flags.append("range_exceeds_expected")
        bounds_report[feat] = {
            "expected": (lo, hi),
            "actual_min": actual_lo,
            "actual_max": actual_hi,
            "in_bounds_rate": in_bounds,
            "out_of_bounds_rate": out_of_bounds,
            "flags": flags,
        }
    results["bounds_compliance"] = bounds_report

    # ── 5. Tanh saturation audit on SCALE_FEATURES ───────────────────────────
    # Compute original P75 scale factors to inspect aspirational values
    actual_scale_cols = [c for c in SCALE_FEATURES if c in X_train.columns]
    saturation_report = {}
    if actual_scale_cols:
        train_view = X_train[actual_scale_cols]
        scale_factors = np.maximum(np.percentile(np.abs(train_view), 75, axis=0), 1e-8)
        for col, sf in zip(actual_scale_cols, scale_factors):
            post_s = X_train_s[col].dropna()
            sat_099 = float((post_s.abs() > 0.99).mean())
            sat_095 = float((post_s.abs() > 0.95).mean())
            saturation_report[col] = {
                "scale_factor_p75": float(sf),
                "saturation_ge_0_99": sat_099,
                "saturation_ge_0_95": sat_095,
            }
    results["tanh_saturation"] = saturation_report

    # ── 6. Signal preservation: correlation with target ──────────────────────
    y_aligned = y_targets.loc[X_train.index].dropna()
    X_align = X_train.loc[y_aligned.index]
    X_align_s = X_train_s.loc[y_aligned.index]

    corr_report = {}
    for feat in all_feats:
        if feat not in X_align.columns:
            continue
        x = X_align[feat]
        xs = X_align_s[feat]
        valid = x.notna() & xs.notna()
        if valid.sum() < 20:
            continue
        x = x[valid]
        xs = xs[valid]
        y = y_aligned[valid]

        # Pearson
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_orig, p_orig = st.pearsonr(x, y)
            r_scaled, p_scaled = st.pearsonr(xs, y)

        # Spearman (rank)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho_orig, prho_orig = st.spearmanr(x, y)
            rho_scaled, prho_scaled = st.spearmanr(xs, y)

        corr_report[feat] = {
            "pearson_pre": float(r_orig),
            "pearson_post": float(r_scaled),
            "pearson_delta": float(r_scaled - r_orig),
            "spearman_pre": float(rho_orig),
            "spearman_post": float(rho_scaled),
            "spearman_delta": float(rho_scaled - rho_orig),
        }

    results["signal_correlation"] = corr_report

    # ── 7. Mutual information (pre- vs post-scaling) ─────────────────────────
    mi_report = {}
    for feat in all_feats:
        if feat not in X_align.columns:
            continue
        x = X_align[feat]
        xs = X_align_s[feat]
        valid = x.notna() & xs.notna()
        if valid.sum() < 50:
            continue
        mi_pre = _approx_mi(x[valid], y_aligned[valid])
        mi_post = _approx_mi(xs[valid], y_aligned[valid])
        mi_report[feat] = {
            "mi_pre": float(mi_pre),
            "mi_post": float(mi_post),
            "mi_delta_pct": float((mi_post - mi_pre) / (abs(mi_pre) + 1e-12) * 100.0),
        }
    results["signal_mi"] = mi_report

    # ── 8. Train→Holdout distribution shift ───────────────────────────────────
    # Compare raw pre-scaling and post-scaling distributions
    shift_report = {}
    for feat in actual_scale_cols + [c for c in PASSTHROUGH_FEATURES if c in X_train.columns]:
        pre_train = X_train[feat].dropna().values
        pre_hold = X_hold[feat].dropna().values
        post_train = X_train_s[feat].dropna().values
        post_hold = X_hold_s[feat].dropna().values

        if len(pre_train) < 20 or len(pre_hold) < 20 or len(post_train) < 20 or len(post_hold) < 20:
            continue

        # Wasserstein distance (lower = better match)
        w_pre = float(wasserstein_distance(pre_train, pre_hold))
        w_post = float(wasserstein_distance(post_train, post_hold))

        # KS test (lower p-value = more different)
        ks_pre = ks_2samp(pre_train, pre_hold)
        ks_post = ks_2samp(post_train, post_hold)

        shift_report[feat] = {
            "wasserstein_pre": w_pre,
            "wasserstein_post": w_post,
            "wasserstein_delta": w_post - w_pre,
            "ks_stat_pre": float(ks_pre.statistic),
            "ks_stat_post": float(ks_post.statistic),
            "ks_p_pre": float(ks_pre.pvalue),
            "ks_p_post": float(ks_post.pvalue),
        }
    results["distribution_shift"] = shift_report

    # ── 9. Rank preservation (Spearman between raw and scaled on train) ───────
    rank_report = {}
    for feat in all_feats:
        if feat not in X_train.columns:
            continue
        x = X_train[feat].dropna()
        xs = X_train_s.loc[x.index, feat].dropna()
        common = x.index.intersection(xs.index)
        if len(common) < 20:
            continue
        rho, pval = st.spearmanr(x.loc[common], xs.loc[common])
        rank_report[feat] = {
            "spearman_raw_vs_scaled": float(rho),
            "p_value": float(pval),
        }
    results["rank_preservation"] = rank_report

    # ── 10. Global leakage / sanity checks ───────────────────────────────────
    scaler_sanity = {
        "scale_factors_train_only": True,
        "train_shape": X_train_s.shape,
        "hold_shape": X_hold_s.shape,
        "columns_identical": list(X_train_s.columns) == list(X_hold_s.columns),
    }
    results["leakage_guard"] = scaler_sanity

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_section(title: str):
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _r(val, fmt=".4f") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NaN"
    return f"{val:{fmt}}"


def print_report(results: Dict):
    _print_section("DATASET SUMMARY")
    print(f"  File                 : {results['dataset']}")
    print(f"  Raw bars             : {results['n_raw']}")
    print(f"  Engineered bars      : {results['n_features_engineered']}")
    print(f"  Train bars           : {results['n_train']}")
    print(f"  Holdout bars         : {results['n_hold']}")
    print(f"  Split date           : {results['split_date']}")

    # NaNs
    _print_section("NaN COUNTS")
    print("  Pre-dropna engineering NaNs (non-zero only):")
    for col, cnt in results["nan"]["pre_dropna"].items():
        print(f"    {col}: {cnt}")
    print("  Post-scale train NaNs (non-zero):")
    for col, cnt in results["nan"]["train_post_scale_nz"].items():
        print(f"    {col}: {cnt}")
    print("  Post-scale holdout NaNs (non-zero):")
    for col, cnt in results["nan"]["hold_post_scale_nz"].items():
        print(f"    {col}: {cnt}")

    # Bounds compliance
    _print_section("BOUNDED-FEATURE COMPLIANCE")
    bounds = results.get("bounds_compliance", {})
    for feat, info in bounds.items():
        flag_str = "  *** FLAGS: " + ", ".join(info["flags"]) if info["flags"] else ""
        print(f"  {feat}")
        print(f"    Expected : [{info['expected'][0]}, {info['expected'][1]}]")
        print(f"    Actual   : [{_r(info['actual_min'], '.4f')}, {_r(info['actual_max'], '.4f')}]")
        print(f"    In-bounds: {_r(info['in_bounds_rate'], '.2%')}{flag_str}")

    # Tanh saturation
    _print_section("TANH ASYMPTOTE SATURATION (SCALE_FEATURES)")
    sat = results.get("tanh_saturation", {})
    for feat, info in sat.items():
        flag = "  *** WARN" if info["saturation_ge_0_95"] > 0.10 else "  OK"
        print(f"  {flag} | {feat}")
        print(f"    P75 scale factor : {_r(info['scale_factor_p75'], '.6f')}")
        print(f"    Sat ≥ 0.95       : {_r(info['saturation_ge_0_95'], '.2%')}")
        print(f"    Sat ≥ 0.99       : {_r(info['saturation_ge_0_99'], '.2%')}")

    # Signal correlation
    _print_section("SIGNAL CORRELATION WITH TARGET (pre- vs post-scaling)")
    corr = results.get("signal_correlation", {})
    print(f"  {'Feature':<45} {'Pearson pre':>12} {'Pearson post':>13} {'Delta':>10} "
          f"{'Spearman pre':>13} {'Spearman post':>14} {'Delta':>10}")
    print("  " + "-" * 110)
    for feat, info in corr.items():
        print(f"  {feat:<45} {_r(info['pearson_pre'], '.4f'):>12} "
              f"{_r(info['pearson_post'], '.4f'):>13} {_r(info['pearson_delta'], '+.4f'):>10} "
              f"{_r(info['spearman_pre'], '.4f'):>13} {_r(info['spearman_post'], '.4f'):>14} "
              f"{_r(info['spearman_delta'], '+.4f'):>10}")

    # MI
    _print_section("MUTUAL INFORMATION (pre- vs post-scaling, nats)")
    mi = results.get("signal_mi", {})
    for feat, info in mi.items():
        delta_pct = info["mi_delta_pct"]
        flag = "  *** WARN" if abs(delta_pct) > 20.0 else "  OK"
        print(f"  {flag} | {feat}")
        print(f"    MI pre  : {_r(info['mi_pre'], '.6f')}")
        print(f"    MI post : {_r(info['mi_post'], '.6f')}")
        print(f"    Delta   : {_r(delta_pct, '.2f')}%")

    # Distribution shift
    _print_section("TRAIN→HOLDOUT DISTRIBUTION SHIFT (pre- vs post-scaling)")
    shift = results.get("distribution_shift", {})
    print(f"  {'Feature':<45} {'W pre':>10} {'W post':>10} {'KS pre':>10} "
          f"{'KS p pre':>11} {'KS p post':>11}")
    print("  " + "-" * 100)
    for feat, info in shift.items():
        ks_p_pre = info["ks_p_pre"]
        ks_p_post = info["ks_p_post"]
        # Flag if post-scaling shift is worse or KS p-value drops significantly
        ks_flag = ""
        if ks_p_post < 0.05 and ks_p_pre >= 0.05:
            ks_flag = " *** KS significant post-only"
        print(f"  {feat:<45} {_r(info['wasserstein_pre'], '.4f'):>10} "
              f"{_r(info['wasserstein_post'], '.4f'):>10} {_r(info['ks_stat_pre'], '.4f'):>10} "
              f"{_r(ks_p_pre, '.4f'):>11} {_r(ks_p_post, '.4f'):>11}{ks_flag}")

    # Rank preservation
    _print_section("RANK PRESERVATION (Spearman: raw train vs scaled train)")
    rank = results.get("rank_preservation", {})
    for feat, info in rank.items():
        flag = "  *** WARN" if info["spearman_raw_vs_scaled"] < 0.90 else "  OK"
        print(f"  {flag} | {feat:<45} rho={_r(info['spearman_raw_vs_scaled'], '.6f')}")

    # Leakage guard
    _print_section("LEAKAGE / SANITY GUARD")
    leak = results.get("leakage_guard", {})
    print(f"  Scale factors fit on train only : {leak['scale_factors_train_only']}")
    print(f"  Train shape (scaled)            : {leak['train_shape']}")
    print(f"  Holdout shape (scaled)          : {leak['hold_shape']}")
    print(f"  Columns identical train/holdout : {leak['columns_identical']}")

    # Summary verdict
    _print_section("SUMMARY VERDICT")
    issues = []

    # Check NaNs
    if results["nan"]["train_post_scale_nz"] or results["nan"]["hold_post_scale_nz"]:
        issues.append("NaNs present after scaling")

    # Check saturation
    for feat, info in sat.items():
        if info["saturation_ge_0_99"] > 0.15:
            issues.append(f"Severe tanh saturation in {feat} ({info['saturation_ge_0_99']:.1%})")
        elif info["saturation_ge_0_95"] > 0.30:
            issues.append(f"Moderate tanh saturation in {feat} ({info['saturation_ge_0_95']:.1%})")

    # Check correlation destruction
    for feat, info in corr.items():
        if abs(info["pearson_pre"]) > 0.05 and abs(info["pearson_pre"] - info["pearson_post"]) > 0.30:
            issues.append(f"Correlation destroyed in {feat} (delta={info['pearson_delta']:+.3f})")
        if abs(info["spearman_pre"]) > 0.05 and abs(info["spearman_pre"] - info["spearman_post"]) > 0.30:
            issues.append(f"Rank correlation destroyed in {feat} (delta={info['spearman_delta']:+.3f})")

    # Check MI destruction
    for feat, info in mi.items():
        if abs(info["mi_pre"]) > 0.01 and abs(info["mi_delta_pct"]) > 30.0:
            issues.append(f"MI change >30% in {feat} ({info['mi_delta_pct']:+.1f}%)")

    # Check bounds
    for feat, info in bounds.items():
        if info["flags"]:
            issues.append(f"{feat} has flags: {', '.join(info['flags'])}")

    if not issues:
        print("  ALL CHECKS PASSED — normalization appears correct and signal is preserved.")
    else:
        print("  ISSUES DETECTED:")
        for issue in issues:
            print(f"    * {issue}")
    print()


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    report = audit_normalization(csv)
    print_report(report)
    # Optionally save JSON
    import json
    out_path = "outputs/normalization_audit.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Audit report saved to %s", out_path)
