#!/usr/bin/env python3
"""
oracle_audit.py  —  Triple Barrier Method (TBM) Target Diagnostics
==================================================================
Standalone script to audit the TBM target generation logic.
Uses src.target_generator and src.config.

Reports
-------
Per data file and aggregate:
  • Total bars / valid signal bars
  • Long signals vs Short signals (count + %)
  • Exit type breakdown (TP, SL, Timeout, Both-SL, Wide-Candle)
  • Average holding period
"""

from __future__ import annotations
import sys, os, textwrap
# ── Ensure `src/` is resolvable regardless of working directory ───────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import numba

# ── import project modules from src ──────────────────────────────────────────
import src.config as config
from src.target_generator import run_first_touch_triple_barrier, EVENT_NAME_MAP


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (minimal — just OHLC + ATR)
# ─────────────────────────────────────────────────────────────────────────────

def _load_ohlc_with_atr(filepath: str) -> pd.DataFrame:
    """Load CSV and compute ATR — identical to target_generator logic."""
    df = pd.read_csv(filepath)

    # normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    for required in ("open", "high", "low", "close"):
        if required not in df.columns:
            raise ValueError(f"CSV missing required column: {required}")

    # Triple Barrier Method often uses Wilder ATR or SMA ATR. 
    # target_generator.py uses _compute_wilder_atr.
    # We'll use a simple SMA ATR here for diagnostic consistency if preferred, 
    # but for true audit we should match target_generator.
    
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Matching Wilder ATR logic from target_generator if possible, 
    # or just use the same window.
    df["atr"]  = true_range.rolling(config.ATR_PERIOD).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  N/A"
    return f"{100 * num / denom:5.1f}%"


def audit_one_file(filepath: str) -> dict:
    """Run TBM diagnostics on a single CSV. Returns summary dict."""
    df = _load_ohlc_with_atr(filepath)

    # Call the production TBM logic
    (targets, event_type, event_bar,
     entry_price, exit_price, atr_out, is_valid) = run_first_touch_triple_barrier(
        df["open"].values.astype(np.float32),
        df["high"].values.astype(np.float32),
        df["low"].values.astype(np.float32),
        df["close"].values.astype(np.float32),
        df["atr"].values.astype(np.float32),
        max_hold=config.ORACLE_MAX_HOLD,
        tp_mult=config.TP_ATR_MULT,
        sl_mult=config.SL_ATR_MULT,
    )

    # ── decompose ────────────────────────────────────────────────────────
    # The kernel returns arrays of length N.
    # We only care about rows where targets are NOT nan.
    mask = ~np.isnan(targets)
    targets    = targets[mask]
    event_type = event_type[mask]
    event_bar  = event_bar[mask]
    
    n_total = len(targets)
    n_long  = int((targets == 1.0).sum())
    n_short = int((targets == -1.0).sum())
    n_neutral = int((targets == 0.0).sum())
    n_whipsaw = int((targets == -99.0).sum())

    # Exit counts
    unique_events, event_counts = np.unique(event_type, return_counts=True)
    event_map = {unique_events[i]: event_counts[i] for i in range(len(unique_events))}
    
    # Event names from target_generator
    # EVENT_LONG_TP=1, EVENT_SHORT_TP=2, EVENT_LONG_SL=3, EVENT_SHORT_SL=4, 
    # EVENT_BOTH_TP=5, EVENT_BOTH_SL=6, EVENT_AMBIGUOUS=7, EVENT_TIMEOUT=8, EVENT_WIDE_CANDLE=9
    
    def get_count(code): return int(event_map.get(code, 0))

    stats = {
        "file": os.path.basename(filepath),
        "n_total": n_total,
        "n_long": n_long,
        "n_short": n_short,
        "n_neutral": n_neutral,
        "n_whipsaw": n_whipsaw,
        "signal_pct": (n_long + n_short) / max(1, n_total),
        
        "tp_count": get_count(1) + get_count(2),
        "sl_count": get_count(3) + get_count(4),
        "timeout_count": get_count(8),
        "both_tp_count": get_count(5),
        "both_sl_count": get_count(6),
        "ambiguous_count": get_count(7),
        "wide_candle_count": get_count(9),
        
        "avg_hold": 0.0  # Placeholder, calculated below
    }
    
    # Fix avg_hold: event_bar is the index where it hit. Entry was i+1.
    # So hold = event_bar - (indices + 1)
    # We use the original mask to get the correct indices for the rows we kept.
    full_indices = np.arange(len(mask))
    kept_indices = full_indices[mask]
    hold_periods = event_bar - (kept_indices + 1)
    
    # Only for valid events (event_bar != -1)
    valid_hold_mask = event_bar != -1
    if valid_hold_mask.any():
        stats["avg_hold"] = float(np.mean(hold_periods[valid_hold_mask]))
    else:
        stats["avg_hold"] = np.nan
    
    return stats


def _print_report(stats: dict) -> None:
    """Pretty-print one file's diagnostic report."""
    s = stats
    w = 60  # report width

    print("=" * w)
    print(f"  FILE: {s['file']}")
    print("=" * w)
    print(f"  Total valid decision bars : {s['n_total']:,}")
    print(f"  Labels:")
    print(f"    Long  (+1)      : {s['n_long']:,}  ({_pct(s['n_long'], s['n_total'])})")
    print(f"    Short (-1)      : {s['n_short']:,}  ({_pct(s['n_short'], s['n_total'])})")
    print(f"    Neutral (0)     : {s['n_neutral']:,}  ({_pct(s['n_neutral'], s['n_total'])})")
    print(f"    Whipsaw (-99)   : {s['n_whipsaw']:,}  ({_pct(s['n_whipsaw'], s['n_total'])})")
    print()

    print("  ── Event Type Breakdown ──")
    print(f"    TP (Long or Short)    : {s['tp_count']:,}  ({_pct(s['tp_count'], s['n_total'])})")
    print(f"    SL (Long or Short)    : {s['sl_count']:,}  ({_pct(s['sl_count'], s['n_total'])})")
    print(f"    Timeout               : {s['timeout_count']:,}  ({_pct(s['timeout_count'], s['n_total'])})")
    print(f"    Both TP (Ambiguous)   : {s['both_tp_count']:,}")
    print(f"    Both SL (Whipsaw)     : {s['both_sl_count']:,}")
    print(f"    Wide Candle           : {s['wide_candle_count']:,}")
    print(f"    Ambiguous             : {s['ambiguous_count']:,}")
    print()

    print("  ── Metrics ──")
    print(f"    Avg hold duration     : {s['avg_hold']:.1f} bars")
    print()


def main():
    # allow CLI override, otherwise use config.DATAPATH
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = [config.DATAPATH]

    print("\n" + "─" * 60)
    print("  TBM Target Audit (src.target_generator)")
    print("─" * 60)
    print(f"  ATR_PERIOD      = {config.ATR_PERIOD}")
    print(f"  ORACLE_MAX_HOLD = {config.ORACLE_MAX_HOLD}")
    print(f"  TP_ATR_MULT     = {config.TP_ATR_MULT}")
    print(f"  SL_ATR_MULT     = {config.SL_ATR_MULT}")
    print("─" * 60 + "\n")

    all_stats = []
    for f in files:
        if not os.path.exists(f):
            print(f"  ⚠️  File not found: {f}  — skipping.")
            continue
        print(f"  Processing: {f}")
        try:
            stats = audit_one_file(f)
            _print_report(stats)
            all_stats.append(stats)
        except Exception as e:
            print(f"  ❌ Error processing {f}: {e}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
