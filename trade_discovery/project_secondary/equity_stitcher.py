"""
equity_stitcher.py — Stitches OOS portfolio returns across winning WFO folds
into a single equity curve and computes composite performance metrics.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def _calmar_ratio(equity: pd.Series, periods_per_year: int = 252 * 375) -> float:
    """Calmar = CAGR / Max Drawdown. Uses minute-bar count for NSE."""
    total_return  = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    n_bars        = len(equity)
    years         = n_bars / periods_per_year
    cagr          = (1 + total_return) ** (1 / max(years, 1e-6)) - 1
    rolling_max   = equity.cummax()
    drawdown      = (equity - rolling_max) / rolling_max
    max_dd        = float(drawdown.min())
    return cagr / abs(max_dd) if max_dd != 0 else np.inf


def stitch_equity_curves(
    winning_formulas:  list,
    vectorbt_stats_dir: str = "outputs/vectorbt_stats",
    output_dir:         str = "outputs",
) -> pd.DataFrame | None:
    """
    Loads per-fold winner CSVs, extracts 'Total Return [%]' as a proxy
    daily NAV point, and stitches them into a compound equity curve.

    Returns a DataFrame with columns: [fold, return_pct, nav, drawdown]
    Saves: outputs/combined_equity.parquet + outputs/combined_equity.csv
    """
    records = []
    nav     = 1.0

    for w in sorted(winning_formulas,
                    key=lambda x: x.get('fold', x.get('gen', 0))):
        fold      = w.get('fold', w.get('gen', 0))
        fold_ret  = w['return_pct'] / 100.0      # fractional return for this OOS window
        nav_end   = nav * (1 + fold_ret)

        records.append({
            'fold':       fold,
            'return_pct': w['return_pct'],
            'sharpe':     w['sharpe'],
            'max_dd':     w['max_dd'],
            'win_rate':   w['win_rate'],
            'nav_start':  nav,
            'nav_end':    nav_end,
        })
        nav = nav_end

    if not records:
        return None

    df = pd.DataFrame(records)

    # Compound equity series (one point per fold)
    equity        = df['nav_end']
    rolling_max   = equity.cummax()
    df['drawdown'] = (equity - rolling_max) / rolling_max

    # Composite metrics
    composite_return  = (nav - 1.0) * 100.0
    composite_sharpe  = df['sharpe'].mean()
    worst_dd          = float(df['drawdown'].min() * 100)
    avg_win_rate      = df['win_rate'].mean()
    calmar            = _calmar_ratio(equity, periods_per_year=len(df))

    print("\n" + "=" * 55)
    print("  COMPOSITE WALK-FORWARD PERFORMANCE REPORT")
    print("=" * 55)
    print(f"  Winning Folds      : {len(df)}")
    print(f"  Compound Return    : {composite_return:>8.2f}%")
    print(f"  Avg OOS Sharpe     : {composite_sharpe:>8.2f}")
    print(f"  Worst Fold DD      : {worst_dd:>8.2f}%")
    print(f"  Avg Win Rate       : {avg_win_rate:>8.1f}%")
    print(f"  Calmar Ratio       : {calmar:>8.2f}")
    print("=" * 55 + "\n")

    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    df.to_parquet(str(out_path / "combined_equity.parquet"), index=False)
    df.to_csv(str(out_path / "combined_equity.csv"), index=False)

    return df
