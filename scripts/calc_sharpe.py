#!/usr/bin/env python3
"""Calculate Sharpe ratio from trade_journal.db daily snapshots."""

import sqlite3
import math
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "trade_journal.db"


def main():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Check tables
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    print(f"Tables: {tables}\n")

    # Check daily_snapshots (table may be named daily_snapshots_tm)
    snap_table = None
    for t in tables:
        if "daily_snapshot" in t:
            snap_table = t
            break
    if not snap_table:
        print("ERROR: no daily_snapshots table found")
        conn.close()
        return

    print(f"Using table: {snap_table}")
    # First check schema
    schema = conn.execute(f"PRAGMA table_info({snap_table})").fetchall()
    col_names = [r[1] for r in schema]
    print(f"Columns: {col_names}")

    df = pd.read_sql(f"SELECT * FROM {snap_table}", conn)
    conn.close()

    print(f"Total snapshot rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}\n")

    if df.empty:
        print("No data in daily_snapshots.")
        return

    # Group by market + session
    for (market, sid), g in df.groupby(["market", "session_id"]):
        g = g.sort_values("date").reset_index(drop=True)
        total_vals = g["total_value"].astype(float)
        daily_ret = total_vals.pct_change().dropna()

        days = len(g)
        start_val = total_vals.iloc[0]
        end_val = total_vals.iloc[-1]
        total_return = (end_val / start_val - 1) * 100

        # Max drawdown
        cummax = total_vals.cummax()
        drawdown = (total_vals / cummax - 1)
        max_dd = drawdown.min() * 100

        # Sharpe
        if len(daily_ret) < 2 or daily_ret.std() == 0:
            sharpe = 0.0
        else:
            ann_factor = 365 if market == "crypto" else 252
            sharpe = (daily_ret.mean() / daily_ret.std()) * math.sqrt(ann_factor)

        # Win days vs loss days
        daily_pnl = g["daily_pnl"].astype(float)
        win_days = (daily_pnl > 0).sum()
        loss_days = (daily_pnl < 0).sum()
        flat_days = (daily_pnl == 0).sum()

        print(f"{'='*60}")
        print(f"Market: {market} | Session: {sid}")
        print(f"{'='*60}")
        print(f"  Period:     {g['date'].iloc[0]} \u2192 {g['date'].iloc[-1]} ({days} days)")
        print(f"  Capital:    ${start_val:,.2f} → ${end_val:,.2f}")
        print(f"  Return:     {total_return:+.2f}%")
        print(f"  Max DD:     {max_dd:.2f}%")
        print(f"  Sharpe:     {sharpe:.2f}")
        print(f"  Day stats:  {win_days} win / {loss_days} loss / {flat_days} flat")
        print()


def summary():
    """Print a ranked summary table sorted by Sharpe ratio."""
    conn = sqlite3.connect(str(DB_PATH))
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    snap_table = next((t for t in tables if "daily_snapshot" in t), None)
    if not snap_table:
        print("No snapshot table")
        return

    df = pd.read_sql(f"SELECT * FROM {snap_table}", conn)
    conn.close()

    rows = []
    for (market, sid), g in df.groupby(["market", "session_id"]):
        g = g.sort_values("date").reset_index(drop=True)
        total_vals = g["total_value"].astype(float)
        daily_ret = total_vals.pct_change().dropna()
        days = len(g)
        start_val = total_vals.iloc[0]
        end_val = total_vals.iloc[-1]
        total_return = (end_val / start_val - 1) * 100
        cummax = total_vals.cummax()
        max_dd = ((total_vals / cummax - 1).min()) * 100
        if len(daily_ret) < 2 or daily_ret.std() == 0:
            sharpe = 0.0
        else:
            ann_factor = 365 if market == "crypto" else 252
            sharpe = (daily_ret.mean() / daily_ret.std()) * math.sqrt(ann_factor)
        rows.append({
            "Market": market, "Session": sid, "Days": days,
            "Return%": round(total_return, 2),
            "MaxDD%": round(max_dd, 2),
            "Sharpe": round(sharpe, 2),
        })

    result = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    print("\n" + "=" * 80)
    print("SHARPE RATIO RANKING (all sessions, sorted best → worst)")
    print("=" * 80)
    print(result.to_string(index=False))
    print(f"\nTotal sessions: {len(result)}")
    print(f"Positive Sharpe: {(result['Sharpe'] > 0).sum()}")
    print(f"Sharpe > 0.5 (acceptable): {(result['Sharpe'] > 0.5).sum()}")
    print(f"Sharpe > 1.0 (good): {(result['Sharpe'] > 1.0).sum()}")


if __name__ == "__main__":
    import sys
    if "--summary" in sys.argv:
        summary()
    else:
        main()
