#!/usr/bin/env python3
"""Validate Fabio Valentino trading insights against backtest data.

Runs a 5y US backtest, then retroactively analyzes the trade data to
determine if each proposed improvement would have helped.

Insights tested:
  #1 - Trailing profit protection (would capping profit give-back help?)
  #3 - Day-of-week filter (are some weekdays consistently bad?)
  #5 - Win Rate vs R:R balance (what's our profit factor / avg win:loss ratio?)
  #7 - Consecutive loss breaker (do loss streaks cause outsized damage?)

Usage:
    python validate_fabio_insights.py
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime

import pandas as pd

from trading_engine.config import WATCHLIST
from trading_engine.price_engine import get_history
from trading_engine.time_machine import TimeMachineBacktest


def run_backtest() -> dict:
    """Run 5y US time-machine backtest and return results."""
    print("=" * 80)
    print("  FABIO INSIGHT VALIDATION — 5y US Backtest Analysis")
    print("=" * 80)

    print("\n  [1/2] Downloading US data (5y)...")
    t0 = time.time()
    data = {}
    for sym in WATCHLIST:
        df = get_history(sym, days=1500)
        if not df.empty:
            data[sym] = df
    print(f"  Downloaded {len(data)} stocks in {time.time() - t0:.0f}s")

    print("\n  [2/2] Running time-machine backtest...")
    t0 = time.time()
    tm = TimeMachineBacktest(
        market="us", initial_capital=10_000,
        confidence_threshold=0.80,
        max_position_pct=0.15,
        stop_loss_pct=0.03, take_profit_pct=0.05,
        session_id="fabio_validation",
        enable_learning=True,
    )
    results = tm.run(history_data=data)
    print(f"  Backtest done in {time.time() - t0:.0f}s")
    print(f"  Return: {results['return_pct']:+.2f}% | "
          f"Trades: {results['total_trades']} | "
          f"Closed: {results['closed_trades']} | "
          f"Win rate: {results['win_rate']:.1f}%")
    return results


def validate_day_of_week(trades: list[dict]) -> dict:
    """Insight #3: Day-of-week filter.

    Groups closed trades by day-of-week, computes win rate per day,
    then simulates removing the worst day(s).
    """
    closed = [t for t in trades if t.get("net_pnl") is not None]
    if not closed:
        return {"verdict": "SKIP", "reason": "No closed trades"}

    dow_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]

    for t in closed:
        try:
            dt = datetime.strptime(t["date"][:10], "%Y-%m-%d")
        except (ValueError, KeyError):
            continue
        dow = dt.weekday()
        if dow >= 5:
            continue
        dow_stats[dow]["count"] += 1
        dow_stats[dow]["pnl"] += t["net_pnl"]
        if t["net_pnl"] > 0:
            dow_stats[dow]["wins"] += 1
        else:
            dow_stats[dow]["losses"] += 1

    print("\n" + "=" * 80)
    print("  INSIGHT #3: Day-of-Week Filter")
    print("  Fabio: 'Remove Fridays — you always lose money'")
    print("=" * 80)

    total_pnl = sum(s["pnl"] for s in dow_stats.values())

    print(f"\n  {'Day':<6} {'Trades':>7} {'Wins':>6} {'Losses':>7} {'Win%':>7} {'PnL':>10}")
    print(f"  {'-'*48}")

    worst_day = None
    worst_wr = 1.0
    for dow in range(5):
        s = dow_stats[dow]
        if s["count"] == 0:
            continue
        wr = s["wins"] / s["count"]
        print(f"  {dow_names[dow]:<6} {s['count']:>7} {s['wins']:>6} {s['losses']:>7} "
              f"{wr:>6.1%} {s['pnl']:>+10,.2f}")
        if wr < worst_wr and s["count"] >= 3:
            worst_wr = wr
            worst_day = dow

    if worst_day is not None:
        ws = dow_stats[worst_day]
        pnl_without = total_pnl - ws["pnl"]
        print(f"\n  Worst day: {dow_names[worst_day]} (win rate: {worst_wr:.1%}, PnL: {ws['pnl']:+,.2f})")
        print(f"  Total PnL with all days:    {total_pnl:>+10,.2f}")
        print(f"  Total PnL without {dow_names[worst_day]:>3}:     {pnl_without:>+10,.2f}")
        print(f"  Delta:                      {pnl_without - total_pnl:>+10,.2f}")

        would_help = pnl_without > total_pnl
        verdict = "VALIDATED" if would_help else "NOT VALIDATED"
        print(f"\n  VERDICT: {verdict}")
        if would_help:
            print(f"  Removing {dow_names[worst_day]} would have improved PnL by "
                  f"${pnl_without - total_pnl:+,.2f}")
        else:
            print(f"  Removing {dow_names[worst_day]} would NOT have improved PnL")

        return {
            "verdict": verdict,
            "worst_day": dow_names[worst_day],
            "worst_wr": worst_wr,
            "total_pnl": total_pnl,
            "pnl_without_worst": pnl_without,
            "delta": pnl_without - total_pnl,
            "day_stats": {dow_names[d]: dict(s) for d, s in dow_stats.items()},
        }

    return {"verdict": "SKIP", "reason": "Not enough data per day"}


def validate_win_rate_rr(trades: list[dict]) -> dict:
    """Insight #5: Win Rate vs R:R Balance.

    Computes profit factor, average win/loss ratio, and compares
    to Fabio's target: 43-49% WR with 1.7x avg win:loss.
    """
    closed = [t for t in trades if t.get("net_pnl") is not None]
    if not closed:
        return {"verdict": "SKIP", "reason": "No closed trades"}

    wins = [t for t in closed if t["net_pnl"] > 0]
    losses = [t for t in closed if t["net_pnl"] <= 0]

    total_win_pnl = sum(t["net_pnl"] for t in wins)
    total_loss_pnl = abs(sum(t["net_pnl"] for t in losses))

    avg_win = total_win_pnl / len(wins) if wins else 0
    avg_loss = total_loss_pnl / len(losses) if losses else 0
    profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float("inf")
    rr_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")
    win_rate = len(wins) / len(closed) * 100

    print("\n" + "=" * 80)
    print("  INSIGHT #5: Win Rate vs R:R Balance")
    print("  Fabio: '43-49% WR with avg win = 1.7x avg loss'")
    print("=" * 80)

    print(f"\n  Current Performance:")
    print(f"    Win rate:           {win_rate:.1f}%")
    print(f"    Avg win:            ${avg_win:+,.2f}")
    print(f"    Avg loss:           ${avg_loss:,.2f}")
    print(f"    Win:Loss ratio:     {rr_ratio:.2f}x")
    print(f"    Profit factor:      {profit_factor:.2f}")
    print(f"    Total wins PnL:     ${total_win_pnl:+,.2f}")
    print(f"    Total losses PnL:   ${-total_loss_pnl:+,.2f}")

    print(f"\n  Fabio's Target:")
    print(f"    Win rate:           43-49%")
    print(f"    Win:Loss ratio:     1.7x")
    print(f"    Profit factor:      ~1.3-1.6")

    issues = []
    if rr_ratio < 1.0:
        issues.append(f"R:R ratio {rr_ratio:.2f}x < 1.0 — avg loss bigger than avg win")
    if rr_ratio < 1.7 and win_rate < 50:
        issues.append(f"R:R {rr_ratio:.2f}x too low for {win_rate:.0f}% WR — need tighter SL or wider TP")
    if profit_factor < 1.0:
        issues.append(f"Profit factor {profit_factor:.2f} < 1.0 — system is losing money")
    if profit_factor >= 1.0 and rr_ratio >= 1.5:
        issues.append(f"R:R {rr_ratio:.2f}x is healthy — close to Fabio's 1.7x target")

    # Simulate ATR-based dynamic SL/TP: tighter SL (2% instead of 3%)
    sim_wins = 0
    sim_pnl = 0.0
    for t in closed:
        pnl_pct = t.get("pnl_pct", 0)
        if pnl_pct is not None:
            # With tighter 2% SL, losses that were -3% would be capped at -2%
            if pnl_pct < -2.0:
                adjusted_pnl = t["net_pnl"] * (2.0 / abs(pnl_pct))
            else:
                adjusted_pnl = t["net_pnl"]
            sim_pnl += adjusted_pnl
            if adjusted_pnl > 0:
                sim_wins += 1

    current_pnl = sum(t["net_pnl"] for t in closed)
    tighter_sl_helps = sim_pnl > current_pnl

    print(f"\n  Simulation — Tighter SL (2% vs 3%):")
    print(f"    Current total PnL:  ${current_pnl:+,.2f}")
    print(f"    Simulated PnL:      ${sim_pnl:+,.2f}")
    print(f"    Delta:              ${sim_pnl - current_pnl:+,.2f}")

    verdict = "VALIDATED" if (rr_ratio < 1.5 or tighter_sl_helps) else "NOT VALIDATED"
    print(f"\n  VERDICT: {verdict}")
    if issues:
        for issue in issues:
            print(f"    - {issue}")

    return {
        "verdict": verdict,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "rr_ratio": rr_ratio,
        "profit_factor": profit_factor,
        "tighter_sl_pnl": sim_pnl,
        "current_pnl": current_pnl,
        "tighter_sl_helps": tighter_sl_helps,
        "issues": issues,
    }


def validate_consecutive_loss_breaker(trades: list[dict]) -> dict:
    """Insight #7: Consecutive loss circuit breaker.

    Walks through closed trades chronologically, detects loss streaks,
    and simulates halting after N consecutive losses.
    """
    closed = [t for t in trades if t.get("net_pnl") is not None]
    if not closed:
        return {"verdict": "SKIP", "reason": "No closed trades"}

    # Sort by exit date
    closed_sorted = sorted(closed, key=lambda t: t.get("date", ""))

    # Group by day
    day_trades = defaultdict(list)
    for t in closed_sorted:
        day_trades[t["date"][:10]].append(t)

    print("\n" + "=" * 80)
    print("  INSIGHT #7: Consecutive Loss Circuit Breaker")
    print("  Fabio: 'Days with 3-5 stop losses end up losing $15-20K'")
    print("=" * 80)

    # Analyze loss streaks across all trades
    streaks = []
    current_streak = 0
    streak_pnl = 0.0
    for t in closed_sorted:
        if t["net_pnl"] <= 0:
            current_streak += 1
            streak_pnl += t["net_pnl"]
        else:
            if current_streak >= 2:
                streaks.append({"length": current_streak, "pnl": streak_pnl})
            current_streak = 0
            streak_pnl = 0.0
    if current_streak >= 2:
        streaks.append({"length": current_streak, "pnl": streak_pnl})

    print(f"\n  Loss Streak Analysis ({len(closed_sorted)} closed trades):")
    if not streaks:
        print(f"    No significant loss streaks detected (all < 2)")
        return {"verdict": "SKIP", "reason": "No loss streaks detected"}

    streak_counts = defaultdict(int)
    streak_pnls = defaultdict(float)
    for s in streaks:
        streak_counts[s["length"]] += 1
        streak_pnls[s["length"]] += s["pnl"]

    print(f"\n  {'Streak':>8} {'Count':>7} {'Total PnL':>12} {'Avg PnL':>12}")
    print(f"  {'-'*42}")
    for length in sorted(streak_counts.keys()):
        avg = streak_pnls[length] / streak_counts[length]
        print(f"  {length:>5}x L {streak_counts[length]:>7} {streak_pnls[length]:>+12,.2f} {avg:>+12,.2f}")

    # Simulate: halt after N consecutive losses
    total_pnl = sum(t["net_pnl"] for t in closed_sorted)
    best_n = None
    best_pnl = total_pnl

    for halt_after in [2, 3, 4, 5]:
        sim_pnl = 0.0
        consecutive_losses = 0
        halted = False
        current_day = None
        for t in closed_sorted:
            trade_day = t["date"][:10]
            if trade_day != current_day:
                current_day = trade_day
                halted = False
                consecutive_losses = 0

            if halted:
                continue

            sim_pnl += t["net_pnl"]
            if t["net_pnl"] <= 0:
                consecutive_losses += 1
                if consecutive_losses >= halt_after:
                    halted = True
            else:
                consecutive_losses = 0

        if sim_pnl > best_pnl:
            best_pnl = sim_pnl
            best_n = halt_after

    print(f"\n  Simulation — Halt After N Consecutive Losses (per day):")
    print(f"  {'Halt After':>12} {'Simulated PnL':>14} {'Delta':>10}")
    print(f"  {'-'*40}")
    print(f"  {'No halt':>12} {total_pnl:>+14,.2f} {'baseline':>10}")

    for halt_after in [2, 3, 4, 5]:
        sim_pnl = 0.0
        consecutive_losses = 0
        halted = False
        current_day = None
        skipped = 0
        for t in closed_sorted:
            trade_day = t["date"][:10]
            if trade_day != current_day:
                current_day = trade_day
                halted = False
                consecutive_losses = 0
            if halted:
                skipped += 1
                continue
            sim_pnl += t["net_pnl"]
            if t["net_pnl"] <= 0:
                consecutive_losses += 1
                if consecutive_losses >= halt_after:
                    halted = True
            else:
                consecutive_losses = 0
        delta = sim_pnl - total_pnl
        print(f"  {halt_after:>9}x L {sim_pnl:>+14,.2f} {delta:>+10,.2f}  (skipped {skipped})")

    verdict = "VALIDATED" if best_n is not None else "NOT VALIDATED"
    print(f"\n  VERDICT: {verdict}")
    if best_n:
        print(f"    Halting after {best_n} consecutive losses would have improved PnL by "
              f"${best_pnl - total_pnl:+,.2f}")
    else:
        print(f"    No halt-after-N threshold improves total PnL")

    return {
        "verdict": verdict,
        "total_pnl": total_pnl,
        "best_halt_after": best_n,
        "best_pnl": best_pnl,
        "improvement": best_pnl - total_pnl if best_n else 0,
        "streaks": streaks,
    }


def validate_trailing_profit(daily_results: list[dict]) -> dict:
    """Insight #1: Trailing profit protection.

    Analyzes daily results to detect days where accumulated profit
    was given back, simulates trailing profit protection.
    """
    if not daily_results:
        return {"verdict": "SKIP", "reason": "No daily results"}

    print("\n" + "=" * 80)
    print("  INSIGHT #1: Trailing Profit Protection")
    print("  Fabio: 'Once up $20K, I'm not giving it back'")
    print("=" * 80)

    initial = daily_results[0]["total_value"] if daily_results else 10000
    actual_final = daily_results[-1]["total_value"] if daily_results else initial

    # Track peak daily PnL and give-back events
    cum_pnl = 0.0
    peak_pnl = 0.0
    giveback_events = []
    daily_pnls = []

    for i, day in enumerate(daily_results):
        day_pnl = day.get("pnl", 0)
        daily_pnls.append(day_pnl)
        cum_pnl = day["total_value"] - initial
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        giveback = peak_pnl - cum_pnl
        if giveback > 0 and peak_pnl > 0:
            giveback_pct = giveback / peak_pnl * 100 if peak_pnl > 0 else 0
            if giveback_pct > 30:
                giveback_events.append({
                    "day_idx": i,
                    "date": day.get("date", f"day_{i}"),
                    "peak_pnl": peak_pnl,
                    "current_pnl": cum_pnl,
                    "giveback": giveback,
                    "giveback_pct": giveback_pct,
                })

    print(f"\n  Portfolio Performance:")
    print(f"    Initial capital:    ${initial:,.0f}")
    print(f"    Final value:        ${actual_final:,.0f}")
    print(f"    Peak cumulative PnL: ${peak_pnl:+,.2f}")
    print(f"    Final cumulative PnL: ${cum_pnl:+,.2f}")
    print(f"    Giveback events (>30% of peak): {len(giveback_events)}")

    if giveback_events:
        print(f"\n  {'Date':<12} {'Peak PnL':>10} {'Current':>10} {'Giveback':>10} {'%':>6}")
        print(f"  {'-'*52}")
        for ev in giveback_events[:10]:
            print(f"  {str(ev['date']):<12} {ev['peak_pnl']:>+10,.2f} {ev['current_pnl']:>+10,.2f} "
                  f"{ev['giveback']:>10,.2f} {ev['giveback_pct']:>5.1f}%")

    # Simulate: risk max 50% of accumulated profit
    sim_value = initial
    sim_peak_pnl = 0.0
    protection_triggers = 0

    for day in daily_results:
        day_pnl = day.get("pnl", 0)
        sim_pnl = sim_value - initial

        if sim_pnl > 0:
            sim_peak_pnl = max(sim_peak_pnl, sim_pnl)
            max_risk = sim_pnl * 0.5
            if day_pnl < -max_risk and day_pnl < 0:
                day_pnl = -max_risk
                protection_triggers += 1

        sim_value += day_pnl

    print(f"\n  Simulation — Risk Max 50% of Accumulated Profit:")
    print(f"    Actual final:       ${actual_final:,.2f}")
    print(f"    Simulated final:    ${sim_value:,.2f}")
    print(f"    Delta:              ${sim_value - actual_final:+,.2f}")
    print(f"    Protection triggers: {protection_triggers}")

    verdict = "VALIDATED" if sim_value > actual_final else "NOT VALIDATED"
    print(f"\n  VERDICT: {verdict}")

    return {
        "verdict": verdict,
        "actual_final": actual_final,
        "simulated_final": sim_value,
        "delta": sim_value - actual_final,
        "giveback_events": len(giveback_events),
        "protection_triggers": protection_triggers,
    }


def main():
    results = run_backtest()

    trades = results.get("trades", [])
    daily = results.get("daily_results", [])

    print(f"\n  Analyzing {len(trades)} trades and {len(daily)} daily results...")

    v1 = validate_trailing_profit(daily)
    v3 = validate_day_of_week(trades)
    v5 = validate_win_rate_rr(trades)
    v7 = validate_consecutive_loss_breaker(trades)

    print("\n" + "=" * 80)
    print("  SUMMARY — Fabio Insight Validation")
    print("=" * 80)
    # Build summary notes (avoid nested f-string issues)
    if v1["verdict"] != "SKIP":
        delta1 = v1.get("delta", 0)
        note1 = f"Delta: ${delta1:+,.2f}"
    else:
        note1 = v1.get("reason", "")

    if v3["verdict"] != "SKIP":
        wd = v3.get("worst_day", "N/A")
        wwr = v3.get("worst_wr", 0)
        note3 = f"Worst: {wd} ({wwr:.0%} WR)"
    else:
        note3 = v3.get("reason", "")

    if v5["verdict"] != "SKIP":
        rr = v5.get("rr_ratio", 0)
        pf = v5.get("profit_factor", 0)
        note5 = f"R:R={rr:.2f}x PF={pf:.2f}"
    else:
        note5 = v5.get("reason", "")

    if v7["verdict"] != "SKIP":
        bh = v7.get("best_halt_after", "N/A")
        note7 = f"Best halt={bh}x"
    else:
        note7 = v7.get("reason", "")

    print(f"\n  {'#':<4} {'Insight':<35} {'Verdict':<18} {'Notes'}")
    print(f"  {'-'*80}")
    print(f"  {'1':<4} {'Trailing profit protection':<35} {v1['verdict']:<18} {note1}")
    print(f"  {'3':<4} {'Day-of-week filter':<35} {v3['verdict']:<18} {note3}")
    print(f"  {'5':<4} {'Win Rate vs R:R balance':<35} {v5['verdict']:<18} {note5}")
    print(f"  {'7':<4} {'Consecutive loss breaker':<35} {v7['verdict']:<18} {note7}")

    validated = sum(1 for v in [v1, v3, v5, v7] if v["verdict"] == "VALIDATED")
    total = sum(1 for v in [v1, v3, v5, v7] if v["verdict"] != "SKIP")
    print(f"\n  Result: {validated}/{total} insights validated as beneficial")
    print(f"  Only validated insights should be considered for implementation.")
    print("=" * 80)


if __name__ == "__main__":
    main()
