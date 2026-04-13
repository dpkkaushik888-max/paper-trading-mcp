"""Strategy Autoresearch — Karpathy's autoresearch pattern for rules.json.

Constraint: rules.json must produce valid entry/exit signals
Mechanical metric: composite backtest score (return + win rate + cost efficiency + Sharpe)
Iteration: mutate ONE parameter → backtest → keep if improved, revert if worse
Guard rail: no single metric drops >5%

Usage:
    python -m trading_engine.strategy_optimizer --iterations 20 --days 90
"""

from __future__ import annotations

import copy
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

from .backtest import run_backtest
from .config import PROJECT_ROOT, WATCHLIST
from .price_engine import get_history


RESULTS_DIR = PROJECT_ROOT / "optimization_results"

MAX_METRIC_DROP = 5.0

METRIC_WEIGHTS = {
    "net_return_pct": 0.30,
    "win_rate": 0.25,
    "cost_efficiency": 0.20,
    "sharpe_proxy": 0.25,
}

MUTATION_SPACE = {
    "entry_rsi_threshold": {
        "path": ["entry_rules", "long", 0, "condition"],
        "template": "rsi_14 < {value}",
        "values": [25, 30, 35, 40, 45, 50],
        "description": "RSI oversold threshold",
    },
    "entry_rsi_weight": {
        "path": ["entry_rules", "long", 0, "weight"],
        "values": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        "description": "RSI rule weight",
    },
    "entry_ema_period": {
        "path": ["entry_rules", "long", 1, "condition"],
        "template": "price > ema_{value}",
        "values": [20, 50, 200],
        "description": "EMA trend period",
    },
    "entry_ema_weight": {
        "path": ["entry_rules", "long", 1, "weight"],
        "values": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        "description": "EMA rule weight",
    },
    "entry_macd_weight": {
        "path": ["entry_rules", "long", 2, "weight"],
        "values": [0.10, 0.15, 0.20, 0.25, 0.30],
        "description": "MACD rule weight",
    },
    "entry_bb_weight": {
        "path": ["entry_rules", "long", 3, "weight"],
        "values": [0.10, 0.15, 0.20, 0.25, 0.30],
        "description": "Bollinger Band rule weight",
    },
    "entry_min_score": {
        "path": ["entry_rules", "min_score"],
        "values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "description": "Minimum score to trigger entry",
    },
    "exit_rsi_threshold": {
        "path": ["exit_rules", 0, "condition"],
        "template": "rsi_14 > {value}",
        "values": [60, 65, 70, 75, 80],
        "description": "RSI overbought exit threshold",
    },
    "exit_ema_period": {
        "path": ["exit_rules", 1, "condition"],
        "template": "price < ema_{value}",
        "values": [20, 50, 200],
        "description": "EMA exit period",
    },
    "stop_loss_pct": {
        "path": ["risk_rules", "stop_loss_pct"],
        "values": [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
        "description": "Stop loss percentage",
    },
    "take_profit_pct": {
        "path": ["risk_rules", "take_profit_pct"],
        "values": [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10],
        "description": "Take profit percentage",
    },
}


def _get_nested(obj: dict, path: list):
    """Get value from nested dict/list by path."""
    current = obj
    for key in path:
        current = current[key]
    return current


def _set_nested(obj: dict, path: list, value):
    """Set value in nested dict/list by path."""
    current = obj
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def score_backtest(result: dict) -> dict:
    """Convert backtest results into a composite score (0-100).

    Dimensions:
    - net_return_pct: return on capital (higher = better)
    - win_rate: % of profitable closed trades (higher = better)
    - cost_efficiency: 100 - cost_drag_pct (lower drag = better)
    - sharpe_proxy: return / volatility approximation
    """
    if "error" in result:
        return {
            "composite": 0, "net_return_pct": 0, "win_rate": 0,
            "cost_efficiency": 0, "sharpe_proxy": 0, "error": result["error"],
        }

    net_return = result.get("return_pct", 0)
    net_return_score = max(0, min(100, 50 + net_return * 5))

    win_rate = result.get("win_rate", 0)
    win_rate_score = min(100, win_rate)

    cost_drag = result.get("cost_drag_pct", 100)
    cost_efficiency_score = max(0, 100 - cost_drag)

    daily_results = result.get("daily_results", [])
    if daily_results:
        daily_pnls = [d["daily_pnl_net"] for d in daily_results]
        avg_pnl = sum(daily_pnls) / len(daily_pnls) if daily_pnls else 0
        std_pnl = (sum((p - avg_pnl) ** 2 for p in daily_pnls) / len(daily_pnls)) ** 0.5 if daily_pnls else 1
        sharpe = (avg_pnl / std_pnl) if std_pnl > 0 else 0
        sharpe_score = max(0, min(100, 50 + sharpe * 25))
    else:
        sharpe_score = 0

    total_trades = result.get("total_trades", 0)
    if total_trades == 0:
        return {
            "composite": 5, "net_return_pct": net_return_score, "win_rate": 0,
            "cost_efficiency": 100, "sharpe_proxy": 50,
            "note": "No trades generated — rules too restrictive",
        }

    composite = (
        METRIC_WEIGHTS["net_return_pct"] * net_return_score +
        METRIC_WEIGHTS["win_rate"] * win_rate_score +
        METRIC_WEIGHTS["cost_efficiency"] * cost_efficiency_score +
        METRIC_WEIGHTS["sharpe_proxy"] * sharpe_score
    )

    return {
        "composite": round(composite, 2),
        "net_return_pct": round(net_return_score, 2),
        "win_rate": round(win_rate_score, 2),
        "cost_efficiency": round(cost_efficiency_score, 2),
        "sharpe_proxy": round(sharpe_score, 2),
        "raw": {
            "return_pct": result.get("return_pct", 0),
            "win_rate": result.get("win_rate", 0),
            "cost_drag_pct": result.get("cost_drag_pct", 0),
            "total_trades": total_trades,
            "net_pnl": result.get("total_pnl_net", 0),
        },
    }


def mutate_rules(rules: dict, mutation_key: Optional[str] = None) -> tuple[dict, str, str]:
    """Apply ONE random mutation to the rules.

    Returns (mutated_rules, mutation_key, description).
    """
    new_rules = copy.deepcopy(rules)

    if mutation_key is None:
        mutation_key = random.choice(list(MUTATION_SPACE.keys()))

    mutation = MUTATION_SPACE[mutation_key]
    path = mutation["path"]
    values = mutation["values"]

    current_val = _get_nested(new_rules, path)
    available = [v for v in values if _format_value(mutation, v) != current_val]
    if not available:
        available = values

    new_val = random.choice(available)
    formatted = _format_value(mutation, new_val)
    _set_nested(new_rules, path, formatted)

    description = f"{mutation['description']}: {current_val} → {formatted}"
    return new_rules, mutation_key, description


def _format_value(mutation: dict, value) -> str | float:
    """Format a value using template if present."""
    if "template" in mutation:
        return mutation["template"].format(value=value)
    return value


def check_guard_rails(baseline_scores: dict, new_scores: dict) -> dict:
    """Check no single metric dropped more than MAX_METRIC_DROP."""
    violations = []
    for key in METRIC_WEIGHTS:
        old = baseline_scores.get(key, 0)
        new = new_scores.get(key, 0)
        drop = old - new
        if drop > MAX_METRIC_DROP:
            violations.append(f"{key}: {old:.1f} → {new:.1f} (dropped {drop:.1f}, max {MAX_METRIC_DROP})")
    return {"passed": len(violations) == 0, "violations": violations}


def run_optimization(
    iterations: int = 20,
    backtest_days: int = 90,
    initial_capital: float = 1000.0,
    broker: str = "etoro",
    account_currency: str = "EUR",
    rules_path: Optional[str] = None,
) -> dict:
    """Run the full autoresearch optimization loop.

    1. Load current rules.json
    2. Backtest → baseline score
    3. For each iteration:
       a. Mutate ONE parameter
       b. Write temp rules, backtest
       c. Score → keep if improved + guard rails pass, else revert
    4. Save best rules.json + optimization log
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rules_file = Path(rules_path) if rules_path else PROJECT_ROOT / "rules.json"
    with open(rules_file) as f:
        current_rules = json.load(f)

    print("=" * 60)
    print("  STRATEGY AUTORESEARCH")
    print(f"  Iterations: {iterations} | Backtest: {backtest_days} days")
    print(f"  Capital: ${initial_capital:,.0f} | Broker: {broker}")
    print("=" * 60)

    print("\n📊 Downloading historical data (one-time)...")
    cached_history = {}
    symbols = current_rules.get("watchlist", WATCHLIST)
    for symbol in symbols:
        df = get_history(symbol, days=backtest_days + 60)
        if not df.empty:
            cached_history[symbol] = df
            print(f"  {symbol}: {len(df)} days")

    print("\n📊 Measuring baseline...")
    temp_rules_path = RESULTS_DIR / "temp_rules.json"
    with open(temp_rules_path, "w") as f:
        json.dump(current_rules, f, indent=2)

    baseline_result = run_backtest(
        days=backtest_days, rules_path=str(temp_rules_path),
        initial_capital=initial_capital, broker=broker,
        account_currency=account_currency,
        cached_history=cached_history,
    )
    baseline_scores = score_backtest(baseline_result)

    print(f"  Composite: {baseline_scores['composite']:.1f}/100")
    print(f"  Return: {baseline_scores['net_return_pct']:.1f} | Win rate: {baseline_scores['win_rate']:.1f} | "
          f"Cost eff: {baseline_scores['cost_efficiency']:.1f} | Sharpe: {baseline_scores['sharpe_proxy']:.1f}")
    if "raw" in baseline_scores:
        raw = baseline_scores["raw"]
        print(f"  Raw: {raw['return_pct']:+.2f}% return, {raw['win_rate']:.0f}% wins, "
              f"{raw['total_trades']} trades, ${raw['net_pnl']:+.2f} net P&L")

    log = [{
        "iteration": 0,
        "status": "baseline",
        "mutation": "none",
        "description": "Initial baseline",
        "scores": baseline_scores,
        "rules_snapshot": _rules_fingerprint(current_rules),
    }]

    best_rules = copy.deepcopy(current_rules)
    best_scores = baseline_scores
    best_composite = baseline_scores["composite"]
    kept = 0
    discarded = 0

    for i in range(1, iterations + 1):
        mutated_rules, mutation_key, description = mutate_rules(current_rules)

        with open(temp_rules_path, "w") as f:
            json.dump(mutated_rules, f, indent=2)

        result = run_backtest(
            days=backtest_days, rules_path=str(temp_rules_path),
            initial_capital=initial_capital, broker=broker,
            account_currency=account_currency,
            cached_history=cached_history,
        )
        new_scores = score_backtest(result)
        delta = new_scores["composite"] - best_composite

        guard = check_guard_rails(best_scores, new_scores)

        if not guard["passed"]:
            status = "discard"
            discarded += 1
            icon = "✗"
            reason = f"Guard rail: {guard['violations'][0]}"
        elif delta > 0:
            status = "keep"
            kept += 1
            icon = "✓"
            reason = f"Composite: {best_composite:.1f} → {new_scores['composite']:.1f} ({delta:+.1f})"
            current_rules = mutated_rules
            best_rules = copy.deepcopy(mutated_rules)
            best_scores = new_scores
            best_composite = new_scores["composite"]
        elif delta == 0:
            status = "neutral"
            discarded += 1
            icon = "—"
            reason = "No change"
        else:
            status = "discard"
            discarded += 1
            icon = "✗"
            reason = f"Composite: {best_composite:.1f} → {new_scores['composite']:.1f} ({delta:+.1f})"

        print(f"  [{i:2d}/{iterations}] {icon} {mutation_key}: {description}")
        print(f"         {reason}")

        log.append({
            "iteration": i,
            "status": status,
            "mutation": mutation_key,
            "description": description,
            "scores": new_scores,
            "delta": round(delta, 2),
            "rules_snapshot": _rules_fingerprint(mutated_rules if status == "keep" else current_rules),
        })

    with open(rules_file, "w") as f:
        json.dump(best_rules, f, indent=2)
        f.write("\n")

    best_result = run_backtest(
        days=backtest_days, rules_path=str(rules_file),
        initial_capital=initial_capital, broker=broker,
        account_currency=account_currency,
        cached_history=cached_history,
    )
    final_scores = score_backtest(best_result)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "iterations": iterations,
        "kept": kept,
        "discarded": discarded,
        "backtest_days": backtest_days,
        "initial_capital": initial_capital,
        "broker": broker,
        "baseline_scores": baseline_scores,
        "final_scores": final_scores,
        "improvement": round(final_scores["composite"] - baseline_scores["composite"], 2),
        "final_backtest": {
            "return_pct": best_result.get("return_pct", 0),
            "net_pnl": best_result.get("total_pnl_net", 0),
            "win_rate": best_result.get("win_rate", 0),
            "total_trades": best_result.get("total_trades", 0),
            "cost_drag_pct": best_result.get("cost_drag_pct", 0),
        },
        "log": log,
    }

    log_path = RESULTS_DIR / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"  Iterations: {iterations} ({kept} kept, {discarded} discarded)")
    print(f"  Baseline → Final composite: {baseline_scores['composite']:.1f} → {final_scores['composite']:.1f} "
          f"({summary['improvement']:+.1f})")
    print(f"\n  Final backtest ({backtest_days} days):")
    print(f"    Return:      {best_result.get('return_pct', 0):+.2f}%")
    print(f"    Net P&L:     ${best_result.get('total_pnl_net', 0):+.2f}")
    print(f"    Win rate:    {best_result.get('win_rate', 0):.0f}%")
    print(f"    Trades:      {best_result.get('total_trades', 0)}")
    print(f"    Cost drag:   {best_result.get('cost_drag_pct', 0):.1f}%")
    print(f"\n  Best rules saved to: {rules_file}")
    print(f"  Full log saved to:   {log_path}")
    print(f"{'='*60}")

    temp_rules_path.unlink(missing_ok=True)

    return summary


def _rules_fingerprint(rules: dict) -> dict:
    """Extract key parameters for logging."""
    entry = rules.get("entry_rules", {})
    long_rules = entry.get("long", [])
    risk = rules.get("risk_rules", {})
    exit_rules = rules.get("exit_rules", [])

    return {
        "entry_conditions": [r.get("condition") for r in long_rules],
        "entry_weights": [r.get("weight") for r in long_rules],
        "min_score": entry.get("min_score"),
        "exit_conditions": [r.get("condition") for r in exit_rules],
        "stop_loss": risk.get("stop_loss_pct"),
        "take_profit": risk.get("take_profit_pct"),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Strategy Autoresearch Optimizer")
    parser.add_argument("--iterations", type=int, default=20, help="Number of optimization iterations")
    parser.add_argument("--days", type=int, default=90, help="Backtest period in days")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital")
    parser.add_argument("--broker", default="etoro", help="Broker profile")
    parser.add_argument("--currency", default="EUR", help="Account currency")
    args = parser.parse_args()

    run_optimization(
        iterations=args.iterations,
        backtest_days=args.days,
        initial_capital=args.capital,
        broker=args.broker,
        account_currency=args.currency,
    )


if __name__ == "__main__":
    main()
