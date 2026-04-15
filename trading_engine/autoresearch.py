"""Autoresearch Strategy Optimizer — automated iteration loop.

Tries different parameter configurations for a given market,
runs backtests, keeps improvements, reverts failures.

Usage:
    python -m trading_engine.autoresearch --market crypto --period 2y --iterations 20
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import time
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import CRYPTO_WATCHLIST, PROJECT_ROOT
from .ml_model import MARKET_CONFIGS
from .time_machine import TimeMachineBacktest

warnings.filterwarnings("ignore")

RESULTS_DIR = PROJECT_ROOT / "autoresearch"
RESULTS_DIR.mkdir(exist_ok=True)


SEARCH_SPACE = {
    "confidence_threshold": [0.65, 0.70, 0.75, 0.80, 0.85],
    "train_window": [100, 150, 200, 300],
    "retrain_every": [7, 14, 21, 30],
    "sl_tp": [(0.03, 0.06), (0.05, 0.08), (0.05, 0.10), (0.07, 0.12)],
    "max_position_pct": [0.10, 0.15, 0.20],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "max_depth": [2, 3, 4, 5],
    "n_estimators": [100, 150, 200, 300],
    "min_child_samples": [10, 15, 20, 30],
}

SEARCH_SPACE_WINRATE = {
    "confidence_threshold": [0.70, 0.75, 0.80, 0.85, 0.90],
    "train_window": [100, 150, 200],
    "retrain_every": [7, 10, 14, 21],
    "sl_tp": [(0.05, 0.08), (0.05, 0.10), (0.07, 0.12), (0.07, 0.15), (0.10, 0.15)],
    "max_position_pct": [0.10, 0.15],
    "learning_rate": [0.02, 0.03, 0.05],
    "max_depth": [2, 3],
    "n_estimators": [100, 150, 200],
    "min_child_samples": [15, 20, 30],
}


def _safe_float(val, default: float = 1.0) -> float:
    """Convert value to float safely."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


CACHE_DIR = PROJECT_ROOT / "autoresearch" / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_crypto_data(period: str = "2y", force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Download crypto data with disk cache. Reuses cached data across sessions."""
    import pickle
    from datetime import datetime

    cache_file = CACHE_DIR / f"crypto_{period}.pkl"

    if cache_file.exists() and not force_refresh:
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            print(f"  (loaded {len(data)} assets from cache, {age_hours:.1f}h old)")
            return data

    data = {}
    for sym in CRYPTO_WATCHLIST:
        for attempt in range(3):
            try:
                df = yf.Ticker(sym).history(period=period, interval="1d")
                if df.empty:
                    break
                df.index = df.index.tz_localize(None) if df.index.tz else df.index
                data[sym] = df
                break
            except Exception:
                if attempt < 2:
                    time.sleep(2)

    if data:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"  (cached {len(data)} assets to {cache_file.name})")

    return data


def _run_single_config(
    data: dict[str, pd.DataFrame],
    config: dict,
    iteration: int,
) -> dict:
    """Run a single backtest with the given config and return metrics."""
    sl, tp = config["sl_tp"]

    lgbm_params = {
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "learning_rate": config["learning_rate"],
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_samples": config["min_child_samples"],
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }

    original_cfg = MARKET_CONFIGS["crypto"].copy()
    MARKET_CONFIGS["crypto"].update({
        "train_window": config["train_window"],
        "retrain_every": config["retrain_every"],
        "lgbm_params": lgbm_params,
    })

    try:
        tm = TimeMachineBacktest(
            market="crypto",
            initial_capital=10_000,
            confidence_threshold=config["confidence_threshold"],
            max_position_pct=config["max_position_pct"],
            stop_loss_pct=sl,
            take_profit_pct=tp,
            session_id=f"autoresearch_iter_{iteration}",
            enable_learning=True,
        )
        result = tm.run(history_data=data)
    finally:
        MARKET_CONFIGS["crypto"].update(original_cfg)

    return {
        "return_pct": result.get("return_pct", 0),
        "win_rate": result.get("win_rate", 0),
        "max_dd": max(
            (abs(cb.get("dd", 0)) for cb in result.get("circuit_breaker_log", [])),
            default=0.0,
        ),
        "cal_error": _safe_float(result.get("calibration", {}).get("calibration_error", 1.0)),
        "total_trades": result.get("total_trades", 0),
        "closed_trades": result.get("closed_trades", 0),
    }


def _score(metrics: dict, mode: str = "balanced") -> float:
    """Compute composite score from metrics. Higher is better."""
    ret = max(-50, min(50, metrics["return_pct"])) / 50.0
    wr = metrics["win_rate"] / 100.0
    try:
        cal = 1.0 - min(1.0, float(metrics["cal_error"]))
    except (ValueError, TypeError):
        cal = 0.0
    dd = 1.0 - min(1.0, metrics["max_dd"] / 20.0)

    if mode == "winrate":
        return 0.20 * ret + 0.45 * wr + 0.15 * cal + 0.20 * dd
    return 0.40 * ret + 0.25 * wr + 0.20 * cal + 0.15 * dd


def _generate_candidates(n: int, seed: int = 42, mode: str = "balanced") -> list[dict]:
    """Generate n random configs from the search space."""
    rng = random.Random(seed)
    space = SEARCH_SPACE_WINRATE if mode == "winrate" else SEARCH_SPACE
    candidates = []

    for _ in range(n):
        config = {}
        for key, values in space.items():
            config[key] = rng.choice(values)
        candidates.append(config)

    return candidates


def run_autoresearch(
    market: str = "crypto",
    period: str = "2y",
    max_iterations: int = 20,
    patience: int = 5,
    verbose: bool = True,
    mode: str = "balanced",
):
    """Run the autoresearch loop."""
    print("=" * 90)
    print(f"  AUTORESEARCH — {market.upper()} Strategy Optimizer")
    print(f"  Max iterations: {max_iterations} | Patience: {patience} | Period: {period} | Mode: {mode}")
    print("=" * 90)

    print(f"\n  Downloading {market} data ({period})...")
    t0 = time.time()
    data = _fetch_crypto_data(period)
    print(f"  Loaded {len(data)} assets in {time.time() - t0:.0f}s")

    if not data:
        print("  ERROR: No data loaded. Aborting.")
        return

    suffix = f"_{mode}" if mode != "balanced" else ""
    tsv_path = RESULTS_DIR / f"{market}_results{suffix}.tsv"
    tsv_file = open(tsv_path, "w", newline="")
    writer = csv.writer(tsv_file, delimiter="\t")
    writer.writerow([
        "iter", "score", "return_pct", "win_rate", "max_dd", "cal_error",
        "trades", "closed", "confidence", "train_window", "retrain_every",
        "sl", "tp", "max_pos", "lr", "depth", "estimators", "min_child",
        "kept", "runtime_s",
    ])

    candidates = _generate_candidates(max_iterations, seed=42 if mode == "balanced" else 99, mode=mode)

    best_score = -999
    best_config = None
    best_metrics = None
    no_improve_count = 0

    for i, config in enumerate(candidates):
        sl, tp = config["sl_tp"]
        print(f"\n  --- Iteration {i+1}/{max_iterations} ---")
        print(f"  conf={config['confidence_threshold']:.2f} "
              f"tw={config['train_window']} rt={config['retrain_every']} "
              f"sl={sl:.0%}/{tp:.0%} pos={config['max_position_pct']:.0%} "
              f"lr={config['learning_rate']} d={config['max_depth']} "
              f"n={config['n_estimators']} mc={config['min_child_samples']}")

        t0 = time.time()
        try:
            metrics = _run_single_config(data, config, i + 1)
        except Exception as e:
            print(f"  ERROR: {e}")
            writer.writerow([
                i + 1, "ERR", 0, 0, 0, 0, 0, 0,
                config["confidence_threshold"], config["train_window"],
                config["retrain_every"], sl, tp, config["max_position_pct"],
                config["learning_rate"], config["max_depth"],
                config["n_estimators"], config["min_child_samples"],
                "ERR", 0,
            ])
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\n  PATIENCE EXHAUSTED ({patience} consecutive failures). Stopping.")
                break
            continue

        elapsed = time.time() - t0
        score = _score(metrics, mode=mode)
        kept = score > best_score

        if kept:
            best_score = score
            best_config = config.copy()
            best_metrics = metrics.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        status = "KEPT ★" if kept else "reverted"
        print(f"  Return: {metrics['return_pct']:+.2f}% | WR: {metrics['win_rate']:.1f}% | "
              f"DD: {metrics['max_dd']:.2f}% | Cal: {metrics['cal_error']:.3f} | "
              f"Trades: {metrics['total_trades']} | Score: {score:.4f} | {status} "
              f"({elapsed:.0f}s)")

        writer.writerow([
            i + 1, f"{score:.4f}", f"{metrics['return_pct']:.2f}",
            f"{metrics['win_rate']:.1f}", f"{metrics['max_dd']:.2f}",
            f"{metrics['cal_error']:.3f}", metrics['total_trades'],
            metrics['closed_trades'],
            config["confidence_threshold"], config["train_window"],
            config["retrain_every"], sl, tp, config["max_position_pct"],
            config["learning_rate"], config["max_depth"],
            config["n_estimators"], config["min_child_samples"],
            "YES" if kept else "no", f"{elapsed:.0f}",
        ])
        tsv_file.flush()

        if no_improve_count >= patience:
            print(f"\n  PATIENCE EXHAUSTED ({patience} consecutive failures). Stopping.")
            break

    tsv_file.close()

    print(f"\n{'='*90}")
    print(f"  AUTORESEARCH COMPLETE")
    print(f"{'='*90}")

    if best_config:
        sl, tp = best_config["sl_tp"]
        print(f"\n  BEST CONFIG (score={best_score:.4f}):")
        print(f"    confidence:     {best_config['confidence_threshold']:.2f}")
        print(f"    train_window:   {best_config['train_window']}")
        print(f"    retrain_every:  {best_config['retrain_every']}")
        print(f"    SL/TP:          {sl:.0%} / {tp:.0%}")
        print(f"    max_position:   {best_config['max_position_pct']:.0%}")
        print(f"    learning_rate:  {best_config['learning_rate']}")
        print(f"    max_depth:      {best_config['max_depth']}")
        print(f"    n_estimators:   {best_config['n_estimators']}")
        print(f"    min_child:      {best_config['min_child_samples']}")
        print(f"\n  BEST METRICS:")
        print(f"    Return:         {best_metrics['return_pct']:+.2f}%")
        print(f"    Win rate:       {best_metrics['win_rate']:.1f}%")
        print(f"    Max DD:         {best_metrics['max_dd']:.2f}%")
        print(f"    Cal error:      {best_metrics['cal_error']:.3f}")
        print(f"    Trades:         {best_metrics['total_trades']}")
    else:
        print("\n  No successful iterations.")

    print(f"\n  Results saved to: {tsv_path}")
    print(f"{'='*90}")

    return {
        "best_score": best_score,
        "best_config": best_config,
        "best_metrics": best_metrics,
        "tsv_path": str(tsv_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Autoresearch Strategy Optimizer")
    parser.add_argument("--market", default="crypto", choices=["crypto", "us", "india"])
    parser.add_argument("--period", default="2y")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--mode", default="balanced", choices=["balanced", "winrate"])
    args = parser.parse_args()

    run_autoresearch(
        market=args.market,
        period=args.period,
        max_iterations=args.iterations,
        patience=args.patience,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
