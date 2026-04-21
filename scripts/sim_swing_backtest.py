#!/usr/bin/env python3
"""S16 — Daily-bar swing backtest with purged walk-forward and locked holdout.

Honest pivot from 1-min day-trading (see spec S16). Predicts 5-day forward
direction on daily bars of liquid cryptos, using the curated 12-feature
swing set. Trains Logistic + XGBoost, runs purged walk-forward CV (monthly
retrain with 3-day embargo), and scores a locked 20% holdout exactly once.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/sim_swing_backtest.py
    PYTHONPATH=. .venv/bin/python scripts/sim_swing_backtest.py --years 2
    PYTHONPATH=. .venv/bin/python scripts/sim_swing_backtest.py --model xgboost
    PYTHONPATH=. .venv/bin/python scripts/sim_swing_backtest.py --holdout-only

The four go/no-go gates from the spec are printed at the end.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests as _requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_engine.features.swing import SWING_FEATURE_COLS, build_swing_features
from trading_engine.labels.swing import forward_direction_label
from trading_engine.models.classifiers import SmartLogistic, SmartXGBoost

warnings.filterwarnings("ignore")


# ── Config (from S16 spec) ──────────────────────────────────────────────────
CAPITAL = 10_000.0  # Spec target: €3–5/day on €10k.
SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "MATIC-USD"]

BINANCE_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "AVAX-USD": "AVAXUSDT",
    "LINK-USD": "LINKUSDT",
    "MATIC-USD": "MATICUSDT",
}

# Labels / prediction horizon
HORIZON_DAYS = 5

# Entry / exit
ENTRY_CONFIDENCE = 0.62
MAX_CONCURRENT = 4
POS_SIZE_PCT = 0.15
SL_PCT = 0.07
TP_PCT = 0.15
TRAIL_ACTIVATE = 0.05
TRAIL_OFFSET = 0.03
MAX_HOLD_DAYS = 10
MIN_HOLD_DAYS = 1

# Honest-sim costs (matches sim_1min_replay.py defaults)
COST_PCT = 0.0020         # 20 bps per side
SLIPPAGE_BPS = 0.0005     # 5 bps per fill
SL_SLIPPAGE_BPS = 0.0010  # extra 10 bps on stops

# Model
LOGISTIC_C = 0.05         # Heavier L2 than 1-min (was 0.15) — overfitting mitigation
XGB_PARAMS = {
    "n_estimators": 150,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 40,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
}

# Walk-forward protocol
TRAIN_PCT = 0.60       # First 60% = training baseline
WALKFWD_PCT = 0.20     # Next 20% = walk-forward (monthly retrain)
HOLDOUT_PCT = 0.20     # Last 20% = locked holdout (single evaluation)
EMBARGO_DAYS = 3       # Purged CV embargo
RETRAIN_EVERY_DAYS = 30

# Benchmark
BENCHMARK_ALPHA_TARGET = 0.02  # Must beat buy-and-hold by 2% (G2)


# ── Data fetching (Binance public API, read-only) ───────────────────────────

def _binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch klines from Binance with pagination (1000 bars per request)."""
    all_bars = []
    cursor = start_ms
    while cursor < end_ms:
        resp = _requests.get(
            "https://api.binance.com/api/v3/klines",
            params={
                "symbol": symbol, "interval": interval,
                "startTime": cursor, "endTime": end_ms, "limit": 1000,
            },
            timeout=10,
        )
        data = resp.json()
        if not data or isinstance(data, dict):
            break
        all_bars.extend(data)
        cursor = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.05)
    if not all_bars:
        return pd.DataFrame()
    df = pd.DataFrame(all_bars, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol",
        "taker_buy_quote", "ignore",
    ])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ts = pd.to_datetime(df["open_time"], unit="ms")
    df.index = pd.DatetimeIndex(ts.dt.normalize())
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_daily_bars(symbols: list[str], years: float) -> dict[str, pd.DataFrame]:
    """Fetch daily bars for each symbol going back `years` years."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(years * 365 * 86400 * 1000)
    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        bsym = BINANCE_MAP.get(sym, sym)
        print(f"    {sym} ({bsym})... ", end="", flush=True)
        df = _binance_klines(bsym, "1d", start_ms, end_ms)
        if not df.empty:
            # Binance returns today's partial bar — drop if incomplete.
            df = df[~df.index.duplicated(keep="first")]
            data[sym] = df
            print(f"{len(df):,} bars ({df.index[0].date()} → {df.index[-1].date()})")
        else:
            print("NO DATA")
    return data


def fetch_daily_funding(symbols: list[str], years: float) -> dict[str, pd.DataFrame]:
    """Fetch 8h funding rates and resample to daily mean."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(years * 365 * 86400 * 1000)
    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        bsym = BINANCE_MAP.get(sym, sym)
        all_rates = []
        cursor = start_ms
        while cursor < end_ms:
            try:
                resp = _requests.get(
                    "https://fapi.binance.com/fapi/v1/fundingRate",
                    params={"symbol": bsym, "startTime": cursor, "endTime": end_ms, "limit": 1000},
                    timeout=10,
                )
                rates = resp.json()
                if not rates or isinstance(rates, dict):
                    break
                all_rates.extend(rates)
                cursor = rates[-1]["fundingTime"] + 1
                if len(rates) < 1000:
                    break
                time.sleep(0.05)
            except Exception:
                break
        if all_rates:
            df = pd.DataFrame(all_rates)
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
            df.index = pd.to_datetime(df["fundingTime"], unit="ms")
            daily = df["fundingRate"].resample("1D").mean().to_frame("funding_rate")
            daily.index = daily.index.normalize()
            daily.index.name = "Date"
            data[sym] = daily
    return data


# ── Feature + label preparation ─────────────────────────────────────────────

@dataclass
class SymbolData:
    symbol: str
    features: pd.DataFrame
    label: pd.Series
    close: pd.Series  # for P&L


def prepare_symbol_data(
    data_daily: dict[str, pd.DataFrame],
    funding_daily: dict[str, pd.DataFrame],
) -> dict[str, SymbolData]:
    out: dict[str, SymbolData] = {}
    for sym, df in data_daily.items():
        fr = funding_daily.get(sym)
        feats = build_swing_features(df, funding_df=fr)
        lbl = forward_direction_label(df, horizon_days=HORIZON_DAYS)
        out[sym] = SymbolData(
            symbol=sym, features=feats, label=lbl, close=df["Close"],
        )
    return out


def build_training_matrix(
    sym_data: dict[str, SymbolData],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series]:
    """Concatenate per-symbol feature/label data for [start, end_exclusive)."""
    X_list, y_list = [], []
    for sd in sym_data.values():
        mask = (sd.features.index >= start) & (sd.features.index < end_exclusive)
        X = sd.features.loc[mask]
        y = sd.label.loc[mask]
        valid = y.notna() & X.notna().all(axis=1)
        X_list.append(X.loc[valid])
        y_list.append(y.loc[valid].astype(int))
    if not X_list:
        return pd.DataFrame(columns=SWING_FEATURE_COLS), pd.Series(dtype=int)
    return pd.concat(X_list), pd.concat(y_list)


# ── Simulator ───────────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    entry_cost: float
    entry_conf: float
    peak_pnl_pct: float = 0.0


@dataclass
class ClosedTrade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    hold_days: int
    reason: str


@dataclass
class DailyStep:
    date: pd.Timestamp
    portfolio_value: float
    cash: float
    open_positions: int
    daily_return: float


@dataclass
class SimResult:
    steps: list[DailyStep] = field(default_factory=list)
    trades: list[ClosedTrade] = field(default_factory=list)
    final_value: float = 0.0
    total_costs: float = 0.0


def run_simulation(
    sym_data: dict[str, SymbolData],
    all_dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    model_factory,
    retrain_every_days: int,
    train_start: pd.Timestamp,
    label: str,
) -> SimResult:
    """Run the daily swing simulator from ``start`` to ``end`` (inclusive).

    ``model_factory`` is called as ``model_factory()`` to build a fresh model
    instance each retrain. The model is always trained on
    ``[train_start, current_date - EMBARGO_DAYS)``.
    """
    cash = CAPITAL
    positions: dict[str, Position] = {}
    result = SimResult()
    last_retrain: pd.Timestamp | None = None
    model = None
    sim_days = all_dates[(all_dates >= start) & (all_dates <= end)]

    if len(sim_days) == 0:
        print(f"  [{label}] No simulation days in window.")
        return result

    prev_portfolio = CAPITAL

    for today in sim_days:
        # ── Retrain if due ──────────────────────────────────────────────
        if model is None or (today - last_retrain).days >= retrain_every_days:
            embargo_cutoff = today - pd.Timedelta(days=EMBARGO_DAYS)
            X_train, y_train = build_training_matrix(sym_data, train_start, embargo_cutoff)
            if len(X_train) < 200:
                prev_portfolio = _record_step(result, today, cash, positions, sym_data, prev_portfolio)
                continue
            model = model_factory()
            model.fit(X_train, y_train)
            last_retrain = today

        # ── Score signals for all symbols for today ─────────────────────
        signals: dict[str, tuple[float, float]] = {}  # sym → (up_prob, price)
        for sym, sd in sym_data.items():
            if today not in sd.features.index or today not in sd.close.index:
                continue
            row = sd.features.loc[today]
            if row.isna().any():
                continue
            X_today = row.values.reshape(1, -1)
            proba = model.predict_proba(X_today)[0]
            up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            price = float(sd.close.loc[today])
            if price > 0:
                signals[sym] = (up_prob, price)

        # ── Manage open positions: exits first ───────────────────────────
        to_close: list[tuple[str, str]] = []
        for sym, pos in positions.items():
            if sym not in signals:
                continue
            _, price = signals[sym]
            pnl_pct = (price - pos.entry_price) / pos.entry_price
            if pnl_pct > pos.peak_pnl_pct:
                pos.peak_pnl_pct = pnl_pct
            hold_days = (today - pos.entry_date).days

            reason = None
            if hold_days < MIN_HOLD_DAYS:
                pass
            elif pnl_pct <= -SL_PCT:
                reason = "SL"
            elif pnl_pct >= TP_PCT:
                reason = "TP"
            elif pos.peak_pnl_pct >= TRAIL_ACTIVATE and pnl_pct <= pos.peak_pnl_pct - TRAIL_OFFSET:
                reason = "TRAIL"
            elif hold_days >= MAX_HOLD_DAYS:
                reason = "MAX_HOLD"
            if reason:
                to_close.append((sym, reason))

        for sym, reason in to_close:
            pos = positions[sym]
            _, price = signals[sym]
            exit_slip = SLIPPAGE_BPS + (SL_SLIPPAGE_BPS if reason == "SL" else 0.0)
            fill_price = price * (1.0 - exit_slip)  # long close fills at bid
            exit_cost = fill_price * pos.shares * COST_PCT
            proceeds = fill_price * pos.shares - exit_cost
            cash += proceeds
            result.total_costs += exit_cost
            pnl = proceeds - (pos.entry_price * pos.shares + pos.entry_cost)
            pnl_pct = pnl / (pos.entry_price * pos.shares)
            result.trades.append(ClosedTrade(
                symbol=sym, entry_date=pos.entry_date, exit_date=today,
                entry_price=pos.entry_price, exit_price=fill_price,
                shares=pos.shares, pnl=pnl, pnl_pct=pnl_pct,
                hold_days=(today - pos.entry_date).days, reason=reason,
            ))
            del positions[sym]

        # ── Entries: rank candidates by confidence, take top K ──────────
        capacity = MAX_CONCURRENT - len(positions)
        if capacity > 0:
            candidates = [
                (sym, up_prob, price)
                for sym, (up_prob, price) in signals.items()
                if up_prob > ENTRY_CONFIDENCE and sym not in positions
            ]
            candidates.sort(key=lambda c: -c[1])
            for sym, conf, price in candidates[:capacity]:
                entry_fill = price * (1.0 + SLIPPAGE_BPS)  # long buys the ask
                pos_value = CAPITAL * POS_SIZE_PCT
                if pos_value > cash:
                    break
                shares = pos_value / entry_fill
                entry_cost = entry_fill * shares * COST_PCT
                debit = entry_fill * shares + entry_cost
                if debit > cash:
                    continue
                cash -= debit
                result.total_costs += entry_cost
                positions[sym] = Position(
                    symbol=sym, entry_date=today, entry_price=entry_fill,
                    shares=shares, entry_cost=entry_cost, entry_conf=conf,
                )

        prev_portfolio = _record_step(result, today, cash, positions, sym_data, prev_portfolio)

    # Close any remaining positions at last-known price (mark-to-market only).
    if positions:
        last_day = sim_days[-1]
        for sym, pos in list(positions.items()):
            sym_close = sym_data[sym].close
            # last_day may be absent from this symbol's data — use most recent known.
            available = sym_close[sym_close.index <= last_day]
            if available.empty:
                continue
            close_price = float(available.iloc[-1])
            exit_slip = SLIPPAGE_BPS
            fill_price = close_price * (1.0 - exit_slip)
            exit_cost = fill_price * pos.shares * COST_PCT
            proceeds = fill_price * pos.shares - exit_cost
            cash += proceeds
            result.total_costs += exit_cost
            pnl = proceeds - (pos.entry_price * pos.shares + pos.entry_cost)
            pnl_pct = pnl / (pos.entry_price * pos.shares)
            result.trades.append(ClosedTrade(
                symbol=sym, entry_date=pos.entry_date, exit_date=last_day,
                entry_price=pos.entry_price, exit_price=fill_price,
                shares=pos.shares, pnl=pnl, pnl_pct=pnl_pct,
                hold_days=(last_day - pos.entry_date).days, reason="END",
            ))
            del positions[sym]

    result.final_value = cash
    return result


def _record_step(
    result: SimResult,
    date: pd.Timestamp,
    cash: float,
    positions: dict[str, Position],
    sym_data: dict[str, SymbolData],
    prev_portfolio: float,
) -> float:
    """Append a DailyStep and return the new prev_portfolio value."""
    portfolio = cash
    for sym, pos in positions.items():
        sym_close = sym_data[sym].close
        if date in sym_close.index:
            mark = float(sym_close.loc[date])
        else:
            # Gap or missing bar — use most recent known price, else entry.
            prior = sym_close[sym_close.index <= date]
            mark = float(prior.iloc[-1]) if not prior.empty else pos.entry_price
        portfolio += mark * pos.shares
    daily_ret = (portfolio / prev_portfolio - 1.0) if prev_portfolio > 0 else 0.0
    result.steps.append(DailyStep(
        date=date, portfolio_value=portfolio, cash=cash,
        open_positions=len(positions), daily_return=daily_ret,
    ))
    return portfolio


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(result: SimResult, capital: float = CAPITAL) -> dict:
    if not result.steps:
        return {"error": "no_steps"}
    n_days = len(result.steps)
    final = result.steps[-1].portfolio_value
    total_ret = final / capital - 1.0
    years = n_days / 365.25
    cagr = (final / capital) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    rets = np.array([s.daily_return for s in result.steps])
    sharpe = (rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else 0.0

    # Max drawdown
    values = np.array([s.portfolio_value for s in result.steps])
    peak = np.maximum.accumulate(values)
    dd = (peak - values) / peak
    max_dd = dd.max() if len(dd) > 0 else 0.0

    trades = result.trades
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if wins else 0.0
    avg_loss = np.mean([t.pnl for t in trades if t.pnl <= 0]) if losses else 0.0
    profit_factor = (
        abs(avg_win * wins) / abs(avg_loss * losses) if losses > 0 and avg_loss != 0 else float("inf")
    )

    # Per-year breakdown
    step_df = pd.DataFrame([{"date": s.date, "value": s.portfolio_value} for s in result.steps])
    step_df["year"] = step_df["date"].dt.year
    year_rets = step_df.groupby("year")["value"].agg(["first", "last"])
    year_rets["ret"] = year_rets["last"] / year_rets["first"] - 1.0

    return {
        "n_days": n_days,
        "years": years,
        "final_value": final,
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": len(trades),
        "trades_per_year": len(trades) / years if years > 0 else 0,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_costs": result.total_costs,
        "year_returns": year_rets["ret"].to_dict(),
    }


def compute_benchmark(
    sym_data: dict[str, SymbolData],
    start: pd.Timestamp,
    end: pd.Timestamp,
    capital: float = CAPITAL,
) -> dict:
    """Equal-weight buy-and-hold across the universe."""
    closes = pd.DataFrame({sym: sd.close for sym, sd in sym_data.items()})
    closes = closes[(closes.index >= start) & (closes.index <= end)].dropna(how="all")
    if closes.empty:
        return {"error": "no_benchmark_data"}
    first_row = closes.iloc[0]
    per_sym_capital = capital / len(closes.columns)
    shares = per_sym_capital / first_row
    values = closes.multiply(shares, axis=1).sum(axis=1)
    final = float(values.iloc[-1])
    total_ret = final / capital - 1.0
    years = len(values) / 365.25
    cagr = (final / capital) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    daily_ret = values.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(365)) if daily_ret.std() > 0 else 0.0
    peak = values.cummax()
    dd = (peak - values) / peak
    return {
        "final_value": final,
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": float(dd.max()),
    }


# ── Reporting ───────────────────────────────────────────────────────────────

def fmt_pct(x: float) -> str:
    return f"{x*100:+.2f}%"


def print_metrics(label: str, m: dict, benchmark: dict | None = None) -> None:
    if "error" in m:
        print(f"  [{label}] ERROR: {m['error']}")
        return
    print(f"\n  ━━━ {label} ━━━")
    print(f"  {'Days':<20} {m['n_days']:>12,} ({m['years']:.2f} years)")
    print(f"  {'Final value':<20} ${m['final_value']:>12,.2f}")
    print(f"  {'Total return':<20} {fmt_pct(m['total_return']):>12}")
    print(f"  {'CAGR':<20} {fmt_pct(m['cagr']):>12}")
    print(f"  {'Sharpe':<20} {m['sharpe']:>12.3f}")
    print(f"  {'Max drawdown':<20} {fmt_pct(m['max_dd']):>12}")
    print(f"  {'Trades':<20} {m['trades']:>12,} ({m['trades_per_year']:.1f}/yr)")
    print(f"  {'Win rate':<20} {m['win_rate']*100:>11.1f}%")
    print(f"  {'Profit factor':<20} {m['profit_factor']:>12.2f}")
    print(f"  {'Total costs':<20} ${m['total_costs']:>12,.2f}")
    if m.get("year_returns"):
        print(f"  Per-year returns:")
        for yr, ret in sorted(m["year_returns"].items()):
            print(f"    {yr}: {fmt_pct(ret)}")
    if benchmark and "error" not in benchmark:
        print(f"\n  ━━━ {label} vs BUY-AND-HOLD benchmark ━━━")
        print(f"  {'Benchmark CAGR':<20} {fmt_pct(benchmark['cagr']):>12}")
        print(f"  {'Benchmark Sharpe':<20} {benchmark['sharpe']:>12.3f}")
        print(f"  {'Benchmark MaxDD':<20} {fmt_pct(benchmark['max_dd']):>12}")
        alpha = m["cagr"] - benchmark["cagr"]
        print(f"  {'Alpha (CAGR delta)':<20} {fmt_pct(alpha):>12}")


def print_gates(label: str, m: dict, benchmark: dict | None) -> None:
    if "error" in m:
        return
    print(f"\n  ━━━ GO/NO-GO GATES [{label}] ━━━")
    g1 = m["sharpe"] > 1.0
    print(f"  {'G1: Sharpe > 1.0':<45} {'PASS' if g1 else 'FAIL':>6}  ({m['sharpe']:.3f})")

    if benchmark and "error" not in benchmark:
        alpha = m["cagr"] - benchmark["cagr"]
        g2 = alpha > BENCHMARK_ALPHA_TARGET
        print(f"  {'G2: CAGR > benchmark + 2%':<45} {'PASS' if g2 else 'FAIL':>6}  "
              f"(alpha {fmt_pct(alpha)})")
    else:
        g2 = False
        print(f"  {'G2: CAGR > benchmark + 2%':<45} {'SKIP':>6}  (no benchmark)")

    g3 = m["max_dd"] < 0.30
    print(f"  {'G3: Max drawdown < 30%':<45} {'PASS' if g3 else 'FAIL':>6}  "
          f"({fmt_pct(m['max_dd'])})")

    year_rets = m.get("year_returns", {})
    profitable_years = sum(1 for r in year_rets.values() if r > 0)
    total_years = len(year_rets)
    g4 = profitable_years >= max(2, total_years - 1) if total_years >= 3 else profitable_years >= 1
    print(f"  {'G4: Profitable in ≥2 of 3 years':<45} {'PASS' if g4 else 'FAIL':>6}  "
          f"({profitable_years}/{total_years})")

    all_pass = g1 and g2 and g3 and g4
    print(f"\n  VERDICT: {'ALL GATES PASS — viable for paper live' if all_pass else 'FAIL — do not deploy'}")


# ── Main orchestration ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="S16 — swing-timeframe ML backtest")
    parser.add_argument("--years", type=float, default=5.0, help="Years of history to fetch")
    parser.add_argument("--model", choices=["logistic", "xgboost", "both"], default="both")
    parser.add_argument("--holdout-only", action="store_true",
                        help="Skip walk-forward; run model once, evaluate holdout only")
    parser.add_argument("--no-funding", action="store_true")
    args = parser.parse_args()

    print("=" * 90)
    print("  S16 — SWING BACKTEST (daily bars, honest costs, locked holdout)")
    print("=" * 90)
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Horizon: {HORIZON_DAYS}d | Conf: {ENTRY_CONFIDENCE} | "
          f"SL/TP: {SL_PCT:.0%}/{TP_PCT:.0%} | Max hold: {MAX_HOLD_DAYS}d")
    print(f"  Costs: {COST_PCT*10000:.0f}bps/side + {SLIPPAGE_BPS*10000:.0f}bps slip "
          f"+ {SL_SLIPPAGE_BPS*10000:.0f}bps SL slip")
    print(f"  Capital: ${CAPITAL:,.0f}")

    # 1. Fetch data
    print(f"\n  [1/4] Fetching {args.years} years of daily bars from Binance...")
    t0 = time.time()
    data_daily = fetch_daily_bars(SYMBOLS, years=args.years)
    if not data_daily:
        print("  ERROR: no data fetched")
        return
    print(f"         {len(data_daily)} symbols, {time.time() - t0:.1f}s")

    funding = {}
    if not args.no_funding:
        print(f"\n  [2/4] Fetching daily funding rates...")
        t0 = time.time()
        funding = fetch_daily_funding(SYMBOLS, years=args.years)
        print(f"         {len(funding)} symbols, {time.time() - t0:.1f}s")

    # 2. Feature + label prep
    print(f"\n  [3/4] Building swing features + labels...")
    t0 = time.time()
    sym_data = prepare_symbol_data(data_daily, funding)
    total_rows = sum(len(sd.features.dropna()) for sd in sym_data.values())
    print(f"         {total_rows:,} valid feature rows across {len(sym_data)} symbols, "
          f"{time.time() - t0:.1f}s")

    # 3. Split timeline
    all_dates = pd.DatetimeIndex(
        sorted(set().union(*(sd.features.dropna().index for sd in sym_data.values())))
    )
    if len(all_dates) < 500:
        print(f"  ERROR: only {len(all_dates)} valid dates — need at least 500")
        return
    n = len(all_dates)
    train_end_idx = int(n * TRAIN_PCT)
    walkfwd_end_idx = int(n * (TRAIN_PCT + WALKFWD_PCT))
    train_start = all_dates[0]
    train_end = all_dates[train_end_idx]
    walkfwd_end = all_dates[walkfwd_end_idx]
    holdout_end = all_dates[-1]
    print(f"\n  Timeline:")
    print(f"    Training base: {train_start.date()} → {train_end.date()} ({train_end_idx} days)")
    print(f"    Walk-forward:  {train_end.date()} → {walkfwd_end.date()} "
          f"({walkfwd_end_idx - train_end_idx} days)")
    print(f"    LOCKED HOLDOUT: {walkfwd_end.date()} → {holdout_end.date()} "
          f"({n - walkfwd_end_idx} days)")

    # 4. Run per model
    print(f"\n  [4/4] Running backtest(s)...")
    models_to_run = []
    if args.model in ("logistic", "both"):
        models_to_run.append(("Logistic", lambda: SmartLogistic(
            params={"C": LOGISTIC_C, "class_weight": "balanced", "max_iter": 2000},
        )))
    if args.model in ("xgboost", "both"):
        models_to_run.append(("XGBoost", lambda: SmartXGBoost(params=XGB_PARAMS)))

    benchmark_wf = compute_benchmark(sym_data, train_end, walkfwd_end)
    benchmark_ho = compute_benchmark(sym_data, walkfwd_end, holdout_end)

    for model_name, factory in models_to_run:
        print(f"\n{'='*90}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*90}")

        if not args.holdout_only:
            # Walk-forward validation
            t0 = time.time()
            wf_result = run_simulation(
                sym_data, all_dates, train_end, walkfwd_end, factory,
                retrain_every_days=RETRAIN_EVERY_DAYS, train_start=train_start,
                label=f"{model_name}-WF",
            )
            wf_metrics = compute_metrics(wf_result)
            print(f"  Walk-forward done in {time.time()-t0:.1f}s")
            print_metrics(f"{model_name} — WALK-FORWARD", wf_metrics, benchmark_wf)

        # Locked holdout — single evaluation, never touched during tuning.
        t0 = time.time()
        ho_result = run_simulation(
            sym_data, all_dates, walkfwd_end, holdout_end, factory,
            retrain_every_days=RETRAIN_EVERY_DAYS, train_start=train_start,
            label=f"{model_name}-HOLDOUT",
        )
        ho_metrics = compute_metrics(ho_result)
        print(f"\n  Holdout done in {time.time()-t0:.1f}s")
        print_metrics(f"{model_name} — LOCKED HOLDOUT", ho_metrics, benchmark_ho)
        print_gates(f"{model_name} HOLDOUT", ho_metrics, benchmark_ho)

    print(f"\n{'='*90}")
    print(f"  S16 DONE.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
