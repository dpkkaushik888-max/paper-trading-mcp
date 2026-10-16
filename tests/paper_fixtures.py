"""Deterministic synthetic bars + a multi-day live-mode replay harness for run_daily.

Used by the live-path identity test (golden captured from master b6fd53b) and the
shadow tests. Paper-only; nothing here touches the network or state/journal.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOGEUSDT", "XRPUSDT"]
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "live_path_golden.json"
N_BARS = 330
REPLAY_DAYS = 120


def synthetic_bars(seed: int = 18, n: int = N_BARS, symbols=FIXTURE_SYMBOLS) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    idx.name = "date"
    out = {}
    for k, sym in enumerate(symbols):
        rng = np.random.default_rng(seed + k)
        rets = rng.normal(0.003, 0.04, n)
        close = 100.0 * (k + 1) * np.exp(np.cumsum(rets))
        opn = np.concatenate([[close[0]], close[:-1]])
        spread = np.abs(rng.normal(0.0, 0.02, n))
        high = np.maximum(opn, close) * (1 + spread)
        low = np.minimum(opn, close) * (1 - spread)
        out[sym] = pd.DataFrame({"Open": opn, "High": high, "Low": low, "Close": close,
                                 "Volume": 1000.0}, index=idx)
    return out


def replay_live(run_daily, bars: dict[str, pd.DataFrame], n_days: int = REPLAY_DAYS):
    """Replay the last ``n_days`` as the 00:30 UTC live cron would see them.

    On run day D the live fetch returns only closed bars, i.e. bars opened before D.
    Journal is kept in memory (dry_run=True, Journal.load patched); returns the Journal.
    """
    j = run_daily.Journal.init_fresh()
    all_dates = next(iter(bars.values())).index
    run_dates = all_dates[-n_days:]
    orig = (run_daily.Journal.load, run_daily.fetch_universe, run_daily._today_utc_date)
    try:
        run_daily.Journal.load = classmethod(lambda cls, *a, **k: j)
        for d in run_dates:
            visible = {s: df[df.index < d] for s, df in bars.items()}
            run_daily.fetch_universe = lambda syms, _v=visible: dict(_v)
            run_daily._today_utc_date = lambda _d=d: _d.date().isoformat()
            run_daily.run(dry_run=True)
    finally:
        run_daily.Journal.load, run_daily.fetch_universe, run_daily._today_utc_date = orig
    return j


LIVE_DAY_KEYS = ("date", "portfolio_value", "cash", "n_open", "closes", "decisions", "fills",
                 "skipped", "bh_btc_value", "bh_basket_value", "trend_timed_value", "selector_value")


def live_signature(j) -> dict:
    d = j.to_dict()
    return json.loads(json.dumps({
        "cash": d["cash"],
        "open_positions": d["open_positions"],
        "closed_trades": d["closed_trades"],
        "days": [{k: day.get(k) for k in LIVE_DAY_KEYS} for day in d["days"]],
    }))
