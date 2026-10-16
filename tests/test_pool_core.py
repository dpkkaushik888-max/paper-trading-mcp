"""S35 C1 pool core: registry, generalised book (parity with the S34 shadow step), day context,
failure isolation and the live line's read-only boundary."""
import json
import re
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

from trading_engine.paper import run_daily
from trading_engine.paper.config import COST_PCT, MAX_CONCURRENT, POS_SIZE_PCT, SLIPPAGE_BPS, SL_SLIPPAGE_BPS, STARTING_CAPITAL
from trading_engine.paper.journal import DaySnapshot, Journal
from trading_engine.paper.pool import book as pool_book
from trading_engine.paper.pool.context import DayContext, build_context, contiguous_bars
from trading_engine.paper.pool.families.connors import ConnorsStrategy
from trading_engine.paper.pool.registry import LINES, LIVE_ID, BookParams, Hypothesis, LineSpec, validate
from trading_engine.paper.shadow import SHADOW_VARIANTS
from trading_engine.strategies.connors_swing import atr_wilder, long_entry, long_exit, precompute_indicators
from tests.paper_fixtures import GOLDEN_PATH, live_signature, replay_live, synthetic_bars

POOL_DIR = Path(__file__).parent.parent / "trading_engine" / "paper" / "pool"


def _ref_step(book, ov, gate, snapshot, atr_today, btc_close, btc_sma, today):
    """Verbatim logic of the pre-C1 shadow.step (1f96a6b), kept as the parity reference."""
    if not book["initialized"]:
        book.update(initialized=True, init_date=today, cash=STARTING_CAPITAL, open_positions=[], closed_trades=[])
    t = pd.Timestamp(today)
    for pos in list(book["open_positions"]):
        if pos["symbol"] not in snapshot:
            continue
        c, row = snapshot[pos["symbol"]]
        ed = pd.Timestamp(pos["entry_date"])
        reason = long_exit(c, row, pos["entry_price"], ed, t, overrides=ov, entry_atr=pos.get("entry_atr"))
        if reason is None:
            continue
        fill = c * (1.0 - SLIPPAGE_BPS - (SL_SLIPPAGE_BPS if reason == "SL" else 0.0))
        proceeds = fill * pos["shares"] - fill * pos["shares"] * COST_PCT
        pnl = proceeds - (pos["entry_price"] * pos["shares"] + pos["entry_cost"])
        book["closed_trades"].append(dict(
            symbol=pos["symbol"], entry_date=pos["entry_date"], exit_date=today, entry_price=pos["entry_price"],
            exit_price=fill, shares=pos["shares"], pnl=pnl, pnl_pct=pnl / (pos["entry_price"] * pos["shares"]),
            hold_days=(t - ed).days, reason=reason))
        book["open_positions"] = [p for p in book["open_positions"] if p["symbol"] != pos["symbol"]]
        book["cash"] += proceeds
    gate_open = not gate or (btc_close is not None and btc_sma is not None and btc_close > btc_sma)
    open_syms = {p["symbol"] for p in book["open_positions"]}
    cands = sorted(s for s, (c, r) in snapshot.items()
                   if s not in open_syms and long_entry(c, r, use_adx_filter=True, overrides=ov)) if gate_open else []
    cap = MAX_CONCURRENT - len(book["open_positions"])
    for sym in cands[:cap] if cap > 0 else []:
        fill = snapshot[sym][0] * (1.0 + SLIPPAGE_BPS)
        pv = STARTING_CAPITAL * POS_SIZE_PCT
        if pv > book["cash"]:
            continue
        shares = pv / fill
        cost = fill * shares * COST_PCT
        if fill * shares + cost > book["cash"]:
            continue
        book["cash"] -= fill * shares + cost
        a = atr_today.get(sym)
        book["open_positions"].append(dict(symbol=sym, entry_date=today, entry_price=fill, shares=shares,
                                           entry_cost=cost, entry_atr=None if a is None or pd.isna(a) else float(a)))
    book["value"] = book["cash"] + sum(
        (snapshot[p["symbol"]][0] if p["symbol"] in snapshot else p["entry_price"]) * p["shares"]
        for p in book["open_positions"])


def _strip(trades, keys):
    return [{k: t[k] for k in keys} for t in trades]


# (a) + (b) golden live path unchanged; all four registry lines reproduce the old step exactly.
def test_golden_live_path_and_parity_with_old_shadow_step():
    bars = synthetic_bars()
    j = replay_live(run_daily, bars)
    assert live_signature(j) == json.loads(GOLDEN_PATH.read_text())
    refs = {v.name: dict(initialized=False, cash=STARTING_CAPITAL, open_positions=[], closed_trades=[])
            for v in SHADOW_VARIANTS}
    for d in next(iter(bars.values())).index[-120:]:
        vis = {s: df[df.index < d] for s, df in bars.items()}
        snap = {s: (float(df["Close"].iloc[-1]), precompute_indicators(df).loc[df.index[-1]]) for s, df in vis.items()}
        atr = {s: float(atr_wilder(df).iloc[-1]) for s, df in vis.items()}
        btc = vis["BTCUSDT"]["Close"]
        sma = float(btc.rolling(200).mean().iloc[-1]) if len(btc) >= 200 else None
        for v in SHADOW_VARIANTS:
            _ref_step(refs[v.name], v.overrides, v.btc_sma_gate, snap, atr, float(btc.iloc[-1]), sma, d.date().isoformat())
    assert [v.name for v in SHADOW_VARIANTS] == [l.id for l in LINES]
    for v in SHADOW_VARIANTS:
        new, ref = j.get_benchmark(v.name), refs[v.name]
        assert ref["closed_trades"], v.name
        assert _strip(new["closed_trades"], ref["closed_trades"][0]) == ref["closed_trades"]
        assert _strip(new["open_positions"], ["symbol", "entry_date", "entry_price", "shares", "entry_cost",
                                               "entry_atr"]) == ref["open_positions"]
        assert (new["cash"], new["value"]) == (ref["cash"], ref["value"])
        assert j.days[-1][f"{v.name}_value"] == j.days[-1]["lines"][v.name]["value"] == ref["value"]
    day = j.days[-1]
    assert LIVE_ID in day["lines"] and day["lines"][LIVE_ID]["value"] == day["portfolio_value"]
    assert day["context"]["btc_trend"] != "unknown" and day["pool_errors"] == {}
    t = j.get_benchmark("shadow_rsi15")["closed_trades"][0]
    assert t["entry_ctx"]["symbol"] == t["symbol"] and t["exit_ctx"]["reason"] == t["reason"]
    assert t["mae_pct"] <= t["mfe_pct"] and t["entry_ctx"]["strength"] is not None


# (c) a raising line never affects live or other lines; its error is recorded.
def test_failing_line_isolated(monkeypatch):
    ok = LINES[2]
    ref = replay_live(run_daily, synthetic_bars(), n_days=10)
    good = ref.days[-1][f"{ok.id}_value"]
    bad = LineSpec("boom_line", "no_such_family", {}, registered_on="2026-10-16",
                   hypothesis=Hypothesis("beats_live"), note="test")
    monkeypatch.setattr(run_daily, "POOL_LINES", (bad, ok))
    j = replay_live(run_daily, synthetic_bars(), n_days=10)
    assert live_signature(j) == live_signature(ref)
    day = j.days[-1]
    assert "boom_line" in day["pool_errors"] and day["lines"]["boom_line"]["value"] == 0.0
    assert day[f"{ok.id}_value"] == good and day["shadow_errors"] == {}


# (d) pre-pool journal rows load with empty context/lines/pool_errors.
def test_old_rows_load_with_defaults():
    old = {"date": "2026-05-01", "portfolio_value": 1.0, "cash": 1.0, "n_open": 0, "closes": {},
           "decisions": [], "fills": []}
    s = DaySnapshot(**old)
    assert (s.context, s.lines, s.pool_errors) == ({}, {}, {})
    real = Path(__file__).parent.parent / "state" / "journal.json"
    if real.exists():
        j = Journal.load(real)
        known = DaySnapshot.__dataclass_fields__
        assert all(DaySnapshot(**{k: v for k, v in d.items() if k in known}) for d in j.days)


# (e) the pool package never mutates the live journal.
def test_pool_never_references_live_mutators():
    pat = re.compile(r"add_position|close_position|append_day|(?<!book)\.cash\s*[-+]?=")
    for f in POOL_DIR.rglob("*.py"):
        hits = [l for l in f.read_text().splitlines() if pat.search(l)]
        assert hits == [], (f, hits)


# (f) step refuses the live id.
def test_step_refuses_live_id():
    spec = LineSpec(LIVE_ID, "connors", {}, registered_on="2026-10-16", hypothesis=Hypothesis("beats_live"))
    with pytest.raises(ValueError):
        pool_book.step(pool_book.Book(name=LIVE_ID), spec, ConnorsStrategy({}), {}, DayContext(), "2026-01-01")


def _row(rsi=5.0, sma_5=110.0):
    return pd.Series({"sma_200": 50.0, "sma_5": sma_5, "rsi_2": rsi, "adx_14": 30.0})


# (g) data guard skips a flagged symbol's entry but still exits an open position in it.
def test_data_guard_skips_entry_not_exit():
    spec = LineSpec("guarded", "connors", {}, registered_on="2026-10-16", hypothesis=Hypothesis("beats_live"))
    assert spec.book.data_guard
    b = pool_book.Book(name="guarded", initialized=True, init_date="2026-01-01", cash=8_500.0, open_positions=[dict(
        symbol="MATICUSDT", entry_date="2026-01-01", entry_price=95.0, shares=15.0, entry_cost=1.0)])
    snap = {"MATICUSDT": (100.0, _row(sma_5=90.0)), "TRXUSDT": (100.0, _row()), "ETHUSDT": (100.0, _row())}
    ctx = DayContext(data_flags={"stale": ["MATICUSDT"], "gap200": ["TRXUSDT"]})
    day = pool_book.step(b, spec, ConnorsStrategy({}), snap, ctx, "2026-01-03")
    assert [t["symbol"] for t in b.closed_trades] == ["MATICUSDT"]
    assert [p["symbol"] for p in b.open_positions] == ["ETHUSDT"]
    assert day["blocked"]["data"] == 1 and day["blocked_syms"] == [["TRXUSDT", 5.0]] and day["signals"] == 2


def test_context_gap200_and_trend():
    bars = synthetic_bars()
    gapped = bars["XRPUSDT"].drop(bars["XRPUSDT"].index[-50])
    assert contiguous_bars(gapped) == 49 and contiguous_bars(bars["ETHUSDT"]) == len(bars["ETHUSDT"])
    ctx = build_context({**bars, "XRPUSDT": gapped}, "2026-01-01", ["SOLUSDT"], "BTCUSDT", 1.0, 2.0)
    d = ctx.to_dict()
    assert d["data_flags"] == {"stale": ["SOLUSDT"], "gap200": ["XRPUSDT"]}
    assert d["bar_date"] == "2025-12-31" and d["dow"] == "Wednesday" and d["btc_vol20"] > 0
    assert d["btc_trend"] in ("below_both", "below200_above50") and d["basket_ret20_bucket"] is None


# (h) registry validation.
def test_registry_validation():
    hyp = Hypothesis("beats_live")
    a = LineSpec("x", "connors", {}, registered_on="2026-10-16", hypothesis=hyp, note="waived")
    with pytest.raises(ValueError, match="duplicate"):
        validate((a, a))
    with pytest.raises(ValueError, match="baseline"):
        validate((LineSpec("y", "connors", {}, registered_on="2026-10-17", hypothesis=hyp),))
    with pytest.raises(ValueError, match="hypothesis"):
        validate((LineSpec("z", "connors", {}, registered_on="2026-10-16", note="n"),))
    validate((LineSpec("y", "connors", {}, registered_on="2026-10-17", hypothesis=hyp, baseline="b.json"),))
    assert [l.id for l in LINES] == ["shadow_atr_stop", "shadow_btc200", "shadow_rsi15", "shadow_combo"]
    assert LINES[1].hypothesis == Hypothesis("reduces_dd_in", "below200")
