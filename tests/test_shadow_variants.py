"""Shadow variant portfolios: each variant changes behaviour as specified; the live S18
series stays byte-identical to master (golden captured from b6fd53b on the same fixture)."""
import json

import pandas as pd
import pytest

from trading_engine.paper import run_daily, shadow
from trading_engine.paper.journal import Journal
from trading_engine.paper.shadow import ShadowBook, ShadowVariant, SHADOW_VARIANTS, SHADOW_VALUE_KEYS
from trading_engine.strategies import connors_swing as cs
from trading_engine.strategies.connors_swing import ConnorsOverrides
from tests.paper_fixtures import GOLDEN_PATH, live_signature, replay_live, synthetic_bars

V = {v.name: v for v in SHADOW_VARIANTS}
D0 = pd.Timestamp("2026-01-01")


def _row(rsi=5.0, sma_200=50.0, sma_5=110.0, adx=30.0):
    return pd.Series({"sma_200": sma_200, "sma_5": sma_5, "rsi_2": rsi, "adx_14": adx})


# ── Live-path identity ──────────────────────────────────────────────────────
def test_live_series_identical_to_master_golden():
    j = replay_live(run_daily, synthetic_bars())
    assert live_signature(j) == json.loads(GOLDEN_PATH.read_text())
    assert len(j.closed_trades) > 0


def test_no_override_shadow_reproduces_live_trades(monkeypatch):
    base = ShadowVariant("shadow_baseline")
    monkeypatch.setattr(run_daily, "SHADOW_VARIANTS", (base,))
    j = replay_live(run_daily, synthetic_bars())
    book = j.get_benchmark("shadow_baseline")
    assert book["closed_trades"] == [t.__dict__ for t in j.closed_trades]
    assert book["cash"] == j.cash
    assert book["value"] == j.days[-1]["portfolio_value"]


def test_shadow_failure_never_affects_live(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("shadow bug")
    monkeypatch.setattr(run_daily, "shadow_step", boom)
    j = replay_live(run_daily, synthetic_bars())
    assert live_signature(j) == json.loads(GOLDEN_PATH.read_text())
    assert all(j.days[-1][k] == 0.0 for k in SHADOW_VALUE_KEYS)


def test_shadows_initialise_and_persist_own_books():
    j = replay_live(run_daily, synthetic_bars(), n_days=30)
    first = j.days[0]
    for v in SHADOW_VARIANTS:
        b = j.get_benchmark(v.name)
        assert b["initialized"] and b["init_date"] == first["date"]
        assert all(d[f"{v.name}_value"] > 0 for d in j.days)
    assert j.cash == live_signature(j)["cash"]


# ── Signal-level variant behaviour ──────────────────────────────────────────
def test_defaults_equal_frozen_config():
    o = ConnorsOverrides()
    assert (o.rsi_threshold, o.sl_pct, o.atr_stop_mult) == (cs.RSI_THRESHOLD, cs.SL_PCT, None)


def test_atr_stop_fires_where_7pct_does_not():
    ov = V["shadow_atr_stop"].overrides
    row = _row(sma_5=200.0)
    # -4 %: under the 7 % stop, but below entry - 3*ATR (100 - 3 = 97).
    assert cs.long_exit(96.0, row, 100.0, D0, D0 + pd.Timedelta(days=1)) is None
    assert cs.long_exit(96.0, row, 100.0, D0, D0 + pd.Timedelta(days=1), overrides=ov, entry_atr=1.0) == "SL"


def test_7pct_fires_where_atr_stop_does_not():
    ov = V["shadow_atr_stop"].overrides
    row = _row(sma_5=200.0)
    # -8 %: live 7 % stop fires; ATR stop at 100 - 3*5 = 85 does not.
    assert cs.long_exit(92.0, row, 100.0, D0, D0 + pd.Timedelta(days=1)) == "SL"
    assert cs.long_exit(92.0, row, 100.0, D0, D0 + pd.Timedelta(days=1), overrides=ov, entry_atr=5.0) is None


def test_atr_variant_emergency_stop_25pct():
    ov = V["shadow_atr_stop"].overrides
    row = _row(sma_5=200.0)
    t = D0 + pd.Timedelta(days=1)
    assert cs.long_exit(76.0, row, 100.0, D0, t, overrides=ov, entry_atr=20.0) is None   # stop at 40
    assert cs.long_exit(75.0, row, 100.0, D0, t, overrides=ov, entry_atr=20.0) == "SL"   # -25 %


def test_rsi15_admits_rsi_between_10_and_15():
    row = _row(rsi=12.0)
    assert cs.long_entry(100.0, row) is False
    assert cs.long_entry(100.0, row, overrides=V["shadow_rsi15"].overrides) is True
    assert cs.long_entry(100.0, _row(rsi=16.0), overrides=V["shadow_rsi15"].overrides) is False


# ── Book-level behaviour (BTC gate, combo) ──────────────────────────────────
def _step(variant, snapshot, btc_close, btc_sma, atr=None, book=None, day="2026-01-01"):
    book = book or ShadowBook(name=variant.name)
    shadow.step(book, variant, snapshot, atr or {}, btc_close, btc_sma, day)
    return book


def test_btc_gate_blocks_entry_below_sma200():
    snap = {"ETHUSDT": (100.0, _row())}
    assert _step(V["shadow_btc200"], snap, 90.0, 100.0).open_positions == []
    assert _step(V["shadow_btc200"], snap, None, None).open_positions == []      # unknown -> closed
    assert [p["symbol"] for p in _step(V["shadow_btc200"], snap, 110.0, 100.0).open_positions] == ["ETHUSDT"]
    # The un-gated variants enter regardless of BTC.
    assert len(_step(V["shadow_rsi15"], snap, 90.0, 100.0).open_positions) == 1


def test_btc_gate_does_not_force_exits():
    snap = {"ETHUSDT": (100.0, _row())}
    b = _step(V["shadow_btc200"], snap, 110.0, 100.0, atr={"ETHUSDT": 1.0})
    b = _step(V["shadow_btc200"], {"ETHUSDT": (99.0, _row(sma_5=200.0))}, 90.0, 100.0, book=b, day="2026-01-02")
    assert len(b.open_positions) == 1 and b.closed_trades == []


def test_combo_applies_all_three():
    combo = V["shadow_combo"]
    assert combo.btc_sma_gate
    assert combo.overrides == ConnorsOverrides(rsi_threshold=15, sl_pct=0.25, atr_stop_mult=3.0)
    snap = {"ETHUSDT": (100.0, _row(rsi=12.0))}
    assert _step(combo, snap, 90.0, 100.0).open_positions == []           # gate blocks
    b = _step(combo, snap, 110.0, 100.0, atr={"ETHUSDT": 1.0})           # RSI 12 admitted
    assert len(b.open_positions) == 1
    entry = b.open_positions[0]["entry_price"]
    # -4 % next day: ATR stop (entry - 3) fires, 7 % would not.
    b = _step(combo, {"ETHUSDT": (entry - 4.0, _row(rsi=12.0, sma_5=200.0))}, 110.0, 100.0,
              book=b, day="2026-01-02")
    assert [t["reason"] for t in b.closed_trades] == ["SL"]


def test_shadow_sizing_cap_and_fees_match_live():
    snap = {f"S{i}USDT": (100.0, _row()) for i in range(8)}
    b = _step(ShadowVariant("x"), snap, None, None)
    assert len(b.open_positions) == run_daily.MAX_CONCURRENT
    p = b.open_positions[0]
    assert p["entry_price"] == 100.0 * (1 + run_daily.SLIPPAGE_BPS)
    assert p["shares"] * p["entry_price"] == pytest.approx(run_daily.STARTING_CAPITAL * run_daily.POS_SIZE_PCT)


# ── Persistence / carry-forward ─────────────────────────────────────────────
def test_carry_forward_includes_shadows():
    j = Journal.init_fresh()
    out = run_daily.carry_forward_benchmarks(j)
    assert all(out[k] == 0.0 for k in SHADOW_VALUE_KEYS)       # not started -> stays 0
    j.days.append({"date": "2026-01-01", "portfolio_value": 1.0, **{k: 10_050.0 for k in SHADOW_VALUE_KEYS}})
    j.days.append({"date": "2026-01-02", "portfolio_value": 1.0, **{k: 0.0 for k in SHADOW_VALUE_KEYS}})
    out = run_daily.carry_forward_benchmarks(j)
    assert all(out[k] == 10_050.0 for k in SHADOW_VALUE_KEYS)


def test_repair_script_covers_shadows():
    import importlib.util, pathlib
    p = pathlib.Path(__file__).parent.parent / "scripts" / "repair_benchmark_zero_days.py"
    spec = importlib.util.spec_from_file_location("repair", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    days = [{"date": "a", "shadow_rsi15_value": 0.0}, {"date": "b", "shadow_rsi15_value": 10_000.0},
            {"date": "c", "shadow_rsi15_value": 0.0}]
    m.repair_days(days)
    assert [d["shadow_rsi15_value"] for d in days] == [0.0, 10_000.0, 10_000.0]
