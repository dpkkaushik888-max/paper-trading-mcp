"""S23 Stage 3 — CryptoLeaf execution math (deterministic, no network)."""

import pandas as pd
import pytest

from trading_engine.engine.crypto_leaf import CryptoLeaf
from trading_engine.engine.orchestrator import Order, PortfolioView

DAY0 = pd.Timestamp("2026-06-01")
DAY1 = pd.Timestamp("2026-06-02")


class FakeOrch:
    """Scripted orchestrator: returns canned orders per call, no regime."""

    def __init__(self):
        self.entries: list[Order] = []
        self.exits: list[Order] = []

    def regime_policy(self, btc_df, day):
        return None, None

    def decide_exits(self, view, snapshot, day):
        return [o for o in self.exits if o.symbol in view.positions]

    def decide_entries(self, view, snapshot, policy, day):
        return [o for o in self.entries if o.symbol not in view.positions]


def enter(sym, size=0.12):
    return Order("enter", sym, "A_connors", "long", size, 0.07, 0.0, "test", 1.0)


def exit_(sym, reason="MR_EXIT"):
    return Order("exit", sym, "A_connors", "long", 0.0, 0.0, 0.0, reason, 1.0)


def test_entry_sizing_and_cost():
    orch = FakeOrch()
    orch.entries = [enter("AAA")]
    leaf = CryptoLeaf(orch, capital=10_000.0, cost_pct=0.0020,
                      slippage_bps=0.0005, sl_slippage_bps=0.0010)
    leaf.run_day({"AAA": (100.0, pd.Series())}, None, DAY0)

    pos = leaf.positions["AAA"]
    fill = 100.0 * 1.0005
    pos_value = 10_000.0 * 0.12
    shares = pos_value / fill
    ec = fill * shares * 0.0020
    assert pos["entry_price"] == pytest.approx(fill)
    assert pos["shares"] == pytest.approx(shares)
    assert leaf.cash == pytest.approx(10_000.0 - (fill * shares + ec))
    assert leaf.total_costs == pytest.approx(ec)


def test_exit_before_entry_ordering_and_pnl():
    orch = FakeOrch()
    # day 0: open AAA
    orch.entries = [enter("AAA")]
    leaf = CryptoLeaf(orch, capital=10_000.0)
    leaf.run_day({"AAA": (100.0, pd.Series())}, None, DAY0)
    entry_fill = leaf.positions["AAA"]["entry_price"]
    shares = leaf.positions["AAA"]["shares"]
    entry_cost = leaf.positions["AAA"]["entry_cost"]

    # day 1: exit AAA at 110
    orch.entries = []
    orch.exits = [exit_("AAA")]
    leaf.run_day({"AAA": (110.0, pd.Series())}, None, DAY1)

    assert "AAA" not in leaf.positions
    t = leaf.closed[0]
    exit_fill = 110.0 * (1.0 - 0.0005)
    exit_cost = exit_fill * shares * 0.0020
    expected_pnl = (exit_fill * shares - exit_cost) - (entry_fill * shares + entry_cost)
    assert t.pnl == pytest.approx(expected_pnl)
    assert t.reason == "MR_EXIT"
    assert t.pnl > 0


def test_sl_exit_has_extra_slippage():
    orch = FakeOrch()
    orch.entries = [enter("AAA")]
    leaf = CryptoLeaf(orch, capital=10_000.0)
    leaf.run_day({"AAA": (100.0, pd.Series())}, None, DAY0)
    shares = leaf.positions["AAA"]["shares"]

    orch.entries = []
    orch.exits = [exit_("AAA", reason="SL")]
    leaf.run_day({"AAA": (90.0, pd.Series())}, None, DAY1)

    t = leaf.closed[0]
    # SL fill uses slippage_bps + sl_slippage_bps = 0.0015
    assert t.exit_price == pytest.approx(90.0 * (1.0 - 0.0015))


def test_close_all_marks_reason_end():
    orch = FakeOrch()
    orch.entries = [enter("AAA")]
    leaf = CryptoLeaf(orch, capital=10_000.0)
    leaf.run_day({"AAA": (100.0, pd.Series())}, None, DAY0)
    leaf.close_all({"AAA": 105.0}, DAY1)
    assert not leaf.positions
    assert leaf.closed[-1].reason == "END"


def test_result_rollup():
    orch = FakeOrch()
    orch.entries = [enter("AAA")]
    leaf = CryptoLeaf(orch, capital=10_000.0)
    leaf.run_day({"AAA": (100.0, pd.Series())}, None, DAY0)
    orch.entries = []
    orch.exits = [exit_("AAA")]
    leaf.run_day({"AAA": (110.0, pd.Series())}, None, DAY1)
    r = leaf.result()
    assert r.n_trades == 1 and r.wins == 1 and r.losses == 0
    assert r.final_value == leaf.cash
    assert "A_connors" in r.per_strategy
    assert r.per_strategy["A_connors"]["trades"] == 1


def test_capital_is_fixed_sizing_base():
    # sizing uses starting capital, not current equity (matches oracle)
    orch = FakeOrch()
    orch.entries = [enter("AAA"), enter("BBB")]
    leaf = CryptoLeaf(orch, capital=10_000.0)
    leaf.run_day({"AAA": (100.0, pd.Series()), "BBB": (100.0, pd.Series())}, None, DAY0)
    # both positions sized off 10_000 * 0.12 = 1200 notional each
    assert leaf.positions["AAA"]["shares"] == pytest.approx(leaf.positions["BBB"]["shares"])
