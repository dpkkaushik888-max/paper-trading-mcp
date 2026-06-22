"""S22 Stage 4 — composite allocation, roll-up, halt handling."""

import pytest

from loops.allocators import confidence_weighted, risk_parity
from loops.composite import CompositeLoop
from loops.contracts import Cadence, Mandate, Report
from loops.stub import StubLeaf


def _children():
    return [
        StubLeaf("L2.crypto", period_return=0.05, confidence=0.8, max_drawdown=0.02),
        StubLeaf("L2.equity", period_return=-0.01, confidence=0.3, max_drawdown=0.06),
    ]


def test_confidence_weighted_tilts_to_stronger_child():
    children = _children()
    prior = {
        "L2.crypto": Report.from_values("L2.crypto", "p", 100, 105, confidence=0.8, max_drawdown=0.02),
        "L2.equity": Report.from_values("L2.equity", "p", 100, 99, confidence=0.3, max_drawdown=0.06),
    }
    w = confidence_weighted(children, prior)
    assert w["L2.crypto"] > w["L2.equity"]
    assert sum(w.values()) == pytest.approx(1.0)


def test_halted_child_gets_zero_weight():
    children = _children()
    prior = {
        "L2.crypto": Report.from_values("L2.crypto", "p", 100, 105, confidence=0.8, max_drawdown=0.02),
        "L2.equity": Report.from_values("L2.equity", "p", 100, 90, confidence=0.5,
                                        max_drawdown=0.10, halted=True),
    }
    w = confidence_weighted(children, prior)
    assert w["L2.equity"] == 0.0
    assert w["L2.crypto"] == pytest.approx(1.0)  # reallocated to sibling


def test_no_history_falls_back_equal():
    children = _children()
    w = confidence_weighted(children, {})  # no prior reports
    assert w["L2.crypto"] == pytest.approx(0.5)
    assert w["L2.equity"] == pytest.approx(0.5)


def test_risk_parity_prefers_lower_drawdown():
    children = _children()
    prior = {
        "L2.crypto": Report.from_values("L2.crypto", "p", 100, 105, max_drawdown=0.02),
        "L2.equity": Report.from_values("L2.equity", "p", 100, 99, max_drawdown=0.20),
    }
    w = risk_parity(children, prior)
    assert w["L2.crypto"] > w["L2.equity"]


def test_composite_runs_and_rolls_up():
    parent = CompositeLoop("L1.allocator", Cadence.WEEKLY, _children())
    parent.set_mandate(Mandate("L1.allocator", "L0", "2026-06-22", 10_000.0))
    rpt = parent.run("2026-06-22")
    assert len(rpt.children) == 2
    # budget split sums to the parent's capital
    assert sum(c.starting_value for c in rpt.children) == pytest.approx(10_000.0)
    # roll-up ending value = sum of children's ending values
    assert rpt.ending_value == pytest.approx(sum(c.ending_value for c in rpt.children))


def test_first_run_with_no_priors_splits_equally():
    # Allocation uses PRIOR-period reports; with none, an equal split is correct.
    parent = CompositeLoop("L1.allocator", Cadence.WEEKLY, _children())
    parent.set_mandate(Mandate("L1.allocator", "L0", "2026-06-22", 10_000.0))
    rpt = parent.run("2026-06-22")
    crypto = next(c for c in rpt.children if c.loop_id == "L2.crypto")
    equity = next(c for c in rpt.children if c.loop_id == "L2.equity")
    assert crypto.starting_value == pytest.approx(equity.starting_value)


def test_budget_follows_prior_performance(tmp_path):
    from loops.state import LoopState
    # Seed last-period reports for each child, then run: budget should tilt.
    LoopState("L2.crypto", base_dir=tmp_path).record_report(
        Report.from_values("L2.crypto", "2026-06-15", 5000, 5300, confidence=0.8, max_drawdown=0.02))
    LoopState("L2.equity", base_dir=tmp_path).record_report(
        Report.from_values("L2.equity", "2026-06-15", 5000, 4900, confidence=0.3, max_drawdown=0.08))
    parent = CompositeLoop("L1.allocator", Cadence.WEEKLY, _children(),
                           state=LoopState("L1.allocator", base_dir=tmp_path))
    parent.set_mandate(Mandate("L1.allocator", "L0", "2026-06-22", 10_000.0))
    rpt = parent.run("2026-06-22")
    crypto = next(c for c in rpt.children if c.loop_id == "L2.crypto")
    equity = next(c for c in rpt.children if c.loop_id == "L2.equity")
    assert crypto.starting_value > equity.starting_value  # tilts to stronger prior
