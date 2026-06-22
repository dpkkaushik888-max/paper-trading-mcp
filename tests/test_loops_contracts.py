"""S22 Stage 1 — Mandate/Report contracts + LoopState ledger."""

import pytest

from loops.contracts import Cadence, Mandate, Report
from loops.state import LoopState


class TestMandate:
    def test_round_trip(self):
        m = Mandate(
            loop_id="L2.crypto", issued_by="L1.allocator", period="2026-06-22",
            capital_budget=6000.0, risk_limits={"max_drawdown_pct": 0.15},
            constraints={"long_only": True}, horizon_days=30, notes="recovery regime",
        )
        assert Mandate.from_dict(m.to_dict()) == m
        assert Mandate.from_json(m.to_json()) == m

    def test_defaults(self):
        m = Mandate("L2.crypto", "L1", "2026-06-22", 6000.0)
        assert m.risk_limits == {} and m.constraints == {}
        assert m.objective == "risk_adjusted_return"


class TestReport:
    def test_round_trip_flat(self):
        r = Report.from_values("L2.crypto", "2026-06-22", 10000.0, 10535.69,
                               n_trades=5, confidence=0.8, regime="BULL")
        assert r.period_return == pytest.approx(0.053569, rel=1e-4)
        assert Report.from_dict(r.to_dict()) == r
        assert Report.from_json(r.to_json()) == r

    def test_nested_children_roll_up(self):
        child_a = Report.from_values("L2.crypto", "2026-06-22", 6000.0, 6300.0)
        child_b = Report.from_values("L2.equity", "2026-06-22", 4000.0, 3900.0)
        parent = Report.from_values("L1.allocator", "2026-06-22", 10000.0, 10200.0,
                                    children=[child_a, child_b])
        restored = Report.from_dict(parent.to_dict())
        assert len(restored.children) == 2
        assert restored.children[0] == child_a
        assert restored.children[1] == child_b
        assert isinstance(restored.children[0], Report)

    def test_halt_fields(self):
        r = Report.from_values("L2.crypto", "2026-06-22", 10000.0, 8400.0,
                               halted=True, halt_reason="drawdown 16%")
        assert r.halted and "drawdown" in r.halt_reason
        assert Report.from_json(r.to_json()).halted


class TestCadence:
    def test_values_serialize_as_str(self):
        assert Cadence.DAILY.value == "daily"
        assert Cadence("weekly") is Cadence.WEEKLY


class TestLoopState:
    def test_mandate_ledger(self, tmp_path):
        st = LoopState("L2.crypto", base_dir=tmp_path)
        assert st.current_mandate() is None
        m1 = Mandate("L2.crypto", "L1", "2026-06-20", 5000.0)
        m2 = Mandate("L2.crypto", "L1", "2026-06-22", 6000.0)
        st.record_mandate(m1)
        st.record_mandate(m2)
        # current = latest by period
        assert st.current_mandate() == m2

    def test_report_ledger(self, tmp_path):
        st = LoopState("L2.crypto", base_dir=tmp_path)
        assert st.last_report() is None
        r = Report.from_values("L2.crypto", "2026-06-22", 10000.0, 10500.0)
        st.record_report(r)
        assert st.last_report() == r

    def test_last_reports_reads_siblings(self, tmp_path):
        # parent reads each child's latest report from sibling dirs
        crypto = LoopState("L2.crypto", base_dir=tmp_path)
        equity = LoopState("L2.equity", base_dir=tmp_path)
        crypto.record_report(Report.from_values("L2.crypto", "2026-06-22", 6000.0, 6300.0))
        equity.record_report(Report.from_values("L2.equity", "2026-06-22", 4000.0, 3900.0))
        parent = LoopState("L1.allocator", base_dir=tmp_path)
        got = parent.last_reports(["L2.crypto", "L2.equity"])
        assert set(got) == {"L2.crypto", "L2.equity"}
        assert got["L2.crypto"].ending_value == 6300.0

    def test_agent_call_persisted(self, tmp_path):
        st = LoopState("L1.allocator", base_dir=tmp_path)
        st.record_agent_call("2026-06-22", "allocate",
                             {"options": [1, 2]}, {"selection": 1})
        f = tmp_path / "L1.allocator" / "agent" / "2026-06-22_allocate.json"
        assert f.exists() and "selection" in f.read_text()
