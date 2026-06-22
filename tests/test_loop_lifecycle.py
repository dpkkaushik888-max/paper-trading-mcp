"""S22 Stage 1 — Loop lifecycle ordering, persistence, dry-run."""

from loops.base import Loop
from loops.contracts import Cadence, Mandate, Report
from loops.state import LoopState


class RecordingLoop(Loop):
    """Minimal concrete Loop that records the order of lifecycle calls."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls: list[str] = []

    def observe(self, period):
        self.calls.append("observe")
        return {"period": period}

    def decide(self, observation):
        self.calls.append("decide")
        return {"plan": "noop"}

    def act(self, decision):
        self.calls.append("act")
        return {}

    def measure(self, period):
        self.calls.append("measure")
        return {"start": 10000.0, "end": 10100.0}

    def report(self, measurement):
        self.calls.append("report")
        return Report.from_values(self.loop_id, "2026-06-22",
                                  measurement["start"], measurement["end"])


def test_lifecycle_order():
    loop = RecordingLoop("L2.test", Cadence.DAILY)
    loop.run("2026-06-22")
    assert loop.calls == ["observe", "decide", "act", "measure", "report"]


def test_run_returns_report():
    loop = RecordingLoop("L2.test", Cadence.DAILY)
    rpt = loop.run("2026-06-22")
    assert isinstance(rpt, Report)
    assert rpt.ending_value == 10100.0


def test_run_persists_report(tmp_path):
    st = LoopState("L2.test", base_dir=tmp_path)
    loop = RecordingLoop("L2.test", Cadence.DAILY, state=st)
    loop.run("2026-06-22")
    assert st.last_report() is not None
    assert st.last_report().ending_value == 10100.0


def test_dry_run_does_not_persist(tmp_path):
    st = LoopState("L2.test", base_dir=tmp_path)
    loop = RecordingLoop("L2.test", Cadence.DAILY, state=st)
    loop.run("2026-06-22", dry_run=True)
    assert st.last_report() is None


def test_set_mandate_records(tmp_path):
    st = LoopState("L2.test", base_dir=tmp_path)
    loop = RecordingLoop("L2.test", Cadence.DAILY, state=st)
    m = Mandate("L2.test", "L1", "2026-06-22", 6000.0)
    loop.set_mandate(m)
    assert loop.mandate == m
    assert st.current_mandate() == m


def test_cadence_coerced_from_str():
    loop = RecordingLoop("L2.test", "weekly")
    assert loop.cadence is Cadence.WEEKLY
