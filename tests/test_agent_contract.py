"""S22 Stage 4 — AgentClient validation + deterministic fallback."""

from loops.agent import AgentClient, AgentResponse
from loops.stub import StubLeaf


def _children():
    return [StubLeaf("L2.crypto"), StubLeaf("L2.equity")]


def test_disabled_agent_returns_none():
    ac = AgentClient(enabled=False)
    assert ac.allocate(mandate=None, prior_reports={}, children=_children()) is None


def test_missing_cli_returns_none():
    ac = AgentClient(command="definitely-not-a-real-binary-xyz", enabled=True)
    assert ac.allocate(mandate=None, prior_reports={}, children=_children()) is None


def test_validate_accepts_well_formed_weights():
    resp = AgentResponse(selection={"weights": {"L2.crypto": 0.6, "L2.equity": 0.4}})
    w = AgentClient._validate(resp, ["L2.crypto", "L2.equity"])
    assert w == {"L2.crypto": 0.6, "L2.equity": 0.4}


def test_validate_rejects_wrong_keys():
    resp = AgentResponse(selection={"weights": {"L2.crypto": 1.0}})
    assert AgentClient._validate(resp, ["L2.crypto", "L2.equity"]) is None


def test_validate_rejects_out_of_range():
    resp = AgentResponse(selection={"weights": {"L2.crypto": 1.5, "L2.equity": -0.5}})
    assert AgentClient._validate(resp, ["L2.crypto", "L2.equity"]) is None


def test_validate_rejects_non_normalized():
    resp = AgentResponse(selection={"weights": {"L2.crypto": 0.3, "L2.equity": 0.3}})
    assert AgentClient._validate(resp, ["L2.crypto", "L2.equity"]) is None  # sums 0.6


def test_validate_renormalizes_close_to_one():
    resp = AgentResponse(selection={"weights": {"L2.crypto": 0.5, "L2.equity": 0.51}})
    w = AgentClient._validate(resp, ["L2.crypto", "L2.equity"])
    assert w is not None and abs(sum(w.values()) - 1.0) < 1e-9


def test_parse_extracts_json_from_noise():
    out = 'Here is my answer:\n{"weights": {"L2.crypto": 0.7, "L2.equity": 0.3}, "confidence": 0.6}\nDone.'
    resp = AgentClient._parse(out)
    assert resp is not None
    assert resp.selection["weights"]["L2.crypto"] == 0.7
    assert resp.confidence == 0.6


def test_parse_returns_none_on_garbage():
    assert AgentClient._parse("no json here") is None
    assert AgentClient._parse('{"missing": "weights"}') is None
