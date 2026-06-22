"""S25 — registry manifest: promote, provenance, rollback, strategy loading."""

from trading_engine.discovery.candidate import CandidateSpec, ExitSpec, GeneratedStrategy
from trading_engine.discovery.primitives import Condition
from trading_engine.discovery.registry_manifest import (
    RegistryManifest, load_registry_strategies,
)


def _spec(cid="c1"):
    return CandidateSpec(
        id=cid, name=cid,
        entry=[Condition("rsi_14", "<", 40)],
        exit=ExitSpec(sl_pct=0.07, max_hold_days=10),
    )


def _prov():
    return {"holdout": {"sharpe": 1.8}, "trial_count": 5, "deflated_threshold": 1.6}


def test_promote_appends_with_version_and_provenance(tmp_path):
    m = RegistryManifest(tmp_path / "reg.json")
    e1 = m.promote(_spec("a"), _prov(), period="2026-06-22")
    e2 = m.promote(_spec("b"), _prov(), period="2026-06-22")
    assert e1.version == 1 and e2.version == 2
    assert e1.provenance["deflated_threshold"] == 1.6
    # persisted + reloads
    m2 = RegistryManifest(tmp_path / "reg.json")
    assert {e.candidate["id"] for e in m2.entries} == {"a", "b"}


def test_active_specs_reconstructs_candidates(tmp_path):
    m = RegistryManifest(tmp_path / "reg.json")
    m.promote(_spec("a"), _prov(), period="p")
    specs = m.active_specs()
    assert len(specs) == 1
    assert isinstance(specs[0], CandidateSpec)
    assert specs[0].entry[0].lhs == "rsi_14"


def test_rollback_is_reversible_audit(tmp_path):
    path = tmp_path / "reg.json"
    m = RegistryManifest(path)
    m.promote(_spec("a"), _prov(), period="p")
    assert m.rollback("a", reason="paper-forward failed") is True
    assert m.active_specs() == []
    # entry retained for audit, status flipped, reason recorded
    m2 = RegistryManifest(path)
    assert len(m2.entries) == 1
    assert m2.entries[0].status == "rolled_back"
    assert "paper-forward failed" in m2.entries[0].notes


def test_rollback_missing_returns_false(tmp_path):
    m = RegistryManifest(tmp_path / "reg.json")
    assert m.rollback("nope") is False


def test_load_registry_strategies(tmp_path):
    path = tmp_path / "reg.json"
    m = RegistryManifest(path)
    m.promote(_spec("a"), _prov(), period="p")
    m.promote(_spec("b"), _prov(), period="p")
    strats = load_registry_strategies(path)
    assert len(strats) == 2
    assert all(isinstance(s, GeneratedStrategy) for s in strats)
    # discovered strategies rank after the fixed A/B/C stack
    assert strats[0].priority == 10 and strats[1].priority == 11
