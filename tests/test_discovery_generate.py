"""S25 — deterministic grammar-bounded candidate generation."""

from trading_engine.discovery.generate import propose_candidates
from trading_engine.discovery.primitives import FEATURE_COLUMNS


def test_count_and_known_templates_first():
    cands = propose_candidates(seed=0, n=5)
    assert len(cands) == 5
    assert cands[0].id == "tmpl_connors"
    assert cands[1].id == "tmpl_breakout"


def test_zero_and_one():
    assert propose_candidates(0, 0) == []
    assert len(propose_candidates(0, 1)) == 1


def test_determinism_same_seed():
    a = [c.to_dict() for c in propose_candidates(seed=7, n=12)]
    b = [c.to_dict() for c in propose_candidates(seed=7, n=12)]
    assert a == b


def test_different_seeds_differ():
    a = [c.to_dict() for c in propose_candidates(seed=1, n=12)]
    b = [c.to_dict() for c in propose_candidates(seed=2, n=12)]
    assert a != b   # invented tail diverges


def test_all_candidates_compile_and_are_grammar_valid():
    cands = propose_candidates(seed=3, n=20)
    for c in cands:
        c.to_strategy()                       # must compile to a BaseStrategy
        for cond in c.entry:
            assert cond.lhs in FEATURE_COLUMNS
            assert cond.op in (">", "<", ">=", "<=")
        assert 0 < c.exit.sl_pct < 1
        assert c.exit.max_hold_days >= 1
