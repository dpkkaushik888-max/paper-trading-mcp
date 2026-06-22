"""S25 — StrategyDiscoveryLoop: D1 pipeline, trial budget, determinism (offline)."""

import numpy as np
import pandas as pd

from loops.contracts import Mandate
from loops.research_loop import StrategyDiscoveryLoop, _split_dates
from trading_engine.discovery.registry_manifest import RegistryManifest


def _bars(n=500, seed=0, drift=0.001):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    price = 100 * np.cumprod(1 + rng.normal(drift, 0.02, n))
    return pd.DataFrame({"Open": price, "High": price * 1.01, "Low": price * 0.99,
                         "Close": price, "Volume": rng.uniform(1e6, 2e6, n)}, index=idx)


def _data():
    return {"BTCUSDT": _bars(seed=1, drift=0.0012), "ETHUSDT": _bars(seed=2, drift=0.001)}


def _mandate(trial_budget=5):
    return Mandate(loop_id="L_research", issued_by="owner", period="2026-06-22",
                   capital_budget=0.0, constraints={"trial_budget": trial_budget})


def _run(tmp_path, seed=0, n=8, trial_budget=5, manifest=None):
    bars = _data()
    m = manifest or RegistryManifest(tmp_path / "reg.json")
    loop = StrategyDiscoveryLoop(bars=bars, seed=seed, n_candidates=n, manifest=m)
    loop.set_mandate(_mandate(trial_budget))
    return loop.run("2026-06-22"), m


def test_split_three_way():
    bars = _data()
    start, train_end, wf_end, end = _split_dates(bars)
    dates = sorted(set().union(*[df.index for df in bars.values()]))
    assert start == dates[0] and end == dates[-1]
    assert start < train_end < wf_end < end


def test_pipeline_produces_report_with_full_log(tmp_path):
    rpt, _ = _run(tmp_path, n=8, trial_budget=5)
    d = rpt.diagnostics
    assert d["n_proposed"] == 8
    assert len(d["all_candidates"]) == 8          # ALL logged — no silent multiple testing
    assert d["n_reached_holdout"] <= 5            # trial budget respected
    assert d["n_promoted"] == len(d["promoted_ids"])


def test_trial_budget_caps_holdout_reach(tmp_path):
    rpt, _ = _run(tmp_path, n=12, trial_budget=2)
    assert rpt.diagnostics["n_reached_holdout"] <= 2


def test_determinism_same_seed_same_promotions(tmp_path):
    rpt1, _ = _run(tmp_path / "a", seed=0, n=8, trial_budget=5)
    rpt2, _ = _run(tmp_path / "b", seed=0, n=8, trial_budget=5)
    # candidate flow + promotions identical
    assert rpt1.diagnostics["promoted_ids"] == rpt2.diagnostics["promoted_ids"]
    assert rpt1.diagnostics["all_candidates"] == rpt2.diagnostics["all_candidates"]
    sv1 = [(s["id"], s["holdout_sharpe"], s["promoted"]) for s in rpt1.diagnostics["survivors"]]
    sv2 = [(s["id"], s["holdout_sharpe"], s["promoted"]) for s in rpt2.diagnostics["survivors"]]
    assert sv1 == sv2


def test_promotions_land_in_manifest_with_provenance(tmp_path):
    rpt, m = _run(tmp_path, n=8, trial_budget=5)
    active = m.active_entries()
    assert len(active) == rpt.diagnostics["n_promoted"]
    for e in active:
        assert "holdout" in e.provenance
        assert "deflated_threshold" in e.provenance
        assert e.period == "2026-06-22"


def test_holdout_only_read_for_survivors(tmp_path):
    # survivors carry a holdout verdict; nothing beyond the budget reaches it
    rpt, _ = _run(tmp_path, n=10, trial_budget=3)
    survivors = rpt.diagnostics["survivors"]
    assert len(survivors) <= 3
    assert all("holdout_sharpe" in s for s in survivors)
