"""S35 C3 scoring: bootstraps, Beta posterior, tiers, Holm, the report shape and the run_daily hook."""
import json
from datetime import date, timedelta

import numpy as np
import pytest

from trading_engine.paper import run_daily
from trading_engine.paper.pool import scoring as sc
from trading_engine.paper.pool.registry import LIVE_ID, Hypothesis, LineSpec
from tests.paper_fixtures import live_signature, synthetic_bars

B = 200


def test_cluster_ci_constant_is_zero_width_and_symmetric_contains_zero():
    lo, hi = sc.cluster_bootstrap_ci([0.02] * 12, [f"d{i}" for i in range(12)], B, 1)
    assert lo == pytest.approx(0.02) and hi == pytest.approx(0.02)
    vals = [0.05, -0.05] * 10
    lo, hi = sc.cluster_bootstrap_ci(vals, [f"d{i}" for i in range(20)], B, 2)
    assert lo < 0 < hi
    assert sc.cluster_bootstrap_ci([], [], B, 3) is None


def test_cluster_resampling_keeps_same_day_trades_together():
    # Day A = two trades (0.1, 0.3), day B = one trade (-0.2). Cluster means can only be AA, BB or AB.
    means = sc.cluster_bootstrap_means([0.1, 0.3, -0.2], ["A", "A", "B"], 2000, np.random.default_rng(0))
    allowed = np.array([0.2, -0.2, 0.2 / 3])
    assert np.all(np.min(np.abs(means[:, None] - allowed[None, :]), axis=1) < 1e-12)
    assert len(np.unique(np.round(means, 12))) == 3   # a trade-level bootstrap would give e.g. 0.1, 0.3


def test_beta_posterior_k10_prior_analytic():
    post = sc.beta_posterior(6, 10, 0.6, seed=1)
    assert (post["a"], post["b"]) == pytest.approx((12.0, 8.0)) and post["mean"] == pytest.approx(0.6)
    assert post["lo"] < 0.6 < post["hi"] and not post["disagrees"]
    flat = sc.beta_posterior(0, 0, None, seed=1)
    assert (flat["a"], flat["b"]) == (1.0, 1.0)
    assert flat["lo"] == pytest.approx(0.05, abs=0.01) and flat["hi"] == pytest.approx(0.95, abs=0.01)
    assert sc.beta_posterior(0, 30, 0.6, seed=1)["disagrees"]


@pytest.mark.parametrize("n,days,tier", [(7, 7, "insufficient"), (8, 8, "indicative"), (14, 14, "indicative"),
                                         (15, 15, "usable"), (20, 4, "insufficient"), (20, 5, "usable")])
def test_tiers(n, days, tier):
    assert sc.evidence_tier(n, days) == tier


def test_block_bootstrap_seeded_and_shapes():
    rng = np.random.default_rng(5)
    d = rng.normal(0.001, 0.01, 80)
    a, b = sc.block_bootstrap_test(d, B, 42), sc.block_bootstrap_test(d, B, 42)
    assert a == b and a != sc.block_bootstrap_test(d, B, 43)
    assert a["ci_lo"] <= a["mean_diff"] <= a["ci_hi"] and 0.0 <= a["p_boot"] <= 1.0
    idx = sc.stationary_block_indices(50, 400, 5, np.random.default_rng(0))
    assert idx.shape == (400, 50) and idx.min() >= 0 and idx.max() < 50
    contiguous = (np.diff(idx, axis=1) % 50 == 1).mean()
    assert 0.7 < contiguous < 0.9   # P(continue block) = 1 - 1/5
    z = sc.block_bootstrap_test(np.full(30, 0.002), B, 1)
    assert z["ci_lo"] == pytest.approx(0.002) == z["ci_hi"] and z["p_boot"] == 0.0


def test_sign_test_excludes_ties():
    r = sc.sign_test([0.01] * 5 + [0.0] * 7)
    assert (r["n_pos"], r["n_neg"]) == (5, 0) and r["p_sign"] == pytest.approx(2 / 32)
    assert sc.sign_test([0.0, 0.0])["p_sign"] == 1.0


def test_holm_ordering():
    adj = sc.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adj == pytest.approx({"a": 0.04, "c": 0.09, "b": 0.09, "d": 0.5})
    assert sc.holm_adjust({"x": 0.6, "y": 0.7}) == {"x": 1.0, "y": 1.0}


# ── end-to-end on a synthetic journal ──────────────────────────────────────
def _synthetic_journal(n=60):
    rng = np.random.default_rng(7)
    d0 = date(2026, 1, 1)
    dates = [(d0 + timedelta(days=i)).isoformat() for i in range(n)]
    live = 10000 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    line = 10000 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    days = []
    for i, dt in enumerate(dates):
        lines = {LIVE_ID: dict(value=float(live[i]), exposure=0.3, signals=1)}
        if i >= 5:   # the pool line starts on day 5 (registered_on)
            lines["t_pos"] = dict(value=float(line[i]), exposure=0.5, signals=2)
        days.append(dict(date=dt, portfolio_value=float(live[i]), cash=7000.0, skipped=(i == 30),
                         context=dict(btc_trend="above_both", dow="Monday") if i % 2 else {}, lines=lines))
    trades = []
    for k in range(20):
        ed = dates[6 + 2 * (k // 2)]   # two trades per entry date -> 10 entry days
        ctx = {"btc_trend": "below_both"} if k % 3 == 0 else None
        trades.append(dict(symbol=f"S{k}", entry_date=ed, exit_date=dates[min(n - 1, 8 + 2 * (k // 2))],
                           pnl_pct=float(rng.normal(0.01, 0.05)), pnl=1.0, entry_ctx=ctx,
                           mae_pct=-0.02, mfe_pct=0.03))
    trades.append(dict(symbol="OLD", entry_date=dates[1], exit_date=dates[3], pnl_pct=0.5, pnl=1.0))  # pre-start
    live_trades = [dict(symbol="L", entry_date=dates[i], exit_date=dates[i + 1], pnl_pct=0.01 * (-1) ** i, pnl=1.0)
                   for i in range(0, 40, 4)]
    return dict(days=days, closed_trades=live_trades, benchmarks={"t_pos": dict(closed_trades=trades)}), dates


_LINE = LineSpec("t_pos", "connors", {}, registered_on="2026-01-06", baseline="state/pool_baselines/t_pos.json",
                 hypothesis=Hypothesis("beats_live"))
_CTRL = LineSpec("t_ctrl", "connors", {}, registered_on="2026-01-06", hypothesis=Hypothesis("control"),
                 baseline_waiver="control")


def test_score_pool_end_to_end(tmp_path):
    jd, dates = _synthetic_journal()
    (tmp_path / "t_pos.json").write_text(json.dumps(dict(
        full=dict(win_rate=0.6), years_beating_s18=5,
        buckets=dict(btc_trend=dict(below_both=dict(n=40, wins=30, mean=0.01, sd=0.1))))))
    hist = {dates[6]: {"btc_vol20_bucket": "high", "basket_ret20_bucket": "0..+10", "dow": "Wednesday"}}
    rep = sc.score_pool(jd, tmp_path, hist, B=B, lines=(_LINE, _CTRL))
    s = {l.id: l for l in rep.lines}["t_pos"]
    assert rep.as_of == dates[-1] and rep.live.id == LIVE_ID and rep.live.verdict == "LIVE"
    assert s.start == "2026-01-06" and s.days_active == 54    # 55 days from day 5, minus the skipped day
    assert (s.n_trades, s.n_entry_days, s.tier) == (20, 10, "usable")   # pre-start trade excluded
    assert s.ci_lo <= s.mean_pnl_pct <= s.ci_hi and s.mean_mae_pct == pytest.approx(-0.02)
    assert s.win_post["prior_p"] == 0.6 and s.paired["n_days"] == 54 and s.holm_adjusted_p is not None
    assert s.years_beating_s18 == 5 and s.mean_exposure == pytest.approx(0.5) and s.signals_90d == 2 * 55   # LineDay rows in the trailing 90 journal days
    assert s.verdict in ("KEEP", "RETIRE-cand", "PROMOTE-cand") and s.auto_sentence.startswith("t_pos: 20 trades")
    ctrl = {l.id: l for l in rep.lines}["t_ctrl"]
    assert ctrl.verdict == "CONTROL" and ctrl.paired is None and ctrl.n_trades == 0
    cells = rep.cond_table["t_pos"]
    assert cells["btc_trend"]["below_both"]["prior_p"] == pytest.approx(0.75)        # bucket prior (n >= 30)
    assert cells["btc_vol20_bucket"]["high"]["n"] == 2                                # context history lookup
    assert cells["btc_vol20_bucket"]["unknown"]["n"] == 18                          # no ctx anywhere
    assert sum(c["n"] for c in cells["btc_trend"].values()) == 20 and "unknown" in cells["btc_trend"]
    assert any("insufficient" in n for n in rep.notes) and any("2021" in n for n in rep.notes)
    holm_ids = [h["id"] for h in rep.holm]
    assert holm_ids == ["t_pos"]   # the control has no paired window here; only active lines with p enter


def test_to_dict_deterministic_and_json_roundtrip(tmp_path):
    jd, _ = _synthetic_journal()
    a = sc.score_pool(jd, tmp_path, {}, B=B, lines=(_LINE,)).to_json()
    b = sc.score_pool(jd, tmp_path, {}, B=B, lines=(_LINE,)).to_json()
    assert a == b
    d = json.loads(a)
    assert d["schema_version"] == 1 and list(d) == sorted(d)
    assert json.loads(json.dumps(d, sort_keys=True)) == d


def test_real_journal_scores_without_error():
    from pathlib import Path
    root = Path(__file__).parent.parent / "state"
    rep = sc.score_pool(json.loads((root / "journal.json").read_text()), root / "pool_baselines",
                        sc.load_context_history(root / "pool_context_history.json"), B=B)
    assert rep.live.n_trades > 0 and len(rep.lines) == 9


# ── run_daily hook ──────────────────────────────────────────────────────────
def _replay_saving(monkeypatch, tmp_path, n_days=3):
    bars = synthetic_bars()
    j = run_daily.Journal.init_fresh()
    monkeypatch.setattr(run_daily.Journal, "load", classmethod(lambda cls, *a, **k: j))
    monkeypatch.setattr(run_daily.Journal, "save", lambda self, *a, **k: None)
    monkeypatch.setattr(run_daily, "POOL_REPORT_PATH", str(tmp_path / "pool_report.json"))
    rcs = []
    for d in next(iter(bars.values())).index[-n_days:]:
        vis = {s: df[df.index < d] for s, df in bars.items()}
        monkeypatch.setattr(run_daily, "fetch_universe", lambda syms, _v=vis: dict(_v))
        monkeypatch.setattr(run_daily, "_today_utc_date", lambda _d=d: _d.date().isoformat())
        rcs.append(run_daily.run(dry_run=False))
    return j, rcs


def test_run_daily_writes_pool_report(monkeypatch, tmp_path):
    j, rcs = _replay_saving(monkeypatch, tmp_path)
    rep = json.loads((tmp_path / "pool_report.json").read_text())
    assert rcs == [0, 0, 0] and rep["as_of"] == j.days[-1]["date"] and "_scoring" not in j.days[-1]["pool_errors"]


def test_scoring_failure_lands_in_pool_errors(monkeypatch, tmp_path):
    (tmp_path / "a").mkdir()
    ref, _ = _replay_saving(monkeypatch, tmp_path / "a")

    def boom(*a, **k):
        raise RuntimeError("scoring exploded")
    monkeypatch.setattr(run_daily, "score_pool", boom)
    (tmp_path / "b").mkdir()
    j, rcs = _replay_saving(monkeypatch, tmp_path / "b")
    assert rcs == [0, 0, 0] and live_signature(j) == live_signature(ref)
    assert "scoring exploded" in j.days[-1]["pool_errors"]["_scoring"]
    assert not (tmp_path / "b" / "pool_report.json").exists()
