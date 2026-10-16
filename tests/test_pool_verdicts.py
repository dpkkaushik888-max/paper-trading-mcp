"""S35 C3 verdict rules (recommendations only), on crafted LineScore inputs."""
import pytest

from trading_engine.paper.pool.scoring import LineScore
from trading_engine.paper.pool.verdicts import FLAG_180, apply_verdicts, decide


def _ls(**kw):
    base = dict(id="x", status="active", hypothesis="beats_live", days_active=30, n_trades=5, max_dd=0.05,
                max_consec_losses=0, years_beating_s18=5, lineday_rows_90d=30, signals_90d=50)
    base.update(kw)
    return LineScore(**base)


def _paired(mean, lo, hi, h1, h2, cum=0.01):
    return dict(mean_diff=mean, ci_lo=lo, ci_hi=hi, half1_mean=h1, half2_mean=h2, cum_ret_diff=cum, p_boot=0.01)


def _promo(**kw):
    d = dict(days_active=120, n_trades=30, paired=_paired(0.001, 0.0002, 0.002, 0.001, 0.001),
             holm_adjusted_p=0.04, significant_positive=True)
    d.update(kw)
    return _ls(**d)


def test_keep_default():
    assert decide(_ls())[:2] == ("KEEP", "KEEP")


@pytest.mark.parametrize("kw", [dict(max_dd=0.26), dict(max_consec_losses=10)])
def test_hard_kill_both_branches(kw):
    v, rule, reasons, _ = decide(_ls(**kw))
    assert v == "HARD-KILL" and any("recommend" in r for r in reasons)


def test_hard_kill_not_at_threshold():
    assert decide(_ls(max_dd=0.25, max_consec_losses=9))[0] == "KEEP"


def test_retire_needs_all_three():
    ok = dict(days_active=60, n_trades=15, significant_negative=True, holm_adjusted_p=0.02,
              paired=_paired(-0.001, -0.002, -0.0005, -0.001, -0.001))
    assert decide(_ls(**ok))[0] == "RETIRE-cand"
    for k, bad in (("days_active", 59), ("n_trades", 14), ("significant_negative", False)):
        assert decide(_ls(**{**ok, k: bad}))[0] != "RETIRE-cand", k


def test_dormant_only_with_full_window():
    assert decide(_ls(lineday_rows_90d=90, signals_90d=4))[0] == "DORMANT"
    assert decide(_ls(lineday_rows_90d=90, signals_90d=5))[0] == "KEEP"
    assert decide(_ls(lineday_rows_90d=40, signals_90d=0))[0] == "KEEP"


def test_promote_and_its_failures():
    assert decide(_promo())[0] == "PROMOTE-cand"
    assert decide(_promo(paired=_paired(0.001, 0.0002, 0.002, 0.002, -0.0001)))[0] != "PROMOTE-cand"  # halves
    assert decide(_promo(years_beating_s18=3))[0] != "PROMOTE-cand"
    assert decide(_promo(years_beating_s18=None))[0] != "PROMOTE-cand"
    assert decide(_promo(waived=True))[0] != "PROMOTE-cand"
    assert decide(_promo(days_active=89))[0] != "PROMOTE-cand"
    assert decide(_promo(n_trades=24))[0] != "PROMOTE-cand"
    # CI lower bound <= 0 and not Holm-significant: no promotion.
    assert decide(_promo(paired=_paired(0.001, -0.0001, 0.002, 0.001, 0.001), significant_positive=False))[0] == "KEEP"
    # Holm-significant positive alone is enough (with the other conditions).
    assert decide(_promo(paired=_paired(0.001, -0.0001, 0.002, 0.001, 0.001)))[0] == "PROMOTE-cand"


def test_control_candidate_live():
    v, _, reasons, _ = decide(_ls(hypothesis="control", guard_effect=-0.012, max_dd=0.5))
    assert v == "CONTROL" and "-1.20 pp" in reasons[0]
    assert decide(_ls(status="candidate"))[0] == "NOT-STEPPED"
    assert decide(_ls(status="live"))[0] == "LIVE"


def test_inconclusive_180_flag():
    v, rule, _, flags = decide(_ls(days_active=180))
    assert (v, rule, flags) == ("KEEP", "INCONCLUSIVE-180", [FLAG_180])
    assert decide(_ls(days_active=179))[3] == []


def test_auto_sentence_and_recommendation_wording():
    s = _ls(id="connors_exit_sma10", n_trades=24, days_active=112, win_rate=0.58, tier="usable",
            win_post=dict(lo=0.41, hi=0.73), paired=_paired(0.0001, -0.001, 0.001, 0.0, 0.0, cum=0.018),
            holm_adjusted_p=0.62)
    k = _ls(max_dd=0.3, id="k")
    apply_verdicts([s, k])
    assert s.auto_sentence == ("connors_exit_sma10: 24 trades over 112 days, win 58 % (CI 41–73 %), vs live +1.8 pp, "
                               "not significant (p_adj 0.62) → KEEP")
    assert "recommendation" in k.auto_sentence and k.verdict == "HARD-KILL"
