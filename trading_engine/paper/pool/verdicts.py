"""Keep / retire / promote RECOMMENDATIONS (S35 §5, C3). Nothing here changes a line's status: every
verdict is a recommendation ("cand") that the owner rules on monthly. Rules, in precedence order:

HARD-KILL      max DD of the value path > 25 % since start OR >= 10 consecutive losing closed trades.
RETIRE-cand    days_active >= 60 AND n_trades >= 15 AND Holm-adjusted paired p < 0.10 with mean diff < 0.
DORMANT        < 5 signals (LineDay.signals) in the trailing 90 journal days, once 90 LineDay rows exist.
PROMOTE-cand   days_active >= 90 AND n_trades >= 25 AND (paired CI lower bound > 0 OR Holm-significant
               positive) AND mean daily diff > 0 in both halves AND baseline years_beating_s18 >= 4;
               control and baseline-waived lines never promote.
INCONCLUSIVE-180  days_active >= 180 and none of the above -> KEEP, flag "inconclusive 180d; owner rules monthly".
KEEP           everything else.
Special: live_s18 -> LIVE; control (hypothesis control) -> CONTROL + guard_effect; status candidate -> NOT-STEPPED.
"""

from __future__ import annotations

from .scoring import LineScore

KILL_DD = 0.25
KILL_LOSSES = 10
RETIRE_DAYS, RETIRE_TRADES = 60, 15
DORMANT_WINDOW, DORMANT_SIGNALS = 90, 5
PROMOTE_DAYS, PROMOTE_TRADES, PROMOTE_YEARS = 90, 25, 4
INCONCLUSIVE_DAYS = 180
FLAG_180 = "inconclusive 180d; owner rules monthly"
REC = "recommendation only; owner rules"


def decide(s: LineScore) -> tuple[str, str, list[str], list[str]]:
    """(verdict, rule, reasons, flags) for one scored line."""
    if s.status == "live":
        return "LIVE", "LIVE", ["frozen S18 live line; reference for every paired test"], []
    if s.status == "candidate":
        return "NOT-STEPPED", "NOT-STEPPED", ["candidate: failed the §7 registration gate; not stepped"], []
    if s.status != "active":
        return s.status.upper(), s.status.upper(), [f"status {s.status}"], []
    if s.hypothesis == "control":
        ge = "n/a" if s.guard_effect is None else f"{s.guard_effect * 100:+.2f} pp"
        return "CONTROL", "CONTROL", [f"control line; guard_effect (control - live return, common window) {ge}"], []
    p = s.paired or {}
    if s.max_dd > KILL_DD or s.max_consec_losses >= KILL_LOSSES:
        r = []
        if s.max_dd > KILL_DD:
            r.append(f"max DD {s.max_dd:.1%} > {KILL_DD:.0%}")
        if s.max_consec_losses >= KILL_LOSSES:
            r.append(f"{s.max_consec_losses} consecutive losing trades >= {KILL_LOSSES}")
        return "HARD-KILL", "HARD-KILL", r + [f"recommend kill; {REC}"], []
    if s.days_active >= RETIRE_DAYS and s.n_trades >= RETIRE_TRADES and s.significant_negative:
        return "RETIRE-cand", "RETIRE-cand", [
            f"{s.days_active} days >= {RETIRE_DAYS}, {s.n_trades} trades >= {RETIRE_TRADES}",
            f"underperforms live: Holm p_adj {s.holm_adjusted_p:.3f} < 0.10, mean diff {p.get('mean_diff', 0):+.4%}/day",
            f"recommend retire; {REC}"], []
    if s.lineday_rows_90d >= DORMANT_WINDOW and s.signals_90d < DORMANT_SIGNALS:
        return "DORMANT", "DORMANT", [f"{s.signals_90d} signals in the trailing {DORMANT_WINDOW} journal days "
                                      f"< {DORMANT_SIGNALS}", f"recommend review; {REC}"], []
    h1, h2 = p.get("half1_mean"), p.get("half2_mean")
    positive = bool(p) and (p.get("ci_lo", 0) > 0 or s.significant_positive)
    halves = h1 is not None and h2 is not None and h1 > 0 and h2 > 0
    years_ok = s.years_beating_s18 is not None and s.years_beating_s18 >= PROMOTE_YEARS
    if (s.days_active >= PROMOTE_DAYS and s.n_trades >= PROMOTE_TRADES and positive and halves and years_ok
            and not s.waived):
        return "PROMOTE-cand", "PROMOTE-cand", [
            f"{s.days_active} days >= {PROMOTE_DAYS}, {s.n_trades} trades >= {PROMOTE_TRADES}",
            f"paired CI [{p['ci_lo']:+.4%}, {p['ci_hi']:+.4%}]/day, p_adj {s.holm_adjusted_p}",
            f"same sign both halves ({h1:+.4%}, {h2:+.4%}); baseline beats S18 {s.years_beating_s18}/7 years",
            f"recommend promotion review (new spec + 90-day clock); {REC}"], []
    if s.days_active >= INCONCLUSIVE_DAYS:
        return "KEEP", "INCONCLUSIVE-180", [f"{s.days_active} days >= {INCONCLUSIVE_DAYS}, no rule fired"], [FLAG_180]
    return "KEEP", "KEEP", ["no rule fired"], []


def _pct(x, digits=0):
    return "n/a" if x is None else f"{x * 100:.{digits}f}"


def sentence(s: LineScore) -> str:
    wp = s.win_post or {}
    if s.n_trades:
        win = f"win {_pct(s.win_rate)} % (CI {_pct(wp.get('lo'))}–{_pct(wp.get('hi'))} %)"
    else:
        win = "no closed trades"
    head = f"{s.id}: {s.n_trades} trades over {s.days_active} days, {win}"
    if s.tier == "insufficient":
        head += f", insufficient (n={s.n_trades})"
    if s.status == "live":
        return f"{head} → LIVE"
    p = s.paired
    if p:
        sig = ("significant" if (s.significant_negative or s.significant_positive) else "not significant")
        pa = "n/a" if s.holm_adjusted_p is None else f"{s.holm_adjusted_p:.2f}"
        head += f", vs live {p['cum_ret_diff'] * 100:+.1f} pp, {sig} (p_adj {pa})"
    else:
        head += ", no paired window yet"
    tail = s.verdict + (f" [{'; '.join(s.flags)}]" if s.flags else "")
    if s.verdict not in ("KEEP", "LIVE", "NOT-STEPPED", "CONTROL"):
        tail += " (recommendation; owner rules)"
    return f"{head} → {tail}"


def apply_verdicts(scores: list[LineScore]) -> None:
    for s in scores:
        s.verdict, s.rule, s.reasons, s.flags = decide(s)
        s.auto_sentence = sentence(s)
