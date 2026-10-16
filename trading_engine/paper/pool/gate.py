"""Registration gate (S35 §7): a new line is active only if its full-history baseline beats the S18
reference in >= 4 of 7 years OR its per-bucket evidence supports its stated primary hypothesis.
``control`` lines (live rules + one isolated change, e.g. the data guard) are exempt from the years rule:
they pass only through an explicit baseline waiver."""

from __future__ import annotations

MIN_YEARS_BEATING = 4


def years_beating(line_years: dict, ref_years: dict) -> tuple[int, int]:
    ys = sorted(set(line_years) & set(ref_years))
    return sum(line_years[y]["ret"] > ref_years[y]["ret"] for y in ys), len(ys)


def gate_result(hypothesis_kind: str, beats: int, n_years: int, evidence: dict, waiver: str | None = None) -> dict:
    """``evidence``: {"idle_mean": float|None, "idle_n": int} for earns_when_live_idle,
    {"line_maxdd": float, "comparators": {name: maxdd}} for reduces_dd_in (conditional MaxDD; the line must be
    lower than EVERY comparator). ``waiver`` is the line's baseline_waiver (used only by ``control``)."""
    if hypothesis_kind == "control":
        if waiver:
            return dict(passed=True, by=f"control waiver ({beats}/{n_years} years): {waiver}", status="active")
        return dict(passed=False, by=f"control line without a waiver ({beats}/{n_years} years)", status="candidate")
    if beats >= MIN_YEARS_BEATING:
        return dict(passed=True, by=f"beats S18 in {beats}/{n_years} years", status="active")
    hyp_ok = False
    if hypothesis_kind == "earns_when_live_idle":
        hyp_ok = bool(evidence.get("idle_n")) and (evidence.get("idle_mean") or 0.0) > 0.0
    elif hypothesis_kind == "reduces_dd_in":
        lm, comps = evidence.get("line_maxdd"), evidence.get("comparators") or {}
        hyp_ok = lm is not None and bool(comps) and all(v is not None and lm < v for v in comps.values())
    if hyp_ok:
        return dict(passed=True, by=f"hypothesis {hypothesis_kind} supported ({beats}/{n_years} years)",
                    status="active")
    return dict(passed=False, by=f"{beats}/{n_years} years; hypothesis {hypothesis_kind} not supported",
                status="candidate")
