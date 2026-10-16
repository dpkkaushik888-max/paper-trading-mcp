"""Registration gate (S35 §7): a new line is active only if its full-history baseline beats the S18
reference in >= 4 of 7 years OR its per-bucket evidence supports its stated primary hypothesis."""

from __future__ import annotations

MIN_YEARS_BEATING = 4


def years_beating(line_years: dict, ref_years: dict) -> tuple[int, int]:
    ys = sorted(set(line_years) & set(ref_years))
    return sum(line_years[y]["ret"] > ref_years[y]["ret"] for y in ys), len(ys)


def gate_result(hypothesis_kind: str, beats: int, n_years: int, evidence: dict) -> dict:
    """``evidence``: {"idle_mean": float|None, "idle_n": int} for earns_when_live_idle,
    {"line_maxdd": float, "bench_maxdd": float} for reduces_dd_in."""
    if beats >= MIN_YEARS_BEATING:
        return dict(passed=True, by=f"beats S18 in {beats}/{n_years} years", status="active")
    hyp_ok = False
    if hypothesis_kind == "earns_when_live_idle":
        hyp_ok = bool(evidence.get("idle_n")) and (evidence.get("idle_mean") or 0.0) > 0.0
    elif hypothesis_kind == "reduces_dd_in":
        lm, bm = evidence.get("line_maxdd"), evidence.get("bench_maxdd")
        hyp_ok = lm is not None and bm is not None and lm < bm
    if hyp_ok:
        return dict(passed=True, by=f"hypothesis {hypothesis_kind} supported ({beats}/{n_years} years)",
                    status="active")
    return dict(passed=False, by=f"{beats}/{n_years} years; hypothesis {hypothesis_kind} not supported",
                status="candidate")
