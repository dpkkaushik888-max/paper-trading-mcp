"""Locked-holdout promotion judge (S25 D3/D4).

The existential risk of this loop is multiple testing: evaluate K candidates against
one holdout and ~0.05·K pass by luck. Two deterministic defences live here:

1. **Deflated-Sharpe threshold.** The Sharpe bar a candidate must clear on the
   holdout is *raised* by the expected maximum Sharpe achievable by chance across
   the number of candidates that reached the holdout (López de Prado extreme-value
   correction; Bonferroni in spirit). More trials → higher bar.
2. **Sub-period robustness.** A single lucky stretch is rejected: the candidate must
   show non-negative alpha (CAGR − buy-and-hold CAGR) in ≥ ``min_positive`` of
   ``n_subperiods`` contiguous holdout slices, AND beat buy-and-hold overall.

The holdout is read once per candidate here (D4). No tuning, no re-runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from trading_engine.discovery.candidate import CandidateSpec
from trading_engine.discovery.search import buy_hold, cagr_of, run_candidate_window

EULER_MASCHERONI = 0.5772156649015329


# ── inverse standard-normal CDF (Acklam) — no scipy dependency ──────────────
def norm_ppf(p: float) -> float:
    """Inverse CDF of the standard normal. Acklam's rational approximation."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ── deflated-Sharpe machinery ───────────────────────────────────────────────
def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent trials whose
    estimates each have sampling std ``sr_std`` under the SR=0 null.

    Extreme-value (Gumbel) approximation used by the Deflated Sharpe Ratio:
        E[max] ≈ sr_std · [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]
    """
    if n_trials <= 1:
        return 0.0
    z = ((1 - EULER_MASCHERONI) * norm_ppf(1 - 1.0 / n_trials)
         + EULER_MASCHERONI * norm_ppf(1 - 1.0 / (n_trials * math.e)))
    return sr_std * z


def deflated_sharpe_threshold(
    base_threshold: float, n_trials: int, n_obs: int, periods_per_year: int = 365,
) -> float:
    """Raise the single-trial Sharpe bar for multiple testing.

    The annualized-Sharpe estimator has std ≈ √(periods_per_year / n_obs) under the
    SR=0 null (iid). The bar rises by the expected max chance-Sharpe across n_trials.
    """
    sr_std = math.sqrt(periods_per_year / max(n_obs, 2))
    return base_threshold + expected_max_sharpe(n_trials, sr_std)


# ── sub-period robustness ────────────────────────────────────────────────────
def _subperiod_bounds(data: dict, start, end, n: int) -> list[tuple]:
    """Split the trading dates in [start, end] into n contiguous (a, b) slices."""
    all_dates = sorted(set().union(*[df.index for df in data.values()])) if data else []
    window = [d for d in all_dates if start <= d <= end]
    if len(window) < n:
        return [(start, end)]
    bounds = []
    size = len(window) // n
    for i in range(n):
        lo = i * size
        hi = (i + 1) * size - 1 if i < n - 1 else len(window) - 1
        bounds.append((window[lo], window[hi]))
    return bounds


def subperiod_alphas(
    spec: CandidateSpec, data: dict, feats: dict, start, end,
    n: int = 3, capital: float = 10_000.0, benchmark: Optional[str] = None,
    alpha_mode: str = "raw_cagr",
) -> list[float]:
    """Per-slice excess over buy-and-hold across n holdout slices.

    raw_cagr     → candidate CAGR − BH CAGR.
    risk_adjusted→ candidate Sharpe − BH Sharpe (S26 D2).
    """
    alphas = []
    for a, b in _subperiod_bounds(data, start, end, n):
        res, days = run_candidate_window(spec, data, feats, a, b, capital=capital)
        bh = buy_hold(data, a, b, capital=capital, symbol=benchmark)
        if alpha_mode == "risk_adjusted":
            alphas.append(res.sharpe - bh["sharpe"])
        else:
            alphas.append(cagr_of(res.final_value, days, capital) - bh["cagr"])
    return alphas


@dataclass
class HoldoutVerdict:
    candidate_id: str
    return_pct: float
    cagr: float
    sharpe: float
    max_dd: float
    n_trades: int
    bh_cagr: float
    bh_sharpe: float
    alpha: float
    alpha_mode: str
    deflated_threshold: float
    sharpe_pass: bool
    subperiod_alphas: list
    n_subperiods_positive: int
    robustness_pass: bool
    alpha_pass: bool
    promoted: bool
    reasons: list

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id, "return_pct": self.return_pct,
            "cagr": self.cagr, "sharpe": self.sharpe, "max_dd": self.max_dd,
            "n_trades": self.n_trades, "bh_cagr": self.bh_cagr,
            "bh_sharpe": self.bh_sharpe, "alpha": self.alpha, "alpha_mode": self.alpha_mode,
            "deflated_threshold": self.deflated_threshold, "sharpe_pass": self.sharpe_pass,
            "subperiod_alphas": self.subperiod_alphas,
            "n_subperiods_positive": self.n_subperiods_positive,
            "robustness_pass": self.robustness_pass, "alpha_pass": self.alpha_pass,
            "promoted": self.promoted, "reasons": self.reasons,
        }


def evaluate_holdout(
    spec: CandidateSpec, data: dict, feats: dict, ho_start, ho_end,
    *, n_trials: int, base_sharpe: float = 1.0, capital: float = 10_000.0,
    periods_per_year: int = 365, n_subperiods: int = 3, min_positive: int = 2,
    benchmark: Optional[str] = None, alpha_mode: str = "raw_cagr",
) -> HoldoutVerdict:
    """Judge one candidate on the locked holdout (read once). Promotes only if it
    clears the deflated-Sharpe bar, beats buy-and-hold overall, AND is robust across
    sub-periods. ``n_trials`` = candidates that reached the holdout this run (D4).

    ``alpha_mode`` (S26): "raw_cagr" → beat BH on CAGR; "risk_adjusted" → beat BH on
    Sharpe AND stay profitable (CAGR ≥ 0). The deflated-Sharpe bar is unchanged.
    """
    res, days = run_candidate_window(spec, data, feats, ho_start, ho_end, capital=capital)
    c = cagr_of(res.final_value, days, capital)
    bh = buy_hold(data, ho_start, ho_end, capital=capital, symbol=benchmark)
    alpha = c - bh["cagr"]

    threshold = deflated_sharpe_threshold(base_sharpe, n_trials, days, periods_per_year)
    sharpe_pass = res.sharpe >= threshold

    alphas = subperiod_alphas(spec, data, feats, ho_start, ho_end,
                              n=n_subperiods, capital=capital, benchmark=benchmark,
                              alpha_mode=alpha_mode)
    n_pos = sum(1 for a in alphas if a >= 0.0)
    robustness_pass = n_pos >= min_positive
    if alpha_mode == "risk_adjusted":
        alpha_pass = (res.sharpe >= bh["sharpe"]) and (c >= 0.0)
    else:
        alpha_pass = alpha >= 0.0

    promoted = sharpe_pass and robustness_pass and alpha_pass
    reasons = []
    if not sharpe_pass:
        reasons.append(f"sharpe {res.sharpe:.2f} < deflated bar {threshold:.2f}")
    if not alpha_pass:
        if alpha_mode == "risk_adjusted":
            reasons.append(f"risk-adj: Sharpe {res.sharpe:.2f} < BH {bh['sharpe']:.2f} "
                           f"or CAGR {c*100:+.1f}% < 0")
        else:
            reasons.append(f"alpha {alpha*100:+.1f}% < 0 (lost to buy-and-hold)")
    if not robustness_pass:
        reasons.append(f"robustness {n_pos}/{n_subperiods} positive < {min_positive}")
    if promoted:
        reasons.append("PROMOTED — cleared deflated Sharpe, alpha, and sub-period robustness")

    return HoldoutVerdict(
        candidate_id=spec.id, return_pct=res.return_pct, cagr=c, sharpe=res.sharpe,
        max_dd=res.max_dd, n_trades=res.n_trades, bh_cagr=bh["cagr"], bh_sharpe=bh["sharpe"],
        alpha=alpha, alpha_mode=alpha_mode,
        deflated_threshold=threshold, sharpe_pass=sharpe_pass, subperiod_alphas=alphas,
        n_subperiods_positive=n_pos, robustness_pass=robustness_pass, alpha_pass=alpha_pass,
        promoted=promoted, reasons=reasons,
    )
