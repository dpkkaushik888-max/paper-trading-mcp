"""Pool scoring with confidence (S35 §4, C3). Read-only over journal + baselines + context history.

Unit = closed trade, net ``pnl_pct``. Methods (pure numpy, seeded, reproducible day-to-day):
- trade-mean CI: cluster bootstrap over entry DATES (all trades of a date travel together), 90 % percentile CI;
- win probability: Beta posterior, prior k=10 pseudo-trades at the backtest win rate (Beta(1,1) without a
  baseline); mean analytic a/(a+b), 90 % credible interval from 20 000 seeded numpy Beta draws (no scipy);
- line vs live: stationary block bootstrap (mean block 5 days) of the mean daily return difference, two-sided
  p = 2 min(P*(mean<=0), P*(mean>=0)); sign test on non-zero days; Holm across active lines at alpha 0.10.
Seeds = sha256(line id | purpose [| dimension | bucket]) -- no date, so CIs do not re-jitter daily
when no new trades arrive. Verdicts live in ``verdicts.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import STARTING_CAPITAL
from .registry import LINES, LIVE_ID, LineSpec

SCHEMA_VERSION = 1
B_VERDICT = 5000
B_CELL = 2000
PRIOR_K = 10
ALPHA = 0.10
MEAN_BLOCK = 5
CI_LEVEL = 0.90
BETA_DRAWS = 20000
BUCKET_PRIOR_MIN_N = 30
MIN_TRADES, USABLE_TRADES, MIN_ENTRY_DAYS = 8, 15, 5
DIMENSIONS = ("btc_trend", "btc_vol20_bucket", "basket_ret20_bucket", "dow")
LIVE_BASELINE = "_s18_reference_com"
CAVEAT_2021 = ("Backtest priors come from .com 2020-2025 history in which 2021 dominates the returns; the prior "
               "is capped at k=10 pseudo-trades so 15+ live trades outweigh it. The 20-coin universe is today's "
               "survivors (optimistic for every line equally). Baselines are .com, live is .us (~2 pp/100 d).")
CAVEAT_COND = "Condition table is exploratory and uncorrected; it never drives a verdict."


def stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "big")


def evidence_tier(n_trades: int, n_entry_days: int) -> str:
    if n_trades < MIN_TRADES or n_entry_days < MIN_ENTRY_DAYS:
        return "insufficient"
    return "indicative" if n_trades < USABLE_TRADES else "usable"


def _pct_ci(samples: np.ndarray, level: float = CI_LEVEL) -> tuple[float, float]:
    q = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(samples, [q, 100.0 - q])
    return float(lo), float(hi)


# ── bootstraps ──────────────────────────────────────────────────────────────
def cluster_bootstrap_means(values, clusters, B: int, rng: np.random.Generator) -> np.ndarray:
    """B bootstrap means of ``values``, resampling whole clusters (entry dates) with replacement."""
    values = np.asarray(values, dtype=float)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    sums = np.bincount(inv, weights=values)
    cnts = np.bincount(inv).astype(float)
    idx = rng.integers(0, len(sums), size=(B, len(sums)))
    return sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)


def cluster_bootstrap_ci(values, clusters, B: int, seed: int) -> Optional[tuple[float, float]]:
    if len(values) == 0:
        return None
    return _pct_ci(cluster_bootstrap_means(values, clusters, B, np.random.default_rng(seed)))


def stationary_block_indices(n: int, B: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """(B, n) indices of a Politis-Romano stationary bootstrap (geometric blocks, wrap-around), vectorised."""
    new = rng.random((B, n)) < 1.0 / mean_block
    new[:, 0] = True
    starts = rng.integers(0, n, size=(B, n))
    pos = np.broadcast_to(np.arange(n), (B, n))
    last = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    return (np.take_along_axis(starts, last, axis=1) + (pos - last)) % n


def block_bootstrap_test(diff, B: int, seed: int, mean_block: float = MEAN_BLOCK) -> dict:
    d = np.asarray(diff, dtype=float)
    idx = stationary_block_indices(len(d), B, mean_block, np.random.default_rng(seed))
    means = d[idx].mean(axis=1)
    lo, hi = _pct_ci(means)
    # (count + 1) / (B + 1): a bootstrap p is never exactly 0.
    le, ge = int((means <= 0).sum()), int((means >= 0).sum())
    p = min(1.0, 2.0 * min((le + 1) / (B + 1), (ge + 1) / (B + 1)))
    return dict(mean_diff=float(d.mean()), ci_lo=lo, ci_hi=hi, p_boot=p)


def sign_test(diff) -> dict:
    d = np.asarray(diff, dtype=float)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return dict(n_pos=0, n_neg=0, p_sign=1.0)
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return dict(n_pos=pos, n_neg=neg, p_sign=min(1.0, 2.0 * tail))


def holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def beta_posterior(wins: int, n: int, p_prior: Optional[float], k: float = PRIOR_K, seed: int = 0,
                   draws: int = BETA_DRAWS) -> dict:
    if p_prior is None:
        a0 = b0 = 1.0
    else:
        p = min(max(float(p_prior), 0.01), 0.99)
        a0, b0 = k * p, k * (1.0 - p)
    a, b = a0 + wins, b0 + (n - wins)
    lo, hi = _pct_ci(np.random.default_rng(seed).beta(a, b, size=draws))
    return dict(a=a, b=b, mean=a / (a + b), lo=lo, hi=hi, prior_p=p_prior,
                disagrees=bool(p_prior is not None and n > 0 and not lo <= p_prior <= hi))


def max_drawdown(values) -> float:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return 0.0
    peak = np.maximum.accumulate(v)
    return float(np.max((peak - v) / peak))


STREAK_FLOOR, STREAK_ALPHA, STREAK_DEFAULT_P = 10, 0.01, 0.5


def streak_threshold(p_win: Optional[float]) -> int:
    """HARD-KILL losing-streak threshold (S35 D2): max(10, smallest k with (1 - p)^k < 0.01), p = baseline
    full-period win rate (0.5 without a baseline -> k = 7 -> floor 10)."""
    p = STREAK_DEFAULT_P if p_win is None else min(max(float(p_win), 0.01), 0.99)
    k = math.floor(math.log(STREAK_ALPHA) / math.log(1.0 - p)) + 1
    while (1.0 - p) ** k >= STREAK_ALPHA:   # guard float edge cases
        k += 1
    return max(STREAK_FLOOR, k)


# Primary hypothesis tests built so far (S35 D3); reduces_dd_in / earns_when_live_idle come in C5.
PRIMARY_TESTS_BUILT = ("beats_live", "control")


# ── data model ──────────────────────────────────────────────────────────────
@dataclass
class LineScore:
    id: str
    family: str = ""
    status: str = "active"
    hypothesis: Optional[str] = None
    waived: bool = False
    registered_on: Optional[str] = None
    start: Optional[str] = None
    days_active: int = 0
    n_trades: int = 0
    n_entry_days: int = 0
    win_rate: Optional[float] = None
    mean_pnl_pct: Optional[float] = None
    median_pnl_pct: Optional[float] = None
    profit_factor: Optional[float] = None   # None + no_losses=True when there are no losing trades
    no_losses: bool = False
    expectancy: Optional[float] = None
    mean_mae_pct: Optional[float] = None
    mean_mfe_pct: Optional[float] = None
    total_return: Optional[float] = None
    current_dd: float = 0.0
    max_dd: float = 0.0
    mean_exposure: Optional[float] = None
    max_consec_losses: int = 0
    current_consec_losses: int = 0
    signals_90d: int = 0
    lineday_rows_90d: int = 0
    streak_threshold: int = 10
    primary_test_available: bool = True
    tier: str = "insufficient"
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None
    win_post: dict = field(default_factory=dict)
    paired: Optional[dict] = None
    holm_adjusted_p: Optional[float] = None
    significant_negative: bool = False
    significant_positive: bool = False
    years_beating_s18: Optional[int] = None
    guard_effect: Optional[float] = None
    verdict: str = ""
    rule: str = ""
    reasons: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    auto_sentence: str = ""


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(x[k]) for k in sorted(x, key=str)}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), 6) if math.isfinite(x) else None
    return x


@dataclass
class PoolReport:
    as_of: Optional[str]
    live: LineScore
    lines: list[LineScore]
    holm: list[dict]
    cond_table: dict
    notes: list[str]
    params: dict
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return _clean(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n"


# ── journal extraction ──────────────────────────────────────────────────────
def _jd(journal) -> dict:
    return journal.to_dict() if hasattr(journal, "to_dict") else journal


def _path(days: list[dict], line_id: str, start: Optional[str]) -> list[tuple[str, float, Optional[dict]]]:
    """(date, value, LineDay|None) for non-skipped days with value > 0 from ``start`` (inclusive)."""
    out = []
    for d in days:
        if d.get("skipped") or (start and d["date"] < start):
            continue
        ld = (d.get("lines") or {}).get(line_id)
        if line_id == LIVE_ID:
            v = d.get("portfolio_value")
        else:
            v = (ld or {}).get("value") or d.get(f"{line_id}_value")
        if v and v > 0:
            out.append((d["date"], float(v), ld))
    return out


def _streaks(trades: list[dict]) -> tuple[int, int]:
    best = cur = 0
    for t in sorted(trades, key=lambda t: (t["exit_date"], t["entry_date"])):
        cur = cur + 1 if t["pnl_pct"] < 0 else 0
        best = max(best, cur)
    return best, cur


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _trade_stats(s: LineScore, trades: list[dict], seed_key: str, B: int, p_prior: Optional[float]) -> None:
    pnl = np.array([t["pnl_pct"] for t in trades], dtype=float)
    dates = [t["entry_date"] for t in trades]
    s.n_trades, s.n_entry_days = len(trades), len(set(dates))
    s.tier = evidence_tier(s.n_trades, s.n_entry_days)
    s.max_consec_losses, s.current_consec_losses = _streaks(trades)
    wins = int((pnl > 0).sum())
    s.win_post = beta_posterior(wins, len(pnl), p_prior, seed=stable_seed(seed_key, "beta"))
    if not len(pnl):
        return
    s.win_rate = wins / len(pnl)
    s.mean_pnl_pct, s.median_pnl_pct = float(pnl.mean()), float(np.median(pnl))
    gains, losses = pnl[pnl > 0], pnl[pnl < 0]
    s.no_losses = not len(losses)
    s.profit_factor = None if s.no_losses else float(gains.sum() / abs(losses.sum()))
    avg_w = float(gains.mean()) if len(gains) else 0.0
    avg_l = float(abs(losses.mean())) if len(losses) else 0.0
    s.expectancy = s.win_rate * avg_w - (1.0 - s.win_rate) * avg_l
    s.mean_mae_pct = _mean(t.get("mae_pct") for t in trades)
    s.mean_mfe_pct = _mean(t.get("mfe_pct") for t in trades)
    if s.tier != "insufficient":
        s.ci_lo, s.ci_hi = cluster_bootstrap_ci(pnl, dates, B, stable_seed(seed_key, "cluster"))


def _returns(path: list[tuple[str, float, Any]], dates: list[str]) -> np.ndarray:
    v = dict((d, val) for d, val, _ in path)
    x = np.array([v[d] for d in dates], dtype=float)
    return x[1:] / x[:-1] - 1.0


def paired_vs_live(path, live_path, B: int, seed: int) -> Optional[dict]:
    common = sorted({d for d, _, _ in path} & {d for d, _, _ in live_path})
    if len(common) < 3:
        return None
    diff = _returns(path, common) - _returns(live_path, common)
    res = block_bootstrap_test(diff, B, seed)
    res.update(sign_test(diff))
    h = len(diff) // 2
    lv, vv = dict((d, x) for d, x, _ in live_path), dict((d, x) for d, x, _ in path)
    res.update(n_days=len(common), start=common[0], end=common[-1],
               half1_mean=float(diff[:h].mean()) if h else None, half2_mean=float(diff[h:].mean()),
               cum_ret_diff=(vv[common[-1]] / vv[common[0]]) - (lv[common[-1]] / lv[common[0]]),
               maxdd_diff=max_drawdown([vv[d] for d in common]) - max_drawdown([lv[d] for d in common]))
    return res


def _bucket(t: dict, dim: str, day_ctx: dict, history: dict) -> str:
    for src in (t.get("entry_ctx") or {}, day_ctx.get(t["entry_date"]) or {}, history.get(t["entry_date"]) or {}):
        v = src.get(dim)
        if v is not None and v != "":
            return str(v)
    return "unknown"


def _cond_cells(trades, baseline: Optional[dict], day_ctx, history, seed_key: str, B: int) -> dict:
    full_p = ((baseline or {}).get("full") or {}).get("win_rate")
    out = {}
    for dim in DIMENSIONS:
        groups: dict[str, list[dict]] = {}
        for t in trades:
            groups.setdefault(_bucket(t, dim, day_ctx, history), []).append(t)
        cells = {}
        for b, ts in groups.items():
            pnl = np.array([t["pnl_pct"] for t in ts], dtype=float)
            dates = [t["entry_date"] for t in ts]
            bb = (((baseline or {}).get("buckets") or {}).get(dim) or {}).get(b) or {}
            prior = bb["wins"] / bb["n"] if bb.get("n", 0) >= BUCKET_PRIOR_MIN_N else full_p
            wins = int((pnl > 0).sum())
            tier = evidence_tier(len(ts), len(set(dates)))
            ci = (cluster_bootstrap_ci(pnl, dates, B, stable_seed(seed_key, dim, b))
                  if tier != "insufficient" else None)
            cells[b] = dict(n=len(ts), n_entry_days=len(set(dates)), win_rate=wins / len(ts),
                            mean_pnl_pct=float(pnl.mean()), tier=tier, ci=list(ci) if ci else None,
                            post_mean=beta_posterior(wins, len(ts), prior, draws=1)["mean"], prior_p=prior)
        out[dim] = cells
    return out


def load_baseline(baselines_dir, name: Optional[str]) -> Optional[dict]:
    if not name:
        return None
    p = Path(baselines_dir) / f"{Path(name).stem}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_context_history(path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def _years(baseline: Optional[dict]) -> Optional[int]:
    if not baseline:
        return None
    y = baseline.get("years_beating_s18", (baseline.get("gate") or {}).get("years_beating_s18"))
    return None if y is None else int(y)


def _value_fields() -> set[str]:
    from ..journal import DaySnapshot   # lazy: avoids an import cycle
    return {f.name for f in fields(DaySnapshot) if f.name.endswith("_value")}


def _score_line(spec: Optional[LineSpec], jd: dict, baselines_dir, history, day_ctx, live_path, as_of, B):
    days = jd.get("days", [])
    lid = spec.id if spec else LIVE_ID
    seed_key = lid   # no as_of: seeds are stable day to day
    if spec is None:
        base = load_baseline(baselines_dir, LIVE_BASELINE)
        s = LineScore(LIVE_ID, family="connors", status="live", hypothesis=None)
        path, trades = live_path, list(jd.get("closed_trades", []))
    else:
        base = load_baseline(baselines_dir, spec.baseline)
        s = LineScore(spec.id, family=spec.family, status=spec.status,
                      hypothesis=spec.hypothesis.kind if spec.hypothesis else None,
                      waived=bool(spec.baseline_waiver), registered_on=spec.registered_on)
        s.primary_test_available = bool(spec.hypothesis) and spec.hypothesis.kind in PRIMARY_TESTS_BUILT
        # S34 shadows (declared DaySnapshot *_value field) count from their first journal day.
        is_s34 = f"{spec.id}_value" in _value_fields()
        path = _path(days, spec.id, None if is_s34 else spec.registered_on)
        start = path[0][0] if path else None
        trades = [t for t in (jd.get("benchmarks", {}).get(spec.id) or {}).get("closed_trades", [])
                  if start and t["entry_date"] >= start]
    s.start = path[0][0] if path else None
    s.days_active = len(path)
    s.years_beating_s18 = _years(base)
    vals = [v for _, v, _ in path]
    cap = None
    if vals and spec is not None:
        # pool line: measure from the book's starting capital, not the first marked value, but only when
        # the path starts on the book's init date (else pre-path P&L would be counted).
        bk = jd.get("benchmarks", {}).get(spec.id) or {}
        if bk.get("init_date") == path[0][0]:
            cap = float(bk.get("starting_capital") or STARTING_CAPITAL)
    if vals:
        if cap is not None:
            vals = [cap] + vals
        s.total_return = vals[-1] / vals[0] - 1.0
        s.max_dd = max_drawdown(vals)
        s.current_dd = 1.0 - vals[-1] / max(vals)
    if spec is None:
        by_date = {d["date"]: d for d in days}
        s.mean_exposure = _mean(((ld or {}).get("exposure") if ld else
                                 (v - by_date[dt].get("cash", v)) / v) for dt, v, ld in path)
    else:
        s.mean_exposure = _mean((ld or {}).get("exposure") for _, _, ld in path)
    trailing = days[-90:]
    reg = None if spec is None else spec.registered_on
    rows = [(d.get("lines") or {}).get(lid) for d in trailing
            if not d.get("skipped") and not (reg and d.get("date", "") < reg)]
    s.lineday_rows_90d = sum(1 for r in rows if r)
    s.signals_90d = int(sum((r or {}).get("signals") or 0 for r in rows))
    p_backtest = ((base or {}).get("full") or {}).get("win_rate")
    s.streak_threshold = streak_threshold(p_backtest)
    _trade_stats(s, trades, seed_key, B, p_backtest)
    if spec is not None:
        ppath = path
        if cap is not None:   # day-1 return (cap -> v1) pairs with live's return over the same day
            prev = [x for x in live_path if x[0] < path[0][0]]
            if prev:
                ppath = [(prev[-1][0], cap, None)] + path
        s.paired = paired_vs_live(ppath, live_path, B, stable_seed(seed_key, "paired"))
        if spec.hypothesis and spec.hypothesis.kind == "control" and s.paired:
            s.guard_effect = s.paired["cum_ret_diff"]
    cond = _cond_cells(trades, base, day_ctx, history, seed_key, B_CELL if B >= B_CELL else B)
    return s, cond


def score_pool(journal, baselines_dir, context_history: dict, B: int = B_VERDICT,
               lines: tuple[LineSpec, ...] = LINES) -> PoolReport:
    from .verdicts import apply_verdicts   # lazy: verdicts imports LineScore from here
    jd = _jd(journal)
    days = jd.get("days", [])
    as_of = days[-1]["date"] if days else None
    day_ctx = {d["date"]: d.get("context") or {} for d in days}
    live_path = _path(days, LIVE_ID, None)
    live, live_cond = _score_line(None, jd, baselines_dir, context_history, day_ctx, live_path, as_of, B)
    scores, cond_table = [], {LIVE_ID: live_cond}
    for spec in lines:
        if spec.status not in ("active", "candidate"):
            continue
        s, cond = _score_line(spec, jd, baselines_dir, context_history, day_ctx, live_path, as_of, B)
        scores.append(s)
        cond_table[spec.id] = cond
    # Holm family: active lines with a paired p, control excluded (it measures data handling, not a claim).
    pv = {s.id: s.paired["p_boot"] for s in scores
          if s.status == "active" and s.paired and s.hypothesis != "control"}
    adj = holm_adjust(pv)
    holm = []
    for s in scores:
        if s.id in adj:
            s.holm_adjusted_p = adj[s.id]
            s.significant_negative = adj[s.id] < ALPHA and s.paired["mean_diff"] < 0
            s.significant_positive = adj[s.id] < ALPHA and s.paired["mean_diff"] > 0
            holm.append(dict(id=s.id, p_boot=pv[s.id], p_adj=adj[s.id], mean_diff=s.paired["mean_diff"],
                             significant=adj[s.id] < ALPHA))
    holm.sort(key=lambda r: (r["p_boot"], r["id"]))
    apply_verdicts([live] + scores)
    notes = [CAVEAT_2021, CAVEAT_COND, "Verdicts are recommendations only; the owner rules monthly (S35 §5)."]
    insuff = sorted(s.id for s in scores if s.tier == "insufficient")
    if insuff:
        notes.append(f"insufficient n (<{MIN_TRADES} trades or <{MIN_ENTRY_DAYS} entry days): {', '.join(insuff)}")
    unpaired = sorted(s.id for s in scores if s.status == "active" and not s.paired)
    if unpaired:
        notes.append(f"no paired test yet (<3 common days with live): {', '.join(unpaired)}")
    params = dict(B=B, streak_floor=STREAK_FLOOR, streak_alpha=STREAK_ALPHA, dormant="rows>=30 and signals<5*rows/90", B_cell=min(B, B_CELL), prior_k=PRIOR_K, alpha=ALPHA, mean_block=MEAN_BLOCK,
                  ci_level=CI_LEVEL, beta_draws=BETA_DRAWS, bucket_prior_min_n=BUCKET_PRIOR_MIN_N,
                  tiers=dict(insufficient=f"<{MIN_TRADES} trades or <{MIN_ENTRY_DAYS} entry days",
                             indicative=f"{MIN_TRADES}-{USABLE_TRADES - 1}", usable=f">={USABLE_TRADES}"))
    return PoolReport(as_of=as_of, live=live, lines=scores, holm=holm, cond_table=cond_table, notes=notes,
                      params=params)
