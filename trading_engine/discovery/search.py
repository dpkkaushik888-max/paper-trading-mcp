"""Walk-forward search + trial budget (S25 D1 step 2-3, D4).

Each candidate is a concrete ``GeneratedStrategy``; "search" here means evaluating
every proposed candidate **in isolation** over the train+walk-forward window only —
the holdout is NEVER touched here (D4: search never sees the holdout). Candidates
clear single-strategy gates G1–G4 (mirroring scripts/sim_crypto_leaf.py), survivors
are ranked, and only the top ``trial_budget`` reach the locked holdout. ALL proposed
candidates are logged so there is no silent multiple testing.

Layering: discovery → engine (CryptoLeaf + orchestrator), never discovery → scripts.
A candidate is backtested by feeding the leaf a `build_features` snapshot — the
orchestrator forwards that row straight into ``GeneratedStrategy.entry/exit_reason``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from trading_engine.discovery.candidate import CandidateSpec
from trading_engine.discovery.primitives import build_features
from trading_engine.engine.crypto_leaf import CryptoLeaf, LeafResult
from trading_engine.engine.orchestrator import StrategyOrchestrator


# ── walk-forward single-strategy gate thresholds (S23 G1–G4) ────────────────
@dataclass(frozen=True)
class WFGateConfig:
    min_sharpe: float = 0.8       # G1
    min_alpha: float = 0.0        # G2 (raw_cagr): CAGR − BH_CAGR ≥ this
    max_dd: float = 0.25          # G3
    min_trades: int = 10          # G4
    # G2 benchmark mode (S26): "raw_cagr" = beat BH on CAGR (S25 default);
    # "risk_adjusted" = beat BH on Sharpe AND stay profitable (CAGR ≥ 0).
    alpha_mode: str = "raw_cagr"


@dataclass
class CandidateResult:
    """One candidate's walk-forward evaluation — logged whether it passes or not."""
    candidate: CandidateSpec
    return_pct: float
    cagr: float
    sharpe: float
    max_dd: float
    n_trades: int
    bh_cagr: float
    gates: dict           # {"G1": bool, ...}
    passed: bool          # all gates
    rank_score: float
    reached_holdout: bool = False
    reject_reason: Optional[str] = None
    windows: Optional[list] = None   # per-window breakdown (windowed WF, S28)

    def to_dict(self) -> dict:
        return {
            "id": self.candidate.id, "name": self.candidate.name,
            "return_pct": self.return_pct, "cagr": self.cagr, "sharpe": self.sharpe,
            "max_dd": self.max_dd, "n_trades": self.n_trades, "bh_cagr": self.bh_cagr,
            "gates": self.gates, "passed": self.passed, "rank_score": self.rank_score,
            "reached_holdout": self.reached_holdout, "reject_reason": self.reject_reason,
            "windows": self.windows,
        }


@dataclass
class SearchResult:
    all_candidates: list[CandidateResult]   # ALL proposed, logged (D4)
    survivors: list[CandidateResult]        # passed WF AND within trial budget
    trial_budget: int
    n_proposed: int
    n_passed_wf: int

    def to_dict(self) -> dict:
        return {
            "trial_budget": self.trial_budget, "n_proposed": self.n_proposed,
            "n_passed_wf": self.n_passed_wf,
            "survivors": [c.candidate.id for c in self.survivors],
            "all_candidates": [c.to_dict() for c in self.all_candidates],
        }


# ── shared backtest primitives (also imported by gates.py) ──────────────────
def cagr_of(final_value: float, days: int, capital: float = 10_000.0) -> float:
    years = days / 365.25
    return (final_value / capital) ** (1 / years) - 1.0 if years > 0 else 0.0


def buy_hold(
    data: dict, start, end, capital: float = 10_000.0,
    cost_pct: float = 0.0020, symbol: Optional[str] = None,
) -> dict:
    """Buy-and-hold benchmark over [start, end]. Defaults to BTC, else first symbol.

    Mirrors scripts/sim_crypto_leaf.bh_btc so alpha is measured against the same
    baseline the S23 gates used.
    """
    series = None
    if symbol and symbol in data:
        series = data[symbol]["Close"]
    else:
        for key in ("BTC-USD", "BTCUSDT"):
            if key in data:
                series = data[key]["Close"]
                break
        if series is None and data:
            series = data[sorted(data)[0]]["Close"]
    if series is None:
        return {"cagr": 0.0, "sharpe": 0.0, "total_return": 0.0}
    s = series[(series.index >= start) & (series.index <= end)]
    if len(s) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "total_return": 0.0}
    shares = capital * (1 - cost_pct) / float(s.iloc[0])
    vals = shares * s
    total_ret = float(vals.iloc[-1] / capital - 1.0)
    years = len(s) / 365.25
    cagr = (vals.iloc[-1] / capital) ** (1 / years) - 1.0 if years > 0 else 0.0
    dr = vals.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * (365 ** 0.5)) if dr.std() > 0 else 0.0
    return {"cagr": float(cagr), "sharpe": sharpe, "total_return": total_ret}


def run_candidate_window(
    spec: CandidateSpec, data: dict, feats: dict, start, end,
    capital: float = 10_000.0, **leaf_kwargs,
) -> tuple[LeafResult, int]:
    """Backtest one candidate in isolation over [start, end]. Returns (result, days).

    ``feats[sym]`` must be the build_features frame for ``data[sym]`` — its row is
    what the orchestrator hands to GeneratedStrategy.entry/exit_reason.
    """
    strat = spec.to_strategy()
    orch = StrategyOrchestrator(
        strategies=[strat], regime_filter=None, regime_gating=False,
        global_max_concurrent=8, per_strategy_cap=4,
        base_pos_size_pct=spec.pos_size_pct,
    )
    leaf = CryptoLeaf(orch, capital=capital, **leaf_kwargs)
    all_dates = sorted(set().union(*[df.index for df in data.values()])) if data else []
    window = [d for d in all_dates if start <= d <= end]
    for day in window:
        snap = {
            sym: (float(df["Close"].loc[day]), feats[sym].loc[day])
            for sym, df in data.items()
            if day in df.index and sym in feats and day in feats[sym].index
        }
        if snap:
            leaf.run_day(snap, btc_df=None, day=day)
    if window:
        last = window[-1]
        leaf.close_all(
            {sym: float(df["Close"].loc[last]) for sym, df in data.items() if last in df.index},
            last,
        )
    return leaf.result(), max(1, len(window))


# ── per-candidate walk-forward gating ───────────────────────────────────────
def gate_candidate(
    spec: CandidateSpec, data: dict, feats: dict, wf_start, wf_end,
    capital: float = 10_000.0, gate_cfg: WFGateConfig = WFGateConfig(),
    benchmark: Optional[str] = None,
) -> CandidateResult:
    """Evaluate one candidate on the walk-forward window and apply G1–G4."""
    res, days = run_candidate_window(spec, data, feats, wf_start, wf_end, capital=capital)
    c = cagr_of(res.final_value, days, capital)
    bh = buy_hold(data, wf_start, wf_end, capital=capital, symbol=benchmark)
    alpha = c - bh["cagr"]
    if gate_cfg.alpha_mode == "risk_adjusted":
        # beat BH on Sharpe AND make money (S26 D2)
        g2 = (res.sharpe >= bh["sharpe"]) and (c >= 0.0)
    else:
        g2 = alpha >= gate_cfg.min_alpha
    gates = {
        "G1_sharpe": res.sharpe >= gate_cfg.min_sharpe,
        "G2_alpha": g2,
        "G3_maxdd": res.max_dd < gate_cfg.max_dd,
        "G4_trades": res.n_trades >= gate_cfg.min_trades,
    }
    passed = all(gates.values())
    reject = None if passed else ",".join(k for k, v in gates.items() if not v)
    return CandidateResult(
        candidate=spec, return_pct=res.return_pct, cagr=c, sharpe=res.sharpe,
        max_dd=res.max_dd, n_trades=res.n_trades, bh_cagr=bh["cagr"], gates=gates,
        passed=passed, rank_score=res.sharpe, reject_reason=reject,
    )


def select_survivors(results: list[CandidateResult], trial_budget: int) -> list[CandidateResult]:
    """Rank WF-passers and keep the top ``trial_budget`` (D4 trial cap).

    Deterministic ordering: rank_score desc, then CAGR desc, then candidate id asc.
    Mutates the chosen results' ``reached_holdout`` flag and returns them.
    """
    passers = [r for r in results if r.passed]
    passers.sort(key=lambda r: (-r.rank_score, -r.cagr, r.candidate.id))
    survivors = passers[: max(0, trial_budget)]
    chosen = {id(r) for r in survivors}
    for r in passers:
        if id(r) in chosen:
            r.reached_holdout = True
        else:
            r.reject_reason = "trial_budget_exceeded"
    return survivors


def search_walk_forward(
    candidates: list[CandidateSpec], data: dict, feats: dict, wf_start, wf_end,
    *, trial_budget: int = 10, capital: float = 10_000.0,
    gate_cfg: WFGateConfig = WFGateConfig(), benchmark: Optional[str] = None,
) -> SearchResult:
    """Run the full walk-forward search: gate every candidate, cap survivors to budget.

    Deterministic: candidates are processed in the order given and the backtest is
    seed-free, so same candidates + same data → same result. ALL candidates are
    returned in ``all_candidates`` (no silent multiple testing).
    """
    results = [
        gate_candidate(spec, data, feats, wf_start, wf_end,
                       capital=capital, gate_cfg=gate_cfg, benchmark=benchmark)
        for spec in candidates
    ]
    n_passed = sum(1 for r in results if r.passed)
    survivors = select_survivors(results, trial_budget)
    return SearchResult(
        all_candidates=results, survivors=survivors, trial_budget=trial_budget,
        n_proposed=len(candidates), n_passed_wf=n_passed,
    )


# ── multi-window (full-cycle) walk-forward (S28) ────────────────────────────
@dataclass(frozen=True)
class WindowedWFConfig:
    n_windows: int = 4              # slice WF span into this many contiguous windows
    min_windows_pass: int = 3       # candidate must clear ≥ this many windows
    max_dd: float = 0.30            # worst-window max drawdown ceiling
    min_total_trades: int = 12      # summed across windows
    alpha_mode: str = "risk_adjusted"   # per-window pass criterion


def window_bounds(data: dict, start, end, n: int) -> list[tuple]:
    """Split trading dates in [start, end] into n contiguous (a, b) windows."""
    all_dates = sorted(set().union(*[df.index for df in data.values()])) if data else []
    span = [d for d in all_dates if start <= d <= end]
    if len(span) < n or n <= 1:
        return [(start, end)]
    size = len(span) // n
    bounds = []
    for i in range(n):
        lo = i * size
        hi = (i + 1) * size - 1 if i < n - 1 else len(span) - 1
        bounds.append((span[lo], span[hi]))
    return bounds


def _window_pass(sharpe: float, cagr: float, bh: dict, alpha_mode: str) -> bool:
    if alpha_mode == "risk_adjusted":
        return (sharpe >= bh["sharpe"]) and (cagr >= 0.0)
    return cagr >= bh["cagr"]


def gate_candidate_windowed(
    spec: CandidateSpec, data: dict, feats: dict, wf_start, wf_end,
    capital: float = 10_000.0, cfg: WindowedWFConfig = WindowedWFConfig(),
    benchmark: Optional[str] = None,
) -> CandidateResult:
    """Evaluate a candidate across N WF sub-windows and gate on cross-regime consistency.

    Passes iff ≥ ``cfg.min_windows_pass`` windows show non-negative (risk-adjusted)
    alpha, worst-window drawdown is under the ceiling, and total trades clear the floor.
    """
    bounds = window_bounds(data, wf_start, wf_end, cfg.n_windows)
    wins = []
    for a, b in bounds:
        res, days = run_candidate_window(spec, data, feats, a, b, capital=capital)
        c = cagr_of(res.final_value, days, capital)
        bh = buy_hold(data, a, b, capital=capital, symbol=benchmark)
        passed = _window_pass(res.sharpe, c, bh, cfg.alpha_mode)
        wins.append({"start": str(a.date()), "end": str(b.date()), "cagr": c,
                     "sharpe": res.sharpe, "bh_cagr": bh["cagr"], "bh_sharpe": bh["sharpe"],
                     "max_dd": res.max_dd, "n_trades": res.n_trades, "pass": passed})

    n_pass = sum(1 for w in wins if w["pass"])
    total_trades = sum(w["n_trades"] for w in wins)
    worst_dd = max((w["max_dd"] for w in wins), default=0.0)
    gates = {
        "G1_consistency": n_pass >= cfg.min_windows_pass,
        "G3_worst_dd": worst_dd < cfg.max_dd,
        "G4_trades": total_trades >= cfg.min_total_trades,
    }
    passed = all(gates.values())
    reject = None if passed else ",".join(k for k, v in gates.items() if not v)
    mean_cagr = sum(w["cagr"] for w in wins) / len(wins) if wins else 0.0
    mean_sharpe = sum(w["sharpe"] for w in wins) / len(wins) if wins else 0.0
    mean_bh_cagr = sum(w["bh_cagr"] for w in wins) / len(wins) if wins else 0.0
    return CandidateResult(
        candidate=spec, return_pct=0.0, cagr=mean_cagr, sharpe=mean_sharpe,
        max_dd=worst_dd, n_trades=total_trades, bh_cagr=mean_bh_cagr, gates=gates,
        passed=passed, rank_score=float(n_pass), reject_reason=reject, windows=wins,
    )


def search_walk_forward_windows(
    candidates: list[CandidateSpec], data: dict, feats: dict, wf_start, wf_end,
    *, trial_budget: int = 10, capital: float = 10_000.0,
    cfg: WindowedWFConfig = WindowedWFConfig(), benchmark: Optional[str] = None,
) -> SearchResult:
    """Full-cycle walk-forward search: gate on cross-window consistency (S28).

    Survivors rank by windows-passed (rank_score) then mean CAGR — the most
    regime-robust candidates get the scarce holdout slots. ALL candidates logged.
    """
    results = [
        gate_candidate_windowed(spec, data, feats, wf_start, wf_end,
                                capital=capital, cfg=cfg, benchmark=benchmark)
        for spec in candidates
    ]
    n_passed = sum(1 for r in results if r.passed)
    survivors = select_survivors(results, trial_budget)
    return SearchResult(
        all_candidates=results, survivors=survivors, trial_budget=trial_budget,
        n_proposed=len(candidates), n_passed_wf=n_passed,
    )
