"""L_research — Strategy Discovery Loop ("Agent 1", S25).

A slow (monthly) loop one level above the daily crypto engine. Its Report is *an
update to the strategy registry*. The D1 pipeline, mapped onto the Loop lifecycle:

    observe → fetch/split data (train+WF | LOCKED HOLDOUT), load current registry
    decide  → PROPOSE candidates (agent or deterministic grammar) + SEARCH them on
              train+WF only, gate G1–G4, cap survivors to the trial budget
    act     → PROVE survivors on the holdout ONCE (deflated Sharpe + sub-period
              robustness), PROMOTE passers to the registry manifest with provenance
    measure → counts (proposed / passed-WF / reached-holdout / promoted)
    report  → survivors + evidence + regime maps up; mandate (constraints + trial
              budget) flows down

The holdout is read only in ``act`` and only for survivors — never during search
(D4). Determinism: same seed + same data → same candidates + same verdicts.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from trading_engine.discovery import generate
from trading_engine.discovery.candidate import CandidateSpec
from trading_engine.discovery.gates import evaluate_holdout
from trading_engine.discovery.primitives import FEATURE_COLUMNS, build_features
from trading_engine.discovery.registry_manifest import RegistryManifest
from trading_engine.discovery.search import (
    WFGateConfig, WindowedWFConfig, search_walk_forward, search_walk_forward_windows,
)

from .base import Loop
from .contracts import Cadence, Report

TRAIN_PCT, WALKFWD_PCT = 0.60, 0.20   # remainder = locked holdout (mirrors S23 split)


def _split_dates(bars: dict) -> tuple:
    """Return (start, train_end, wf_end, end) over the union of trading dates."""
    dates = sorted(set().union(*[df.index for df in bars.values()])) if bars else []
    if len(dates) < 4:
        raise ValueError("not enough data to split into train/walk-forward/holdout")
    n = len(dates)
    return (dates[0], dates[int(n * TRAIN_PCT)],
            dates[int(n * (TRAIN_PCT + WALKFWD_PCT))], dates[-1])


class StrategyDiscoveryLoop(Loop):
    def __init__(
        self,
        loop_id: str = "L_research",
        bars: Optional[dict[str, pd.DataFrame]] = None,
        seed: int = 0,
        n_candidates: int = 20,
        manifest: Optional[RegistryManifest] = None,
        manifest_path: str = "state/discovery/registry.json",
        state: Any = None,
        agent: Any = None,
        gate_cfg: Optional[WFGateConfig] = None,
        capital: float = 10_000.0,
        base_sharpe: float = 1.0,
        benchmark: Optional[str] = None,
        alpha_mode: str = "raw_cagr",
        clean_oos: bool = True,
        wf_mode: str = "single",                  # "single" | "windowed" (S28)
        windowed_cfg: Optional[WindowedWFConfig] = None,
    ) -> None:
        super().__init__(loop_id, Cadence.MONTHLY, state=state, agent=agent)
        self._bars = bars or {}
        self._seed = seed
        self._n_candidates = n_candidates
        self._manifest = manifest or RegistryManifest(manifest_path)
        self._alpha_mode = alpha_mode
        # WF gate inherits the alpha_mode unless an explicit gate_cfg overrides it.
        self._gate_cfg = gate_cfg or WFGateConfig(alpha_mode=alpha_mode)
        self._capital = capital
        self._base_sharpe = base_sharpe
        self._benchmark = benchmark
        self._clean_oos = clean_oos     # False ⇒ reused/contaminated holdout (S26 D4)
        self._wf_mode = wf_mode
        self._windowed_cfg = windowed_cfg or WindowedWFConfig(alpha_mode=alpha_mode)
        # filled across the lifecycle
        self._search = None
        self._verdicts: list = []
        self._promoted: list = []

    # ── helpers ───────────────────────────────────────────────────────────
    def _trial_budget(self) -> int:
        if self.mandate is not None:
            return int(self.mandate.constraints.get("trial_budget", 10))
        return 10

    def _propose(self) -> list[CandidateSpec]:
        """Agent proposes (bounded grammar); deterministic generator is the fallback."""
        n = self._n_candidates
        if self.agent is not None:
            example = generate.propose_candidates(self._seed, 1)[0].to_dict()
            specs = self.agent.propose_candidates(
                n=n, primitives=FEATURE_COLUMNS, example_spec=example, mandate=self.mandate)
            if specs:
                return specs[:n]
        return generate.propose_candidates(self._seed, n)

    # ── lifecycle ─────────────────────────────────────────────────────────
    def observe(self, period: str) -> dict:
        feats = {sym: build_features(df) for sym, df in self._bars.items()}
        start, train_end, wf_end, end = _split_dates(self._bars)
        return {
            "period": period, "bars": self._bars, "feats": feats,
            "wf_start": start, "wf_end": wf_end,        # train+WF span (no holdout)
            "ho_start": wf_end, "ho_end": end,          # LOCKED HOLDOUT
            "n_active": len(self._manifest.active_entries()),
        }

    def decide(self, observation: dict) -> dict:
        candidates = self._propose()
        if self._wf_mode == "windowed":
            search = search_walk_forward_windows(
                candidates, observation["bars"], observation["feats"],
                observation["wf_start"], observation["wf_end"],
                trial_budget=self._trial_budget(), capital=self._capital,
                cfg=self._windowed_cfg, benchmark=self._benchmark,
            )
        else:
            search = search_walk_forward(
                candidates, observation["bars"], observation["feats"],
                observation["wf_start"], observation["wf_end"],
                trial_budget=self._trial_budget(), capital=self._capital,
                gate_cfg=self._gate_cfg, benchmark=self._benchmark,
            )
        self._search = search
        return {"search": search, **observation}

    def act(self, decision: dict) -> dict:
        """PROVE on the holdout (once per survivor) then PROMOTE passers."""
        search = decision["search"]
        n_trials = len(search.survivors)        # deflation uses #candidates reaching holdout
        period = decision["period"]
        self._verdicts, self._promoted = [], []
        for cr in search.survivors:
            verdict = evaluate_holdout(
                cr.candidate, decision["bars"], decision["feats"],
                decision["ho_start"], decision["ho_end"],
                n_trials=n_trials, base_sharpe=self._base_sharpe,
                capital=self._capital, benchmark=self._benchmark,
                alpha_mode=self._alpha_mode,
            )
            self._verdicts.append((cr, verdict))
            if verdict.promoted:
                provenance = {
                    "walk_forward": cr.to_dict(), "holdout": verdict.to_dict(),
                    "trial_count": n_trials, "deflated_threshold": verdict.deflated_threshold,
                    "seed": self._seed, "alpha_mode": self._alpha_mode,
                    "clean_oos": self._clean_oos,
                }
                self._manifest.promote(
                    cr.candidate, provenance, period=period,
                    notes=f"discovery seed={self._seed} trials={n_trials}")
                self._promoted.append((cr, verdict))
        return {}

    def measure(self, period: str) -> dict:
        s = self._search
        return {
            "n_proposed": s.n_proposed if s else 0,
            "n_passed_wf": s.n_passed_wf if s else 0,
            "n_reached_holdout": len(s.survivors) if s else 0,
            "n_promoted": len(self._promoted),
        }

    def report(self, measurement: dict) -> Report:
        period = self.mandate.period if self.mandate else (self._search and "")
        survivors_evidence = [
            {"id": cr.candidate.id, "name": cr.candidate.name,
             "wf_sharpe": cr.sharpe, "wf_cagr": cr.cagr,
             "promoted": v.promoted, "holdout_sharpe": v.sharpe, "bh_sharpe": v.bh_sharpe,
             "deflated_threshold": v.deflated_threshold, "alpha": v.alpha,
             "alpha_mode": v.alpha_mode, "subperiod_alphas": v.subperiod_alphas,
             "regime_map": cr.candidate.regime_weights, "reasons": v.reasons}
            for cr, v in self._verdicts
        ]
        promoted_ids = [cr.candidate.id for cr, _ in self._promoted]
        return Report(
            loop_id=self.loop_id,
            period=period or "",
            starting_value=self._capital,   # research loop deploys no capital; counts live in diagnostics
            ending_value=self._capital,
            n_trades=0,
            confidence=min(1.0, measurement["n_promoted"] / max(1, self._trial_budget())),
            diagnostics={
                "n_proposed": measurement["n_proposed"],
                "n_passed_wf": measurement["n_passed_wf"],
                "n_reached_holdout": measurement["n_reached_holdout"],
                "n_promoted": measurement["n_promoted"],
                "trial_budget": self._trial_budget(),
                "promoted_ids": promoted_ids,
                "survivors": survivors_evidence,
                "all_candidates": [c.to_dict() for c in (self._search.all_candidates if self._search else [])],
            },
        )
