#!/usr/bin/env python3
"""S25 — run one Strategy Discovery pass (propose → search → prove → promote).

Fetches daily bars, runs the StrategyDiscoveryLoop on a train+WF | LOCKED HOLDOUT
split, and appends any promotions to the registry manifest with full provenance.
Deterministic by default (no LLM): same --seed + same data → same candidates and
verdicts. Pass --use-agent to let the bounded agent propose (still grammar-validated,
deterministic fallback on any problem).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_discovery.py --trial-budget 10 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.sim_s21_window as oracle  # data layer + universe
from loops.agent import AgentClient
from loops.contracts import Cadence, Mandate
from loops.research_loop import StrategyDiscoveryLoop
from loops.state import LoopState
from trading_engine.discovery.registry_manifest import RegistryManifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="S25 strategy discovery pass")
    p.add_argument("--trial-budget", type=int, default=10, help="max candidates reaching the holdout")
    p.add_argument("--seed", type=int, default=0, help="deterministic candidate seed")
    p.add_argument("--n-candidates", type=int, default=20, help="candidates to propose")
    p.add_argument("--years", type=float, default=5.0, help="years of daily bars")
    p.add_argument("--period", default="2026-06-22", help="ISO date governing this run")
    p.add_argument("--manifest", default="state/discovery/registry.json")
    p.add_argument("--state-base", default=None, help="LoopState base dir (optional)")
    p.add_argument("--use-agent", action="store_true", help="let the bounded agent propose")
    args = p.parse_args(argv)

    print(f"Fetching {args.years}y daily bars for {len(oracle.UNIVERSE)} symbols...")
    bars = oracle.fetch_daily_bars(oracle.UNIVERSE, years=args.years)
    print(f"  {len(bars)}/{len(oracle.UNIVERSE)} symbols fetched\n")

    state = LoopState("L_research", base_dir=args.state_base) if args.state_base else None
    agent = AgentClient(state=state) if args.use_agent else None
    manifest = RegistryManifest(args.manifest)

    loop = StrategyDiscoveryLoop(
        bars=bars, seed=args.seed, n_candidates=args.n_candidates,
        manifest=manifest, state=state, agent=agent,
    )
    loop.set_mandate(Mandate(
        loop_id="L_research", issued_by="owner", period=args.period,
        capital_budget=0.0, constraints={"trial_budget": args.trial_budget},
        horizon_days=30, objective="grow_strategy_registry", notes="discovery pass",
    ))

    rpt = loop.run(args.period)
    d = rpt.diagnostics

    print("── Discovery pass ──")
    print(f"  proposed:        {d['n_proposed']}")
    print(f"  passed WF (G1–4):{d['n_passed_wf']}")
    print(f"  reached holdout: {d['n_reached_holdout']} (trial budget {d['trial_budget']})")
    print(f"  PROMOTED:        {d['n_promoted']}  {d['promoted_ids']}\n")

    print("── Survivors on the LOCKED HOLDOUT ──")
    for s in d["survivors"]:
        mark = "PROMOTED" if s["promoted"] else "rejected"
        print(f"  [{mark:>8}] {s['id']:<14} holdout Sharpe {s['holdout_sharpe']:+.2f} "
              f"vs deflated bar {s['deflated_threshold']:.2f} | alpha {s['alpha']*100:+.1f}% "
              f"| {';'.join(s['reasons'])}")

    print(f"\nManifest: {args.manifest} ({len(manifest.active_entries())} active strategies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
