#!/usr/bin/env python3
"""Live demo: the crypto-leaf agent making a bounded strategy-selection call.

Asks the agent (via `claude --print`) how to weight the three fixed strategies
in different regimes. The agent NEVER trades and NEVER edits rules — it returns
per-strategy weights (0=disable, 1=normal, up to 2=lean in) that the engine then
re-validates against the risk box. Falls back to deterministic weights on any
problem.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/demo_crypto_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loops.agent import AgentClient
from loops.contracts import Mandate

STRATEGY_INFO = {
    "A_connors": "Uptrend pullback: buy deep RSI(2) oversold dips while price is "
                 "above its 200-day average (needs an uptrend).",
    "B_breakout": "Breakout continuation: buy new 20-day highs confirmed by volume "
                  "and strong ADX (needs momentum/trending markets).",
    "C_range": "Range mean-reversion: buy very oversold RSI(2)<5 in low-trend, "
               "tight-range conditions (needs choppy/sideways markets).",
}
OPTIONS = list(STRATEGY_INFO)


def ask(client: AgentClient, regime: str, mandate: Mandate) -> None:
    print(f"\n── Regime: {regime} " + "─" * 40)
    weights = client.select_weights(regime, STRATEGY_INFO, OPTIONS, mandate=mandate)
    if weights is None:
        print("  agent returned nothing valid → engine uses deterministic weights "
              "(all 1.0). [fallback path]")
        return
    for name, w in weights.items():
        bar = "█" * int(round(w * 10))
        print(f"  {name:<12} weight {w:>4.2f}  {bar}")


def main() -> int:
    client = AgentClient(enabled=True, timeout=120)
    mandate = Mandate(
        loop_id="L2.crypto", issued_by="L1.allocator", period="2026-06-22",
        capital_budget=6000.0,
        risk_limits={"max_drawdown_pct": 0.15, "max_concurrent": 8, "pos_size_pct": 0.12},
        constraints={"long_only": True}, horizon_days=30,
    )
    print("Crypto-leaf agent — bounded strategy-weighting demo")
    print("(agent picks weights only; it does NOT trade or change rules)")
    for regime in ("BULL", "BEAR", "NEUTRAL"):
        ask(client, regime, mandate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
