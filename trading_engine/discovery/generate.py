"""Deterministic, grammar-bounded candidate generator (S25 D2).

This is the "invent new + tune known" engine, and the deterministic fallback the
research loop uses when no LLM agent is wired (or when the agent's output is
rejected). Everything it emits is a legal ``CandidateSpec`` over the primitive
grammar — so every candidate compiles to a backtestable ``GeneratedStrategy`` with
no LLM in its hot path.

Determinism: ``propose_candidates(seed, n)`` is a pure function of (seed, n). The
first three candidates are always known templates (Connors, breakout, and the S27
bullish-engulfing reversal) so the verification "a known-good template is
rediscovered" has a path; the rest are sampled from the bounded condition pools below.
"""

from __future__ import annotations

import random

from trading_engine.discovery.candidate import CandidateSpec, ExitSpec
from trading_engine.discovery.primitives import Condition

# ── bounded building blocks (every entry is known-valid by construction) ────
# Each factory returns a fresh Condition so frozen instances are never shared.
ENTRY_POOL = [
    lambda: Condition("rsi_2", "<", 10),
    lambda: Condition("rsi_2", "<", 5),
    lambda: Condition("rsi_14", "<", 35),
    lambda: Condition("rsi_14", "<", 40),
    lambda: Condition("rsi_14", "<", 45),
    lambda: Condition("close", ">", "sma_50"),
    lambda: Condition("close", ">", "sma_200"),
    lambda: Condition("close", "<", "sma_5"),
    lambda: Condition("close", ">", "prior_high_20"),
    lambda: Condition("close", "<", "bb_lower"),
    lambda: Condition("adx_14", ">=", 20),
    lambda: Condition("adx_14", ">=", 25),
    lambda: Condition("volume", ">", "vol_sma_20", 1.5),
    lambda: Condition("macd", ">", "macd_signal"),
    lambda: Condition("roc_20", ">", 0.0),
    # S27 — patterns, sequences, longer-horizon S/R
    lambda: Condition("bullish_engulfing", ">", 0),
    lambda: Condition("hammer", ">", 0),
    lambda: Condition("morning_star", ">", 0),
    lambda: Condition("dragonfly_doji", ">", 0),
    lambda: Condition("piercing_line", ">", 0),
    lambda: Condition("up_streak", ">=", 3),
    lambda: Condition("down_streak", ">=", 3),       # buy after a down-run (dip)
    lambda: Condition("close", ">", "prior_high_55"),
    lambda: Condition("roc_5", ">", 0.0),
    lambda: Condition("sma20_slope", ">", 0.0),
]

EXIT_POOL = [
    lambda: Condition("rsi_14", ">", 55),
    lambda: Condition("rsi_14", ">", 65),
    lambda: Condition("rsi_14", ">", 70),
    lambda: Condition("close", ">", "sma_5"),
    lambda: Condition("close", "<", "sma_10"),
    lambda: Condition("close", ">", "bb_upper"),
    # S27 — bearish patterns / sequences / breakdown
    lambda: Condition("bearish_engulfing", "<", 0),
    lambda: Condition("shooting_star", "<", 0),
    lambda: Condition("evening_star", "<", 0),
    lambda: Condition("up_streak", ">=", 4),
    lambda: Condition("close", "<", "prior_low_55"),
]

SL_CHOICES = [0.05, 0.07, 0.08, 0.10]
HOLD_CHOICES = [5, 10, 15]
TP_CHOICES = [0.0, 0.0, 0.10, 0.15]      # weighted toward "no take-profit"


def _connors_template() -> CandidateSpec:
    return CandidateSpec(
        id="tmpl_connors", name="tmpl_connors",
        entry=[Condition("close", ">", "sma_200"), Condition("rsi_2", "<", 10),
               Condition("close", "<", "sma_5"), Condition("adx_14", ">=", 20)],
        exit=ExitSpec(sl_pct=0.07, max_hold_days=10,
                      exit_conditions=[Condition("close", ">", "sma_5")]),
    )


def _breakout_template() -> CandidateSpec:
    return CandidateSpec(
        id="tmpl_breakout", name="tmpl_breakout",
        entry=[Condition("close", ">", "prior_high_20"),
               Condition("volume", ">", "vol_sma_20", 1.5),
               Condition("adx_14", ">=", 25), Condition("close", ">", "sma_50")],
        exit=ExitSpec(sl_pct=0.08, max_hold_days=15,
                      exit_conditions=[Condition("close", "<", "sma_10")]),
    )


def _pattern_reversal_template() -> CandidateSpec:
    """Price-action reversal: bullish engulfing in an established uptrend (S27)."""
    return CandidateSpec(
        id="tmpl_engulfing", name="tmpl_engulfing",
        entry=[Condition("bullish_engulfing", ">", 0),
               Condition("close", ">", "sma_50"),
               Condition("rsi_14", "<", 50)],
        exit=ExitSpec(sl_pct=0.07, max_hold_days=10,
                      exit_conditions=[Condition("close", ">", "bb_upper"),
                                       Condition("bearish_engulfing", "<", 0)]),
    )


def _invent(rng: random.Random, idx: int) -> CandidateSpec:
    n_entry = rng.randint(1, 3)
    entry = [f() for f in rng.sample(ENTRY_POOL, n_entry)]
    n_exit = rng.randint(0, 1)
    exit_conditions = [f() for f in rng.sample(EXIT_POOL, n_exit)] if n_exit else []
    return CandidateSpec(
        id=f"gen_{idx}", name=f"gen_{idx}",
        entry=entry,
        exit=ExitSpec(
            sl_pct=rng.choice(SL_CHOICES),
            max_hold_days=rng.choice(HOLD_CHOICES),
            tp_pct=rng.choice(TP_CHOICES),
            exit_conditions=exit_conditions,
        ),
        pos_size_pct=0.12,
    )


def propose_candidates(seed: int, n: int) -> list[CandidateSpec]:
    """Return ``n`` deterministic candidate specs for the given seed.

    Candidates 0–2 are the known templates (Connors, breakout, engulfing reversal);
    the remainder are invented from the bounded grammar. Pure function of (seed, n).
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    known = [_connors_template(), _breakout_template(), _pattern_reversal_template()]
    out = known[:n]
    for i in range(len(out), n):
        out.append(_invent(rng, i))
    return out
