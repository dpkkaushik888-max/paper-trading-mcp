"""S23 Stage 3 — CryptoAssetLoop adapts the leaf to the Loop contract."""

import numpy as np
import pandas as pd

from loops.contracts import Mandate, Report
from loops.l2_crypto import CryptoAssetLoop
from trading_engine.strategies.connors_strategy import ConnorsSwingStrategy, default_config as a_cfg


def _synthetic_bars(n=260, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-09-01", periods=n, freq="D")
    price = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    df = pd.DataFrame({
        "Open": price, "High": price * 1.01, "Low": price * 0.99,
        "Close": price, "Volume": rng.uniform(1e6, 2e6, n),
    }, index=idx)
    return df


def test_loop_runs_and_returns_report():
    bars = {"AAA": _synthetic_bars(seed=1), "BBB": _synthetic_bars(seed=2)}
    loop = CryptoAssetLoop(
        strategies=[ConnorsSwingStrategy(a_cfg())],
        bars=bars,
    )
    loop.set_mandate(Mandate("L2.crypto", "L1.allocator", "2026-06-22", 10_000.0,
                             risk_limits={"max_concurrent": 8, "per_strategy_cap": 4,
                                          "pos_size_pct": 0.12}))
    rpt = loop.run("2026-06-22")
    assert isinstance(rpt, Report)
    assert rpt.loop_id == "L2.crypto"
    assert rpt.starting_value == 10_000.0
    assert rpt.ending_value > 0
    # report carries diagnostics for the parent allocator
    assert "per_strategy" in rpt.diagnostics
    assert 0.0 <= rpt.confidence <= 1.0


def test_report_round_trips():
    bars = {"AAA": _synthetic_bars(seed=3)}
    loop = CryptoAssetLoop(strategies=[ConnorsSwingStrategy(a_cfg())], bars=bars)
    loop.set_mandate(Mandate("L2.crypto", "L1", "2026-06-22", 10_000.0))
    rpt = loop.run("2026-06-22")
    assert Report.from_json(rpt.to_json()).ending_value == rpt.ending_value
