#!/usr/bin/env python3
"""Repair benchmark series: any non-positive day -> previous day's value. Idempotent.

Usage: python scripts/repair_benchmark_zero_days.py [state/journal.json ...]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:  # pool line ids from the registry (stdlib-only module)
    from trading_engine.paper.pool.registry import LINES
    _LINE_KEYS = tuple(f"{l.id}_value" for l in LINES)
except ImportError:
    _LINE_KEYS = ("shadow_atr_stop_value", "shadow_btc200_value", "shadow_rsi15_value", "shadow_combo_value")

KEYS = ("bh_btc_value", "bh_basket_value", "trend_timed_value", "selector_value",
        # Pool/shadow series: leading 0.0 = not started yet; untouched (no previous positive value).
        *_LINE_KEYS)


def repair_days(days: list[dict]) -> list[tuple[str, str, float]]:
    fixes = []
    for key in KEYS:
        prev = None
        for d in days:
            v = d.get(key)
            if v is not None and v <= 0 and prev is not None:
                d[key] = prev
                fixes.append((d.get("date", "?"), key, prev))
            elif v is not None and v > 0:
                prev = v
    return fixes


def main(paths: list[str]) -> int:
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        fixes = repair_days(data.get("days", []))
        if fixes:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        print(f"{path}: {len(fixes)} fix(es)", *[f"\n  {d} {k} -> {v}" for d, k, v in fixes])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["state/journal.json"]))
