"""Line families. ``make_strategy(spec)`` returns the LineStrategy for a registered line."""

from __future__ import annotations

from ..registry import LineSpec


def make_strategy(spec: LineSpec):
    if spec.family == "connors":
        from .connors import ConnorsStrategy
        return ConnorsStrategy(spec.params)
    raise ValueError(f"{spec.id}: unknown family {spec.family!r}")
