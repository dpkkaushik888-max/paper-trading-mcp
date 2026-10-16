"""Line families. ``make_strategy(spec)`` returns the LineStrategy for a registered line."""

from __future__ import annotations

from ..registry import LineSpec


def make_strategy(spec: LineSpec):
    if spec.family == "connors":
        from .connors import ConnorsStrategy
        return ConnorsStrategy(spec.params)
    if spec.family == "breakout":
        from .breakout import BreakoutStrategy
        return BreakoutStrategy(spec.params)
    if spec.family == "xsec_momentum":
        from .xsec_momentum import XsecMomentumStrategy
        return XsecMomentumStrategy(spec.params)
    if spec.family == "basket_trend":
        from .basket_trend import BasketTrendStrategy
        return BasketTrendStrategy(spec.params)
    raise ValueError(f"{spec.id}: unknown family {spec.family!r}")
