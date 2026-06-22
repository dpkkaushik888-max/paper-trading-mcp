"""Recursive feedback-loop framework (S22).

Control plane for the loop-engineering redesign: a hierarchy of feedback-control
loops (Personal Finance ⊃ Investment ⊃ {Equity, Crypto} ⊃ strategies). Every
loop obeys one contract — a Mandate flows down, a Report flows up — so loops
compose recursively.

This package depends on ``trading_engine`` (the execution plane); the reverse
must never hold.
"""

from .contracts import Cadence, Mandate, Report

__all__ = ["Cadence", "Mandate", "Report"]
