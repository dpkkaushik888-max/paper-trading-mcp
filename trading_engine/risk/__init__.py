"""Risk management package — exit logic, position sizing, circuit breakers."""

from .exit_manager import ExitManager, ExitDecision, ExitReason

__all__ = ["ExitManager", "ExitDecision", "ExitReason"]
