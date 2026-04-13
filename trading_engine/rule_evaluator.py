"""Rule evaluator — loads rules.json and evaluates conditions against indicator data."""

from __future__ import annotations

import json
import operator
import re
from pathlib import Path
from typing import Optional

from .config import RULES_PATH
from .models import Signal


OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

CONDITION_NUM_PATTERN = re.compile(
    r"^(\w+)\s*(<=|>=|<|>|==|!=)\s*(-?[\d.]+)$"
)

CONDITION_IND_PATTERN = re.compile(
    r"^(\w+)\s*(<=|>=|<|>|==|!=)\s*(\w+)$"
)


def load_rules(rules_path: Optional[str] = None) -> dict:
    """Load strategy rules from JSON file."""
    path = Path(rules_path or RULES_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")
    with open(path) as f:
        return json.load(f)


def parse_condition(condition_str: str) -> Optional[tuple]:
    """Parse conditions into (left, operator_func, right).

    Supports:
    - 'rsi_14 < 30' → (indicator_name, op, numeric_threshold)
    - 'price > bb_lower' → (indicator_name, op, other_indicator_name)
    """
    match = CONDITION_NUM_PATTERN.match(condition_str.strip())
    if match:
        indicator = match.group(1)
        op_str = match.group(2)
        threshold = float(match.group(3))
        return (indicator, OPERATORS[op_str], threshold, "num")

    match = CONDITION_IND_PATTERN.match(condition_str.strip())
    if match:
        left = match.group(1)
        op_str = match.group(2)
        right = match.group(3)
        return (left, OPERATORS[op_str], right, "ind")

    return None


def evaluate_condition(condition_str: str, indicators: dict) -> tuple[bool, str]:
    """Evaluate a single condition against indicator values.

    Returns (passed: bool, detail: str).
    """
    parsed = parse_condition(condition_str)
    if not parsed:
        return False, f"Cannot parse condition: {condition_str}"

    left_name, op_func, right_val, mode = parsed

    if left_name not in indicators:
        return False, f"Indicator '{left_name}' not available"

    left_value = indicators[left_name]

    if mode == "ind":
        if right_val not in indicators:
            return False, f"Indicator '{right_val}' not available"
        threshold = indicators[right_val]
    else:
        threshold = right_val

    passed = op_func(left_value, threshold)
    rhs_display = f"{right_val}={threshold}" if mode == "ind" else str(threshold)
    detail = f"{left_name}={left_value} {'✓' if passed else '✗'} {condition_str} (rhs: {rhs_display})"
    return passed, detail


def evaluate_entry_rules(rules: dict, indicators: dict) -> Signal:
    """Evaluate entry rules and return a Signal."""
    entry_rules = rules.get("entry_rules", {})
    long_rules = entry_rules.get("long", [])
    min_score = entry_rules.get("min_score", 0.6)

    score = 0.0
    reasons = []
    all_details = []

    for rule in long_rules:
        condition = rule.get("condition", "")
        weight = rule.get("weight", 0.25)
        description = rule.get("description", "")

        passed, detail = evaluate_condition(condition, indicators)
        all_details.append(detail)

        if passed:
            score += weight
            reasons.append(f"✓ {description} ({condition})")
        else:
            reasons.append(f"✗ {description} ({condition})")

    price = indicators.get("price", 0)
    symbol = indicators.get("symbol", "")

    if score >= min_score:
        return Signal(
            symbol=symbol,
            action="buy",
            strength=score,
            reasons=reasons,
            indicators=indicators,
            price=price,
        )

    return Signal(
        symbol=symbol,
        action="hold",
        strength=score,
        reasons=reasons,
        indicators=indicators,
        price=price,
    )


def evaluate_exit_rules(rules: dict, indicators: dict, position: dict) -> Signal:
    """Evaluate exit rules for an open position."""
    exit_rules = rules.get("exit_rules", [])
    risk_rules = rules.get("risk_rules", {})
    reasons = []

    price = indicators.get("price", 0)
    symbol = indicators.get("symbol", position.get("symbol", ""))
    entry_price = position.get("entry_price", 0)

    for rule in exit_rules:
        condition = rule.get("condition", "")
        description = rule.get("description", "")
        passed, detail = evaluate_condition(condition, indicators)
        if passed:
            reasons.append(f"EXIT: {description} ({condition})")
            return Signal(
                symbol=symbol,
                action="sell",
                strength=1.0,
                reasons=reasons,
                indicators=indicators,
                price=price,
            )

    if entry_price > 0 and price > 0:
        stop_loss_pct = risk_rules.get("stop_loss_pct", 0.02)
        take_profit_pct = risk_rules.get("take_profit_pct", 0.03)
        change_pct = (price - entry_price) / entry_price

        if change_pct <= -stop_loss_pct:
            reasons.append(
                f"STOP LOSS: price dropped {change_pct:.2%} from entry "
                f"(limit: -{stop_loss_pct:.0%})"
            )
            return Signal(
                symbol=symbol,
                action="sell",
                strength=1.0,
                reasons=reasons,
                indicators=indicators,
                price=price,
            )

        if change_pct >= take_profit_pct:
            reasons.append(
                f"TAKE PROFIT: price up {change_pct:.2%} from entry "
                f"(target: +{take_profit_pct:.0%})"
            )
            return Signal(
                symbol=symbol,
                action="sell",
                strength=1.0,
                reasons=reasons,
                indicators=indicators,
                price=price,
            )

    return Signal(
        symbol=symbol,
        action="hold",
        strength=0.0,
        reasons=["No exit conditions met"],
        indicators=indicators,
        price=price,
    )


def scan_signals(
    watchlist_data: list[dict],
    rules: Optional[dict] = None,
    open_positions: Optional[list[dict]] = None,
) -> list[Signal]:
    """Scan watchlist data and return signals for each symbol."""
    if rules is None:
        rules = load_rules()

    open_positions = open_positions or []
    position_map = {p["symbol"]: p for p in open_positions}

    signals = []
    for data in watchlist_data:
        symbol = data.get("symbol", "")
        if "error" in data:
            continue

        indicators = data.get("indicators", {})
        indicators["symbol"] = symbol
        indicators["price"] = data.get("price", indicators.get("price", 0))

        if symbol in position_map:
            signal = evaluate_exit_rules(rules, indicators, position_map[symbol])
        else:
            signal = evaluate_entry_rules(rules, indicators)

        signals.append(signal)

    return signals
