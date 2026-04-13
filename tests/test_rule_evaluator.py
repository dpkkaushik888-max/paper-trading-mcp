"""Tests for the rule evaluator — condition parsing and signal generation."""

import pytest

from trading_engine.rule_evaluator import (
    evaluate_condition,
    evaluate_entry_rules,
    evaluate_exit_rules,
    load_rules,
    parse_condition,
)


class TestParseCondition:
    def test_numeric_less_than(self):
        result = parse_condition("rsi_14 < 30")
        assert result is not None
        name, op, val, mode = result
        assert name == "rsi_14"
        assert val == 30.0
        assert mode == "num"

    def test_numeric_greater_equal(self):
        result = parse_condition("price >= 100.5")
        assert result is not None
        assert result[2] == 100.5

    def test_indicator_comparison(self):
        result = parse_condition("price > bb_lower")
        assert result is not None
        name, op, right, mode = result
        assert name == "price"
        assert right == "bb_lower"
        assert mode == "ind"

    def test_negative_number(self):
        result = parse_condition("macd_histogram > -0.5")
        assert result is not None
        assert result[2] == -0.5

    def test_invalid_condition(self):
        result = parse_condition("this is not valid")
        assert result is None

    def test_empty_string(self):
        result = parse_condition("")
        assert result is None


class TestEvaluateCondition:
    def test_numeric_true(self):
        passed, detail = evaluate_condition("rsi_14 < 35", {"rsi_14": 25.0})
        assert passed is True
        assert "✓" in detail

    def test_numeric_false(self):
        passed, detail = evaluate_condition("rsi_14 < 35", {"rsi_14": 60.0})
        assert passed is False
        assert "✗" in detail

    def test_indicator_vs_indicator_true(self):
        indicators = {"price": 500.0, "ema_50": 480.0}
        passed, _ = evaluate_condition("price > ema_50", indicators)
        assert passed is True

    def test_indicator_vs_indicator_false(self):
        indicators = {"price": 470.0, "ema_50": 480.0}
        passed, _ = evaluate_condition("price > ema_50", indicators)
        assert passed is False

    def test_missing_left_indicator(self):
        passed, detail = evaluate_condition("rsi_14 < 30", {})
        assert passed is False
        assert "not available" in detail

    def test_missing_right_indicator(self):
        passed, detail = evaluate_condition("price > bb_lower", {"price": 500.0})
        assert passed is False
        assert "not available" in detail

    def test_unparseable_condition(self):
        passed, detail = evaluate_condition("garbage in garbage out", {})
        assert passed is False
        assert "Cannot parse" in detail


class TestEvaluateEntryRules:
    @pytest.fixture
    def rules(self):
        return load_rules()

    def test_strong_buy_signal(self, rules):
        indicators = {
            "symbol": "SPY",
            "price": 500.0,
            "rsi_14": 25.0,
            "ema_20": 490.0,
            "macd_histogram": 2.0,
            "bb_lower": 480.0,
        }
        signal = evaluate_entry_rules(rules, indicators)
        assert signal.action == "buy"
        assert signal.strength >= 0.6

    def test_hold_signal_insufficient_score(self, rules):
        indicators = {
            "symbol": "SPY",
            "price": 500.0,
            "rsi_14": 60.0,
            "ema_20": 510.0,
            "macd_histogram": -1.0,
            "bb_lower": 480.0,
        }
        signal = evaluate_entry_rules(rules, indicators)
        assert signal.action == "hold"
        assert signal.strength < 0.6

    def test_partial_match(self, rules):
        indicators = {
            "symbol": "QQQ",
            "price": 400.0,
            "rsi_14": 60.0,
            "ema_20": 390.0,
            "macd_histogram": 1.5,
            "bb_lower": 380.0,
        }
        signal = evaluate_entry_rules(rules, indicators)
        assert signal.strength == pytest.approx(0.55, abs=0.01)
        assert signal.action == "hold"


class TestEvaluateExitRules:
    @pytest.fixture
    def rules(self):
        return load_rules()

    def test_rsi_overbought_exit(self, rules):
        indicators = {
            "symbol": "SPY",
            "price": 520.0,
            "rsi_14": 75.0,
            "ema_200": 500.0,
            "macd_histogram": 1.0,
        }
        position = {"symbol": "SPY", "entry_price": 500.0}
        signal = evaluate_exit_rules(rules, indicators, position)
        assert signal.action == "sell"

    def test_stop_loss_exit(self, rules):
        indicators = {
            "symbol": "SPY",
            "price": 489.0,
            "rsi_14": 40.0,
            "ema_200": 480.0,
            "macd_histogram": 1.0,
        }
        position = {"symbol": "SPY", "entry_price": 500.0}
        signal = evaluate_exit_rules(rules, indicators, position)
        assert signal.action == "sell"
        assert "STOP LOSS" in signal.reasons[0]

    def test_take_profit_exit(self, rules):
        indicators = {
            "symbol": "SPY",
            "price": 526.0,
            "rsi_14": 65.0,
            "ema_200": 490.0,
            "macd_histogram": 1.0,
        }
        position = {"symbol": "SPY", "entry_price": 500.0}
        signal = evaluate_exit_rules(rules, indicators, position)
        assert signal.action == "sell"
        assert "TAKE PROFIT" in signal.reasons[0]

    def test_hold_no_exit(self, rules):
        indicators = {
            "symbol": "SPY",
            "price": 505.0,
            "rsi_14": 55.0,
            "ema_200": 500.0,
            "macd_histogram": 1.0,
        }
        position = {"symbol": "SPY", "entry_price": 500.0}
        signal = evaluate_exit_rules(rules, indicators, position)
        assert signal.action == "hold"
