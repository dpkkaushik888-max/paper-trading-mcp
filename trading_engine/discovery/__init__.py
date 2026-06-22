"""Strategy discovery (S25, "Agent 1").

The agent proposes candidate strategies from the bounded primitive grammar here;
they compile into ordinary BaseStrategy instances and must earn their way past the
locked-holdout gates before promotion. No LLM in any generated strategy's hot path.
"""
