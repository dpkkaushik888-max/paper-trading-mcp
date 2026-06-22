"""Strategy-selection backtest (S31).

A period-by-period harness: each period a *selector* picks one sleeve
({btc, connors, cash}) using ONLY past data, the account runs that sleeve for the
period, and the result is compared to buy-and-hold. The selector is the "agent" —
pluggable, so a deterministic regime rule today can be swapped for a bounded agent
later, with the no-lookahead contract enforced by construction.
"""
