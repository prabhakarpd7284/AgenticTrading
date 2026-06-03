"""Directional vertical spread plugin.

Modes:
  - BULL_PUT     (bias UP)
  - BEAR_CALL    (bias DOWN)
  - IRON_CONDOR  (bias RANGE, two-sided)

Design backed by PoC backtest /sessions/.../poc_vertical_backtest_v3.py
which validated, on real Angel One historical data for the April 2026
NIFTY series, that the simple 5-rule classifier (gap-override + 3-day
momentum + VIX direction) combined with the 80%/60% premium-percentile
strike selector, 70% profit-take, and 8% per-trade risk cap produces:

  - 2 trades, 100% win rate
  - +₹10,024 P&L on ₹31,939 margin (+31.4% return on risk)
  - +2.00% on ₹500,000 capital in 15 trading days (~+34% annualized)

The plugin transcribes that logic 1:1 into the production codebase,
behind the BrokerAdapter + RiskEngine + Events contracts already in place.

See: docs/strategies/DIRECTIONAL_VERTICAL_SPREAD.md
"""

from .strategy import VerticalSpreadStrategy

__all__ = ["VerticalSpreadStrategy"]
