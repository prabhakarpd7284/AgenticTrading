"""Lifecycle rules — 70% profit-take + re-entry momentum gate.

The lifecycle manager runs once per cycle (every 15–30 min in production,
or every market candle in backtest mode). Given an open spread and the
current market state, it produces one of:

  - HOLD       — keep position
  - CLOSE      — profit-take, hard-stop, or time-stop hit
  - REENTER    — close current + open fresh spread at current ATM
                 (only when re-entry momentum gate passes)

The 70% profit-take and re-entry momentum gate were both surfaced by
the April 2026 backtest: profit-taking at 70% of max profit beats
hold-to-expiry, and blocking re-entry when spot retraced below the
last-close prevented the only loser of the month.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


__all__ = [
    "LifecycleAction", "LifecycleDecision",
    "evaluate_open_spread", "can_reenter",
    "PROFIT_TAKE_PCT", "HARD_STOP_MULTIPLIER", "MIN_DTE_FOR_REENTER",
    "MIN_CREDIT_WIDTH_RATIO",
]


PROFIT_TAKE_PCT: float = 0.70          # close at 70% of max profit
HARD_STOP_MULTIPLIER: float = 1.5      # close if current value > 1.5× credit (defined-risk: bounded anyway, but flags an unwind)
MIN_DTE_FOR_REENTER: int = 3           # don't open a new spread with <3 DTE
MIN_CREDIT_WIDTH_RATIO: float = 0.25   # quality threshold for re-entry


class LifecycleAction(str, Enum):
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    REENTER = "REENTER"


@dataclass(frozen=True)
class LifecycleDecision:
    action: LifecycleAction
    reason: str
    pnl_pct_of_max: float
    """Where we are between 0 (entry) and 1.0 (max profit). Negative = drawdown."""


def evaluate_open_spread(
    entry_credit: float,
    current_spread_value: float,
    max_profit: float,
    dte: int,
) -> LifecycleDecision:
    """Decide what to do with an open spread given today's market state.

    `current_spread_value` is what we'd pay to close (sell_now − buy_now);
    when it falls below entry credit we're in the money. P&L pct of max
    profit = (entry_credit − current_value) / entry_credit  (= 1.0 when
    the spread is worthless).
    """
    if entry_credit <= 0:
        return LifecycleDecision(LifecycleAction.HOLD, "entry credit ≤ 0",
                                 pnl_pct_of_max=0.0)

    pnl_pct = (entry_credit - current_spread_value) / entry_credit

    # Profit-take — the main winning rule
    if pnl_pct >= PROFIT_TAKE_PCT:
        return LifecycleDecision(
            LifecycleAction.CLOSE,
            f"profit-take {pnl_pct*100:.1f}% ≥ {PROFIT_TAKE_PCT*100:.0f}%",
            pnl_pct_of_max=pnl_pct,
        )

    # Hard-stop — current spread value blew past 1.5× credit collected
    if current_spread_value > entry_credit * HARD_STOP_MULTIPLIER:
        return LifecycleDecision(
            LifecycleAction.CLOSE,
            f"hard-stop spread {current_spread_value:.2f} > {HARD_STOP_MULTIPLIER}× credit",
            pnl_pct_of_max=pnl_pct,
        )

    # Time-stop — DTE = 0 means close before settlement risk
    if dte <= 0:
        return LifecycleDecision(
            LifecycleAction.CLOSE,
            "DTE ≤ 0 — close before expiry settlement",
            pnl_pct_of_max=pnl_pct,
        )

    return LifecycleDecision(LifecycleAction.HOLD,
                             f"theta working ({pnl_pct*100:.0f}% of max)",
                             pnl_pct_of_max=pnl_pct)


def can_reenter(
    *,
    spot_now: float,
    spot_at_last_close: Optional[float],
    dte: int,
    new_credit: float,
    new_width: int,
) -> tuple[bool, str]:
    """Re-entry momentum gate.

    Returns (allowed, reason). All four conditions must pass:
      1. Spot has not retraced below the spot at our last close
         (don't chase pullbacks).
      2. DTE ≥ MIN_DTE_FOR_REENTER — enough runway for theta.
      3. New credit ≥ MIN_CREDIT_WIDTH_RATIO × width — quality.
      4. new_width > 0 — sanity.
    """
    if new_width <= 0:
        return False, "invalid width"

    if spot_at_last_close is not None and spot_now < spot_at_last_close:
        return False, (f"momentum retraced: spot {spot_now:.0f} < "
                       f"last close {spot_at_last_close:.0f}")

    if dte < MIN_DTE_FOR_REENTER:
        return False, f"DTE {dte} < {MIN_DTE_FOR_REENTER}"

    ratio = new_credit / new_width
    if ratio < MIN_CREDIT_WIDTH_RATIO:
        return False, (f"credit/width {ratio*100:.1f}% < "
                       f"{MIN_CREDIT_WIDTH_RATIO*100:.0f}% threshold")

    return True, "all re-entry gates passed"
