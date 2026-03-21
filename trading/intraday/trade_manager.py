"""
Trade Manager — active management of open equity positions.

Replaces the "set SL/target and forget" approach with staged exits:
  Stage 1 (0-0.5R): Initial — tight watch, time stop if no progress
  Stage 2 (0.5R):   Move SL to breakeven
  Stage 3 (1.0R):   Take partial profit (50%), trail rest
  Stage 4 (1.5R+):  Trail tighter, let it run

This is the difference between a 0.8R average win and a 1.5R average win.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
from logzero import logger


@dataclass
class ExitDecision:
    """What to do with an open position."""
    action: str      # "HOLD" | "MOVE_SL" | "PARTIAL_EXIT" | "FULL_EXIT"
    new_sl: float = 0
    exit_pct: int = 0     # % of position to exit (0, 50, 100)
    exit_price: float = 0
    reason: str = ""


# Tunable parameters
BREAKEVEN_AT_R = 0.5       # Move SL to breakeven when profit = 0.5R
PARTIAL_AT_R = 1.5         # Take 50% off at 1.5R (spec: let winners run longer before taking partial)
TIME_STOP_CANDLES = 6      # Exit if no progress in 6 candles (30 min — spec)
TIME_STOP_MIN_MOVE = 0.2   # "No progress" = less than 0.2R in time_stop period


def manage_trade(
    side: str,
    entry_price: float,
    current_sl: float,
    target: float,
    current_price: float,
    candles_since_entry: int = 0,
    max_favorable: float = 0,
    partial_taken: bool = False,
    atr: float = 0,
) -> ExitDecision:
    """
    Evaluate an open position and decide what to do.

    Args:
        side: "BUY" or "SELL"
        entry_price: Original entry price
        current_sl: Current stop loss (may have been moved)
        target: Original target
        current_price: Latest market price
        candles_since_entry: How many 5-min candles since entry
        max_favorable: Best price reached since entry (for trail calculation)
        partial_taken: Whether 50% was already taken off

    Returns:
        ExitDecision with action and parameters.
    """
    direction = 1 if side == "BUY" else -1
    risk_per_unit = abs(entry_price - current_sl) if current_sl != entry_price else abs(target - entry_price) / 2

    if risk_per_unit == 0:
        return ExitDecision("HOLD", current_sl, 0, 0, "Zero risk — cannot calculate R")

    # Current R-multiple: how many R of profit/loss
    pnl = (current_price - entry_price) * direction
    r_multiple = pnl / risk_per_unit

    # Max favorable excursion R-multiple
    if max_favorable > 0:
        mfe = (max_favorable - entry_price) * direction
        mfe_r = mfe / risk_per_unit
    else:
        mfe_r = r_multiple

    # ── Check SL hit ──
    if side == "BUY" and current_price <= current_sl:
        return ExitDecision("FULL_EXIT", current_sl, 100, current_price, f"SL hit at {current_price:.1f}")
    if side == "SELL" and current_price >= current_sl:
        return ExitDecision("FULL_EXIT", current_sl, 100, current_price, f"SL hit at {current_price:.1f}")

    # ── Check target hit ──
    if side == "BUY" and current_price >= target:
        return ExitDecision("FULL_EXIT", current_sl, 100, current_price, f"Target hit at {current_price:.1f}")
    if side == "SELL" and current_price <= target:
        return ExitDecision("FULL_EXIT", current_sl, 100, current_price, f"Target hit at {current_price:.1f}")

    # ── Stage 1: Time stop (0 to 0.5R, first 30 min) ──
    if r_multiple < BREAKEVEN_AT_R and candles_since_entry >= TIME_STOP_CANDLES:
        if r_multiple < TIME_STOP_MIN_MOVE:
            return ExitDecision(
                "FULL_EXIT", current_sl, 100, current_price,
                f"Time stop: {candles_since_entry} candles, only {r_multiple:.2f}R progress"
            )

    # ── Stage 2: Move to breakeven (at 0.5R) ──
    if r_multiple >= BREAKEVEN_AT_R:
        breakeven_sl = entry_price + direction * risk_per_unit * 0.05  # Tiny buffer above/below entry
        if side == "BUY" and breakeven_sl > current_sl:
            return ExitDecision(
                "MOVE_SL", breakeven_sl, 0, 0,
                f"Breakeven: {r_multiple:.1f}R profit, moving SL to {breakeven_sl:.1f}"
            )
        if side == "SELL" and breakeven_sl < current_sl:
            return ExitDecision(
                "MOVE_SL", breakeven_sl, 0, 0,
                f"Breakeven: {r_multiple:.1f}R profit, moving SL to {breakeven_sl:.1f}"
            )

    # ── Stage 3: Partial profit at 1.5R ──
    if r_multiple >= PARTIAL_AT_R and not partial_taken:
        return ExitDecision(
            "PARTIAL_EXIT", current_sl, 50, current_price,
            f"Taking 50% at {r_multiple:.1f}R (locking +{pnl:.1f} pts on half)"
        )

    # ── Stage 3b: Trail after partial (0.5 ATR behind best price) ──
    # Use ATR for trailing — adapts to volatility.
    # Fallback to 0.5R if ATR not provided.
    trail_unit = atr * 0.5 if atr > 0 else risk_per_unit * 0.5

    if r_multiple >= PARTIAL_AT_R and partial_taken:
        if side == "BUY":
            trail_sl = max_favorable - trail_unit
            if trail_sl > current_sl:
                return ExitDecision(
                    "MOVE_SL", trail_sl, 0, 0,
                    f"Trail: {r_multiple:.1f}R, SL → {trail_sl:.1f} (0.5 ATR behind {max_favorable:.1f})"
                )
        else:
            trail_sl = max_favorable + trail_unit
            if trail_sl < current_sl:
                return ExitDecision(
                    "MOVE_SL", trail_sl, 0, 0,
                    f"Trail: {r_multiple:.1f}R, SL → {trail_sl:.1f} (0.5 ATR behind {max_favorable:.1f})"
                )

    # ── Stage 4: Tighter trail at 2R+ (0.3 ATR behind) ──
    tight_trail = atr * 0.3 if atr > 0 else risk_per_unit * 0.3
    if r_multiple >= 2.0:
        if side == "BUY":
            trail_sl = max_favorable - tight_trail
            if trail_sl > current_sl:
                return ExitDecision(
                    "MOVE_SL", trail_sl, 0, 0,
                    f"Tight trail: {r_multiple:.1f}R, SL → {trail_sl:.1f} (0.3 ATR)"
                )
        else:
            trail_sl = max_favorable + tight_trail
            if trail_sl < current_sl:
                return ExitDecision(
                    "MOVE_SL", trail_sl, 0, 0,
                    f"Tight trail: {r_multiple:.1f}R, SL → {trail_sl:.1f} (0.3 ATR)"
                )

    return ExitDecision("HOLD", current_sl, 0, 0, f"Holding at {r_multiple:.2f}R")
