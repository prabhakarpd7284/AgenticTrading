"""
Straddle Lifecycle — simple rules that actually work.

Replaces the complex multi-rule validator with 5 clear rules:
  1. Hard stop: combined > 1.3x sold → CLOSE_BOTH
  2. Expiry day 3 PM → CLOSE_BOTH
  3. Max 2 shifts per day
  4. Shift trigger: NIFTY moves >200 pts from straddle center → SHIFT_TO_ATM
  5. Everything else → HOLD

A SHIFT is an atomic action: close both legs + sell new ATM straddle.
No individual leg management. No rolling. No micro-adjustments.

From SPEC V2:
  "Instead of micro-managing individual legs (which caused the 11-roll disaster),
   we treat the straddle as a unit. Either the whole thing is fine (HOLD)
   or the whole thing needs to move (SHIFT)."
"""
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional
from logzero import logger


@dataclass
class LifecycleDecision:
    """What the lifecycle manager decided."""
    action: str          # "HOLD" | "CLOSE_BOTH" | "SHIFT_TO_ATM"
    reason: str
    new_strike: int = 0  # Only for SHIFT_TO_ATM
    urgency: str = "MONITOR"


# ── Parameters from centralized config ──
from trading.config import config as _cfg

HARD_STOP_MULTIPLIER = _cfg.straddle.hard_stop_multiplier
SHIFT_THRESHOLD = _cfg.straddle.shift_threshold
MAX_SHIFTS_PER_DAY = _cfg.straddle.max_shifts_per_day


def decide(
    ce_sell: float,
    pe_sell: float,
    ce_ltp: float,
    pe_ltp: float,
    ce_strike: int,
    pe_strike: int,
    nifty_spot: float,
    is_expiry_day: bool = False,
    shifts_today: int = 0,
) -> LifecycleDecision:
    """
    One function, one decision, no conflicts.

    Args:
        ce_sell: CE premium when sold
        pe_sell: PE premium when sold
        ce_ltp: Current CE LTP
        pe_ltp: Current PE LTP
        ce_strike: Current CE strike price
        pe_strike: Current PE strike price
        nifty_spot: Current NIFTY spot
        is_expiry_day: True if today is the expiry day
        shifts_today: How many times we've already shifted today

    Returns:
        LifecycleDecision with action + reason
    """
    combined_sold = ce_sell + pe_sell
    combined_current = ce_ltp + pe_ltp
    severity = combined_current / combined_sold if combined_sold > 0 else 2.0
    center = (ce_strike + pe_strike) / 2
    drift = abs(nifty_spot - center)

    # ── Rule 1: Hard stop (1.3x — tighter than old 1.5x) ──
    if severity >= HARD_STOP_MULTIPLIER:
        return LifecycleDecision(
            "CLOSE_BOTH",
            f"Hard stop: combined {combined_current:.0f} > {HARD_STOP_MULTIPLIER}x sold {combined_sold:.0f} "
            f"(severity {severity:.2f}x). Closing to limit loss.",
            urgency="IMMEDIATE",
        )

    # ── Rule 2: Expiry day after 3 PM ──
    if is_expiry_day:
        now = datetime.now()
        if now.hour >= 15:
            return LifecycleDecision(
                "CLOSE_BOTH",
                "Expiry day 3 PM — must exit before 3:15 PM.",
                urgency="IMMEDIATE",
            )

    # ── Rule 3: Shift when NIFTY drifts >200 pts (max 2/day) ──
    if drift > SHIFT_THRESHOLD:
        if shifts_today >= MAX_SHIFTS_PER_DAY:
            return LifecycleDecision(
                "HOLD",
                f"NIFTY drifted {drift:.0f} pts but max shifts reached ({shifts_today}/{MAX_SHIFTS_PER_DAY}). "
                f"Holding — next shift would be tomorrow.",
            )

        # Calculate new ATM strike
        new_strike = round(nifty_spot / 50) * 50

        return LifecycleDecision(
            "SHIFT_TO_ATM",
            f"NIFTY drifted {drift:.0f} pts from center {center:.0f}. "
            f"Shifting straddle from {int(center)} → {new_strike}. "
            f"Shift {shifts_today + 1}/{MAX_SHIFTS_PER_DAY}.",
            new_strike=new_strike,
            urgency="IMMEDIATE",
        )

    # ── Rule 4: 0 DTE theta protection ──
    if is_expiry_day and severity < 1.0:
        now = datetime.now()
        if now.hour < 14 or (now.hour == 14 and now.minute < 30):
            decay_pct = (1 - severity) * 100
            return LifecycleDecision(
                "HOLD",
                f"0 DTE: premium decayed {decay_pct:.0f}%. Let theta work until 2:30 PM.",
            )

    # ── Rule 5: Everything else → HOLD ──
    return LifecycleDecision(
        "HOLD",
        f"Position OK. Severity {severity:.2f}x, drift {drift:.0f} pts. "
        f"Theta is working.",
    )


def count_shifts_today(management_log: list) -> int:
    """Count SHIFT_TO_ATM actions in today's management log."""
    today_str = date.today().isoformat()
    count = 0
    for entry in (management_log or []):
        action = entry.get("action", "")
        # Count both SHIFT_TO_ATM and REENTER as shifts
        if action in ("SHIFT_TO_ATM", "REENTER"):
            count += 1
        # Also count ROLLs as shifts (legacy)
        if "ROLL" in action:
            count += 1
    return count
