"""Two-day band relationships and open-location classification (spec §2).

The same 7 relationships apply to any band — CPR (BC–TC), Camarilla value
(L3–H3) or Money Zone value area (VAL–VAH). "Today" is the band computed from
yesterday's session, "yesterday" the band computed from the session before.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Relationship(StrEnum):
    HIGHER = "higher"
    OVERLAPPING_HIGHER = "overlapping_higher"
    LOWER = "lower"
    OVERLAPPING_LOWER = "overlapping_lower"
    UNCHANGED = "unchanged"
    OUTSIDE = "outside"
    INSIDE = "inside"


BIAS = {
    Relationship.HIGHER: "bullish",
    Relationship.OVERLAPPING_HIGHER: "moderately_bullish",
    Relationship.LOWER: "bearish",
    Relationship.OVERLAPPING_LOWER: "moderately_bearish",
    Relationship.UNCHANGED: "sideways_breakout",
    Relationship.OUTSIDE: "sideways",
    Relationship.INSIDE: "breakout",
}

# The book gives no number for "unchanged". Default: both edges within 0.02 %
# of price of yesterday's edges (≈ a couple of ticks on a ₹1,000 stock).
DEFAULT_UNCHANGED_TOL_PCT = 0.02


def two_day_relationship(today_bottom: float, today_top: float,
                         yday_bottom: float, yday_top: float,
                         unchanged_tol_pct: float = DEFAULT_UNCHANGED_TOL_PCT) -> Relationship:
    """Classify today's band against yesterday's.

    Precedence: unchanged (both edges within tolerance) → higher/lower (no
    overlap) → inside/outside (containment, inclusive) → overlapping higher/lower.
    """
    t_lo, t_hi = min(today_bottom, today_top), max(today_bottom, today_top)
    y_lo, y_hi = min(yday_bottom, yday_top), max(yday_bottom, yday_top)
    tol = unchanged_tol_pct / 100.0 * (y_lo + y_hi) / 2.0
    if abs(t_hi - y_hi) <= tol and abs(t_lo - y_lo) <= tol:
        return Relationship.UNCHANGED
    if t_lo > y_hi:
        return Relationship.HIGHER
    if t_hi < y_lo:
        return Relationship.LOWER
    if t_hi <= y_hi and t_lo >= y_lo:
        return Relationship.INSIDE
    if t_hi >= y_hi and t_lo <= y_lo:
        return Relationship.OUTSIDE
    if t_hi > y_hi:  # overlapping, shifted up (t_lo > y_lo follows)
        return Relationship.OVERLAPPING_HIGHER
    return Relationship.OVERLAPPING_LOWER


@dataclass(frozen=True, slots=True)
class OpenLocation:
    in_range: bool
    in_value: bool
    side: str  # "above" | "below" | "inside" — vs the prior range

    @property
    def category(self) -> str:
        """Table 4.1 bucket."""
        if self.in_range and self.in_value:
            return "in_range_in_value"
        if self.in_range:
            return "in_range_out_of_value"
        if not self.in_value:
            return "out_of_range_out_of_value"
        return "out_of_range_in_value"  # possible when value pokes outside range

    @property
    def expected_day(self) -> str:
        return {
            "in_range_in_value": "range",
            "in_range_out_of_value": "moderate",
            "out_of_range_out_of_value": "trend",
            "out_of_range_in_value": "moderate",
        }[self.category]


def classify_open(open_px: float, prior_high: float, prior_low: float,
                  value_low: float, value_high: float) -> OpenLocation:
    """Where the open sits vs the prior day's range and value band (inclusive)."""
    in_range = prior_low <= open_px <= prior_high
    lo, hi = min(value_low, value_high), max(value_low, value_high)
    in_value = lo <= open_px <= hi
    side = "inside" if in_range else ("above" if open_px > prior_high else "below")
    return OpenLocation(in_range=in_range, in_value=in_value, side=side)
