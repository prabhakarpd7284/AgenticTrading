"""
Market Regime Detector — classifies the trading day to adapt strategy.

A great trader doesn't use the same settings every day.
Trending days reward breakouts. Range days punish them.
This module detects the regime and adjusts signal acceptance accordingly.

Based on backtest analysis (Mar 13-18, 2026):
  TRENDING + PDL_BREAK = +1,240 (40% WR) → TRADE
  WITH_GAP + PDL_BREAK = +1,772 (67% WR) → TRADE
  TRENDING + PDH_BREAK = +495 (100% WR) → TRADE
  RANGE + PDH_BREAK = -2,039 (11% WR) → SKIP
  WITH_GAP + PDH_BREAK = -1,168 (0% WR) → SKIP
  RANGE + PDL_BREAK = -566 (0% WR) → SKIP
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketRegime:
    """Today's market regime — determines which setups to trade."""
    classification: str       # TRENDING, RANGE, GAP_UP, GAP_DOWN
    nifty_gap_pct: float     # opening gap from prev close
    nifty_range_pct: float   # intraday range so far (or prev day's)
    nifty_trend_pct: float   # open-to-current change
    vix: float               # current VIX
    confidence: float        # how confident we are in the classification


def classify_regime(
    nifty_open: float,
    nifty_high: float,
    nifty_low: float,
    nifty_current: float,
    nifty_prev_close: float,
    vix: float = 0,
) -> MarketRegime:
    """
    Classify the current market regime from NIFTY intraday data.
    Call this at market open and periodically during the day.
    """
    if nifty_prev_close <= 0:
        return MarketRegime("UNKNOWN", 0, 0, 0, vix, 0)

    gap_pct = (nifty_open - nifty_prev_close) / nifty_prev_close * 100
    range_pct = (nifty_high - nifty_low) / nifty_prev_close * 100
    trend_pct = (nifty_current - nifty_open) / nifty_open * 100

    # Classification hierarchy
    if abs(gap_pct) > 0.5:
        classification = "GAP_UP" if gap_pct > 0 else "GAP_DOWN"
        confidence = min(0.9, abs(gap_pct) / 2)
    elif range_pct > 1.5:
        classification = "TRENDING"
        confidence = min(0.9, range_pct / 3)
    elif range_pct < 1.3:
        # Sub-1.3% range = range-bound. PDH/PDL breakouts fail here.
        classification = "RANGE"
        confidence = min(0.9, (1.3 - range_pct) / 1.3)
    else:
        classification = "NORMAL"
        confidence = 0.5

    return MarketRegime(
        classification=classification,
        nifty_gap_pct=round(gap_pct, 2),
        nifty_range_pct=round(range_pct, 2),
        nifty_trend_pct=round(trend_pct, 2),
        vix=round(vix, 2),
        confidence=round(confidence, 2),
    )


# ──────────────────────────────────────────────
# Signal acceptance rules (from backtest data)
# ──────────────────────────────────────────────

# Each rule: (regime, setup_type) → adjustment factor
# >1.0 = boost confidence, <1.0 = reduce, 0 = skip entirely
REGIME_RULES = {
    # TRENDING days — breakdowns work, breakouts in trend work
    ("TRENDING", "PDL_BREAK"):  1.15,   # +15% conf — this is our best edge
    ("TRENDING", "PDH_BREAK"):  1.10,   # +10% — works when trend is up
    ("TRENDING", "ORB_LONG"):   1.05,
    ("TRENDING", "ORB_SHORT"):  1.05,
    ("TRENDING", "VWAP_RECLAIM"): 1.0,
    ("TRENDING", "VWAP_REJECT"): 0.8,
    ("TRENDING", "GAP_AND_GO"): 0.9,

    # RANGE days — everything fails, be very selective
    ("RANGE", "PDH_BREAK"):     0.5,    # Kill it — 11% WR, -2039
    ("RANGE", "PDL_BREAK"):     0.6,    # Kill it — 0% WR, -566
    ("RANGE", "ORB_LONG"):      0.7,
    ("RANGE", "ORB_SHORT"):     0.7,
    ("RANGE", "VWAP_RECLAIM"):  0.9,    # VWAP mean-reversion works in range
    ("RANGE", "VWAP_REJECT"):   0.9,
    ("RANGE", "GAP_FILL"):      1.0,    # Gap fills work in range

    # GAP_UP — shorts against gap fail, longs into gap lose steam
    ("GAP_UP", "PDH_BREAK"):    0.5,    # 0% WR — gap ate the breakout
    ("GAP_UP", "PDL_BREAK"):    1.2,    # Counter-gap shorts work when gap fails
    ("GAP_UP", "GAP_FILL"):     1.1,    # Gap fill is the play
    ("GAP_UP", "ORB_SHORT"):    0.7,    # Risky to short into gap-up
    ("GAP_UP", "ORB_LONG"):     0.6,    # Gap already priced the long

    # GAP_DOWN — mirror of GAP_UP
    ("GAP_DOWN", "PDL_BREAK"):  0.5,    # Gap ate the breakdown
    ("GAP_DOWN", "PDH_BREAK"):  1.2,    # Counter-gap longs work when gap fails
    ("GAP_DOWN", "GAP_FILL"):   1.1,
    ("GAP_DOWN", "ORB_LONG"):   0.7,
    ("GAP_DOWN", "ORB_SHORT"):  0.6,

    # NORMAL days — default behavior, no adjustment
    ("NORMAL", "PDH_BREAK"):    1.0,
    ("NORMAL", "PDL_BREAK"):    1.0,
}


def adjust_confidence(
    setup_type: str,
    base_confidence: float,
    regime: MarketRegime,
) -> float:
    """
    Adjust signal confidence based on market regime.
    Returns adjusted confidence (0.0-1.0).
    """
    key = (regime.classification, setup_type)
    factor = REGIME_RULES.get(key, 1.0)

    # Scale the adjustment by regime confidence
    # If we're not sure about the regime, don't adjust as much
    effective_factor = 1.0 + (factor - 1.0) * regime.confidence

    adjusted = base_confidence * effective_factor
    return round(max(0.0, min(1.0, adjusted)), 3)
