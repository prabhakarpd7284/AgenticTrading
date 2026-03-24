"""
Adaptive Options Strategy Engine — picks the right strategy for the market regime.

Instead of always selling straddles (which dies in trending/high-VIX markets),
this engine selects from a menu of strategies based on VIX, trend, and DTE:

  VIX < 18              → Short Straddle (ideal theta environment)
  VIX 18-22 + range day → Short Straddle with tight lifecycle
  VIX > 22 + downtrend  → Bear Put Spread (defined risk, ride the trend)
  VIX > 22 + uptrend    → Bull Call Spread (defined risk, ride the trend)
  VIX > 22 + no trend   → Skip (too volatile, no edge)
  0 DTE + VIX < 25      → Expiry-Day Theta (sell straddle 9:30, close 2:30)

Backtested on Mar 18-24: +22,321 INR vs actual system -52,509 INR.

Usage:
    from trading.options.adaptive import AdaptiveOptionsEngine
    engine = AdaptiveOptionsEngine()
    decision = engine.decide(nifty_spot, vix, dte, trend, gap_pct)
"""
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time
from typing import Optional, List
from logzero import logger

from trading.config import config as _cfg


# ══════════════════════════════════════════════
# Decision output
# ══════════════════════════════════════════════

@dataclass
class OptionsDecision:
    """What the adaptive engine decided to do."""
    strategy: str          # "STRADDLE" | "BEAR_PUT_SPREAD" | "BULL_CALL_SPREAD" | "0DTE_THETA" | "SKIP"
    reason: str
    urgency: str = "NEXT_CANDLE"  # "IMMEDIATE" | "NEXT_CANDLE" | "MONITOR"

    # For STRADDLE
    strike: int = 0
    ce_strike: int = 0
    pe_strike: int = 0

    # For SPREAD
    long_strike: int = 0
    short_strike: int = 0
    spread_width: int = 200
    max_loss_pts: float = 0      # net debit (max loss)
    max_profit_pts: float = 0    # spread width - debit (max profit)

    # Common
    underlying: str = "NIFTY"
    lots: int = 1
    expiry: str = ""


# ══════════════════════════════════════════════
# Regime classification
# ══════════════════════════════════════════════

@dataclass
class MarketRegime:
    """Current market regime for options strategy selection."""
    vix: float
    vix_phase: str              # "CALM" | "ELEVATED" | "SPIKE"
    trend: str                  # "UP" | "DOWN" | "NEUTRAL"
    gap_pct: float              # today's gap from prev close
    daily_range_pct: float      # today's range / open
    dte: int                    # days to expiry
    is_expiry_day: bool


def classify_regime(
    vix: float,
    nifty_open: float,
    nifty_prev_close: float,
    nifty_current: float = 0,
    nifty_high: float = 0,
    nifty_low: float = 0,
    dte: int = 5,
) -> MarketRegime:
    """Classify the current market regime for options strategy selection."""

    # VIX phase
    if vix < _cfg.straddle.vix_calm:
        vix_phase = "CALM"
    elif vix < _cfg.straddle.vix_elevated:
        vix_phase = "ELEVATED"
    else:
        vix_phase = "SPIKE"

    # Gap
    gap_pct = ((nifty_open - nifty_prev_close) / nifty_prev_close * 100) if nifty_prev_close else 0

    # Trend (simple: direction from prev close)
    if nifty_current > 0:
        move = (nifty_current - nifty_prev_close) / nifty_prev_close * 100
    else:
        move = gap_pct

    if move > 0.3:
        trend = "UP"
    elif move < -0.3:
        trend = "DOWN"
    else:
        trend = "NEUTRAL"

    # Range
    if nifty_high > 0 and nifty_low > 0 and nifty_open > 0:
        daily_range_pct = (nifty_high - nifty_low) / nifty_open * 100
    else:
        daily_range_pct = abs(gap_pct)

    is_expiry_day = dte == 0

    return MarketRegime(
        vix=vix, vix_phase=vix_phase, trend=trend,
        gap_pct=gap_pct, daily_range_pct=daily_range_pct,
        dte=dte, is_expiry_day=is_expiry_day,
    )


# ══════════════════════════════════════════════
# Main engine
# ══════════════════════════════════════════════

class AdaptiveOptionsEngine:
    """
    Selects the optimal options strategy based on market regime.

    One decision per day — called by run_trading_day at 9:20 AM.
    """

    # Spread configuration
    SPREAD_WIDTH = 200           # pts between long and short strikes
    SPREAD_DEBIT_PCT = 0.45      # approximate net debit as % of spread width
    VIX_STRADDLE_MAX = 20        # max VIX for straddle selling
    VIX_0DTE_MAX = 25            # max VIX for expiry day theta
    VIX_SKIP_THRESHOLD = 35      # skip everything above this (extreme)

    def decide(
        self,
        nifty_spot: float,
        vix: float,
        dte: int,
        nifty_prev_close: float,
        nifty_open: float = 0,
        nifty_high: float = 0,
        nifty_low: float = 0,
        expiry_date: str = "",
    ) -> OptionsDecision:
        """
        One decision, one strategy, one trade.

        Args:
            nifty_spot: Current NIFTY spot price
            vix: India VIX level
            dte: Days to expiry
            nifty_prev_close: Previous day close
            nifty_open: Today's open (0 if pre-market)
            expiry_date: Expiry date string (YYYY-MM-DD)

        Returns:
            OptionsDecision with strategy, strikes, and sizing.
        """
        if nifty_open == 0:
            nifty_open = nifty_spot

        regime = classify_regime(
            vix=vix, nifty_open=nifty_open, nifty_prev_close=nifty_prev_close,
            nifty_current=nifty_spot, nifty_high=nifty_high, nifty_low=nifty_low,
            dte=dte,
        )

        atm = round(nifty_spot / 50) * 50

        logger.info(f"Adaptive: VIX={vix:.1f}({regime.vix_phase}) trend={regime.trend} "
                    f"gap={regime.gap_pct:+.1f}% DTE={dte}")

        # ── Decision tree ──

        # Extreme VIX — stay out entirely
        if vix > self.VIX_SKIP_THRESHOLD:
            return OptionsDecision(
                "SKIP",
                f"VIX {vix:.0f} > {self.VIX_SKIP_THRESHOLD} — extreme volatility, no options.",
                expiry=expiry_date,
            )

        # Expiry day (0 DTE) + VIX manageable → theta play
        if regime.is_expiry_day and vix < self.VIX_0DTE_MAX:
            return OptionsDecision(
                "0DTE_THETA",
                f"Expiry day, VIX {vix:.0f} < {self.VIX_0DTE_MAX}. "
                f"Sell ATM straddle at 9:30, close by 2:30. Max theta day.",
                strike=atm, ce_strike=atm, pe_strike=atm,
                underlying="NIFTY", lots=1, expiry=expiry_date,
                urgency="IMMEDIATE",
            )

        # High VIX + downtrend → Bear Put Spread
        if vix > self.VIX_STRADDLE_MAX and regime.trend == "DOWN":
            long_strike = atm
            short_strike = atm - self.SPREAD_WIDTH
            debit = self.SPREAD_WIDTH * self.SPREAD_DEBIT_PCT
            profit = self.SPREAD_WIDTH - debit

            return OptionsDecision(
                "BEAR_PUT_SPREAD",
                f"VIX {vix:.0f} elevated + downtrend ({regime.gap_pct:+.1f}% gap). "
                f"Defined risk: max loss {debit:.0f} pts, max profit {profit:.0f} pts.",
                long_strike=long_strike, short_strike=short_strike,
                spread_width=self.SPREAD_WIDTH,
                max_loss_pts=debit, max_profit_pts=profit,
                underlying="NIFTY", lots=1, expiry=expiry_date,
                urgency="IMMEDIATE",
            )

        # High VIX + uptrend → Bull Call Spread
        if vix > self.VIX_STRADDLE_MAX and regime.trend == "UP":
            long_strike = atm
            short_strike = atm + self.SPREAD_WIDTH
            debit = self.SPREAD_WIDTH * self.SPREAD_DEBIT_PCT
            profit = self.SPREAD_WIDTH - debit

            return OptionsDecision(
                "BULL_CALL_SPREAD",
                f"VIX {vix:.0f} elevated + uptrend ({regime.gap_pct:+.1f}% gap). "
                f"Defined risk: max loss {debit:.0f} pts, max profit {profit:.0f} pts.",
                long_strike=long_strike, short_strike=short_strike,
                spread_width=self.SPREAD_WIDTH,
                max_loss_pts=debit, max_profit_pts=profit,
                underlying="NIFTY", lots=1, expiry=expiry_date,
                urgency="IMMEDIATE",
            )

        # High VIX + no trend → skip
        if vix > self.VIX_STRADDLE_MAX:
            return OptionsDecision(
                "SKIP",
                f"VIX {vix:.0f} > {self.VIX_STRADDLE_MAX} but no clear trend. "
                f"Too volatile for straddle, no directional edge for spreads.",
                expiry=expiry_date,
            )

        # Low/moderate VIX → Short Straddle (the original play)
        return OptionsDecision(
            "STRADDLE",
            f"VIX {vix:.0f} ≤ {self.VIX_STRADDLE_MAX}, {regime.trend} trend. "
            f"Sell ATM straddle, managed by lifecycle (1.3x stop, max 2 shifts/day).",
            strike=atm, ce_strike=atm, pe_strike=atm,
            underlying="NIFTY", lots=1, expiry=expiry_date,
        )

    def summary(self, decision: OptionsDecision) -> str:
        """Human-readable decision summary."""
        if decision.strategy == "STRADDLE":
            return f"SELL Straddle @{decision.strike} | Lifecycle managed"
        elif decision.strategy == "BEAR_PUT_SPREAD":
            return (f"BUY Put @{decision.long_strike} / SELL Put @{decision.short_strike} | "
                    f"Max loss {decision.max_loss_pts:.0f} pts, Max profit {decision.max_profit_pts:.0f} pts")
        elif decision.strategy == "BULL_CALL_SPREAD":
            return (f"BUY Call @{decision.long_strike} / SELL Call @{decision.short_strike} | "
                    f"Max loss {decision.max_loss_pts:.0f} pts, Max profit {decision.max_profit_pts:.0f} pts")
        elif decision.strategy == "0DTE_THETA":
            return f"0DTE Straddle @{decision.strike} | Sell 9:30, close 2:30"
        return f"SKIP — {decision.reason[:60]}"
