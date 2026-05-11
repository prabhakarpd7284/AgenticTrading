"""Basket strategy configuration."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BasketConfig:
    """Morning basket strategy parameters."""

    # Timing (IST)
    start_time: str = "09:45"
    end_time: str = "10:30"
    eod_close_time: str = "15:15"

    # Market mood thresholds
    ad_bullish: float = 1.2         # A/D ratio > 1.2 = bullish
    ad_bearish: float = 0.8         # A/D ratio < 0.8 = bearish
    vix_elevated: float = 20.0      # VIX > 20 = reduce size

    # Equity selection
    max_equity_legs: int = 5
    buy_phases: tuple = ("WP", "EC", "BB", "GAR", "EB3")
    sell_phases: tuple = ("WD",)
    ema_period: int = 5
    bb_period: int = 20
    bb_std: float = 2.0
    candle_interval: str = "FIVE_MINUTE"

    # Options
    index: str = "NIFTY"
    strike_step: int = 50
    option_sl_pct: float = 30.0     # 30% loss on premium = SL

    # Scale-in tranches
    t1_pct: float = 50.0            # MARKET immediately
    t2_pct: float = 30.0            # LIMIT at entry - 0.1%
    t3_pct: float = 20.0            # MARKET after 3 bars confirm
    t2_cancel_bars: int = 5         # Cancel T2 LIMIT if unfilled
    t3_confirm_bars: int = 3        # T3 fires after 3 bars hold

    # Pyramiding
    pyramid_at_r: float = 1.0       # Add when position is +1R
    pyramid_add_pct: float = 50.0   # Add 50% of original qty
    max_pyramids: int = 3

    # Risk management
    breakeven_at_r: float = 0.5     # Move SL to entry at +0.5R
    trail_start_r: float = 1.0      # Start trailing at +1R
    max_basket_loss_pct: float = 2.0  # Max loss per basket = 2% of capital
    max_risk_per_leg_pct: float = 0.5  # Max risk per leg = 0.5% of capital
