"""
Time utilities — single source of truth for market hours, session phases.

Every file that checks "is market open?" should use these functions
instead of inline hour/minute comparisons.
"""
from datetime import datetime, time as dt_time


# ── Market schedule (IST) ──
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)
PREMARKET_START = dt_time(7, 0)
CANDLE_AVAILABLE = dt_time(9, 16)   # first 5-min candle closes at 9:20, safe to fetch at 9:16
CLOSING_START = dt_time(15, 0)      # stop new entries, start closing
SQUARE_OFF_TIME = dt_time(15, 15)   # force-close all intraday
STRADDLE_CLOSE = dt_time(15, 0)     # force-close 0 DTE straddles


def is_market_open(now: datetime = None) -> bool:
    """Check if Indian market is currently open (9:15 AM - 3:30 PM IST, weekdays)."""
    if now is None:
        now = datetime.now()
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def is_pre_market(now: datetime = None) -> bool:
    """Before market open but after premarket scan window."""
    if now is None:
        now = datetime.now()
    return now.weekday() < 5 and PREMARKET_START <= now.time() < MARKET_OPEN


def can_fetch_candles(now: datetime = None) -> bool:
    """Whether intraday candle data is available from Angel One."""
    if now is None:
        now = datetime.now()
    return now.weekday() < 5 and now.time() >= CANDLE_AVAILABLE


def is_closing_time(now: datetime = None) -> bool:
    """After 3:00 PM — stop new entries, prepare to close."""
    if now is None:
        now = datetime.now()
    return now.time() >= CLOSING_START


def get_session_phase(now: datetime = None) -> str:
    """
    Classify current trading session phase.

    Returns: WEEKEND | PRE_MARKET | OPENING | REGULAR | CLOSING | POST_MARKET
    """
    if now is None:
        now = datetime.now()
    if now.weekday() >= 5:
        return "WEEKEND"
    t = now.time()
    if t < PREMARKET_START:
        return "PRE_MARKET"
    if t < MARKET_OPEN:
        return "PRE_MARKET"
    if t < dt_time(9, 30):
        return "OPENING"
    if t < CLOSING_START:
        return "REGULAR"
    if t <= MARKET_CLOSE:
        return "CLOSING"
    return "POST_MARKET"


def cap_end_time(date_str: str, now: datetime = None) -> str:
    """
    Cap candle end time to current time if fetching today's data.
    Prevents Angel One "future datetime" error.

    Returns: "{date_str} HH:MM" string.
    """
    if now is None:
        now = datetime.now()
    from datetime import date
    if date_str == date.today().isoformat():
        end_t = min(now.time(), MARKET_CLOSE)
        return f"{date_str} {end_t.strftime('%H:%M')}"
    return f"{date_str} 15:30"
