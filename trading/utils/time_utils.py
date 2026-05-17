"""
Time utilities — single source of truth for market hours, session phases.

Every file that checks "is market open?" should use these functions
instead of inline hour/minute comparisons.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, date, time as dt_time, timedelta
from typing import Optional


# Per-request override for "what session date should panels render?".
# Set by the cockpit time-travel `?date=` query param (see views).
# When set, intraday_session_date() returns this instead of computing.
_AS_OF_OVERRIDE: ContextVar[Optional[date]] = ContextVar(
    "intraday_as_of", default=None,
)


@contextmanager
def use_session_date(d: Optional[date]):
    """Scope an intraday session-date override for the duration of a block.

    Used by cockpit views so any service that calls intraday_session_date()
    transparently returns the user-picked historical date, including for
    cache-key construction. Pass None to no-op (live behaviour)."""
    if d is None:
        yield
        return
    tok = _AS_OF_OVERRIDE.set(d)
    try:
        yield
    finally:
        _AS_OF_OVERRIDE.reset(tok)


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


def intraday_session_date(now: datetime = None) -> date:
    """The date whose intraday bars the operator should see right now.

    - Weekday after 9:16 IST → today (live or just-completed session).
    - Weekend, holiday, or pre-9:16 weekday → the previous trading day.

    Use this in cockpit panels so weekends and pre-market don't blank out
    every intraday card — operators still want to see the last session's
    VWAP / tape / ORB while they plan the next day.
    """
    override = _AS_OF_OVERRIDE.get()
    if override is not None:
        return override
    if now is None:
        now = datetime.now()
    if can_fetch_candles(now):
        return now.date()
    return last_trading_day(now)


def last_trading_day(now: datetime = None) -> date:
    """
    Return the most recent completed trading day.
    - During market hours on a weekday → previous trading day (today is incomplete)
    - After market close on a weekday → today
    - Weekend → last Friday
    """
    if now is None:
        now = datetime.now()
    d = now.date()

    # If market is still open, today's data is incomplete — use yesterday
    if now.weekday() < 5 and now.time() < MARKET_CLOSE:
        d = d - timedelta(days=1)

    # Walk back past weekends
    while d.weekday() >= 5:
        d = d - timedelta(days=1)

    return d


def get_candle_date_range(now: datetime = None) -> tuple:
    """
    Return (intraday_date, history_start, history_end) for candle fetching.
    - intraday_date: date to fetch 5-min candles for (today if market open, else last trading day)
    - history_start: 30 days back for daily/weekly/monthly charts
    - history_end: last completed trading day

    Works correctly on weekends, holidays, and after-hours.
    """
    if now is None:
        now = datetime.now()

    # Intraday: use today only if market is open and candles exist
    if now.weekday() < 5 and now.time() >= CANDLE_AVAILABLE:
        intraday_date = now.date()
    else:
        intraday_date = last_trading_day(now)

    history_end = last_trading_day(now)
    history_start = history_end - timedelta(days=90)  # ~3 months for monthly chart

    return intraday_date, history_start, history_end


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
