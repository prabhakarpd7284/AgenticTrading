"""NSE trading-day calendar.

`trading.utils.time_utils` already knows market *hours* but only skips
weekends — it has no holiday awareness. The daily pipeline (premarket
scan, screener session, EOD enrichment) needs a real "is today a trading
day" check so scheduled tasks no-op on NSE holidays instead of hitting
the broker for empty data.

Maintenance: NSE publishes the holiday list yearly. Update ``NSE_HOLIDAYS``
each January from the official circular. A *missing* holiday degrades
gracefully — the task runs, the broker returns empty data, no harm. A
*wrong* entry is worse (it skips a real trading day), so keep the set
conservative: only add dates you have verified.
"""
from __future__ import annotations

from datetime import date, datetime


# Verified NSE full-day trading holidays. Weekends are handled separately
# by the weekday check below, so weekend-falling holidays are omitted.
NSE_HOLIDAYS: set[date] = {
    # ── 2026 — NSE capital-market list (nseindia.com/api/holiday-master
    # ?type=trading, fetched 2026-10-21). Weekend-falling entries omitted:
    # Mahashivratri 15 Feb, Id-Ul-Fitr 21 Mar, Independence Day 15 Aug,
    # Diwali Laxmi Pujan 8 Nov (Sunday; muhurat session only).
    date(2026, 1, 15),   # Municipal Corporation Election — Maharashtra
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Shri Ram Navami
    date(2026, 3, 31),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Id
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali-Balipratipada
    date(2026, 11, 24),  # Prakash Gurpurb Sri Guru Nanak Dev
    date(2026, 12, 25),  # Christmas
}


def is_trading_day(d: date | datetime | None = None) -> bool:
    """True when ``d`` (default: today) is an NSE trading day.

    A trading day is a weekday that is not in ``NSE_HOLIDAYS``.
    """
    if d is None:
        d = date.today()
    elif isinstance(d, datetime):
        d = d.date()
    if d.weekday() >= 5:          # Saturday / Sunday
        return False
    return d not in NSE_HOLIDAYS


def next_trading_day(d: date | datetime | None = None) -> date:
    """Return the next trading day strictly after ``d`` (default: today)."""
    from datetime import timedelta

    if d is None:
        d = date.today()
    elif isinstance(d, datetime):
        d = d.date()
    nxt = d + timedelta(days=1)
    while not is_trading_day(nxt):
        nxt = nxt + timedelta(days=1)
    return nxt
