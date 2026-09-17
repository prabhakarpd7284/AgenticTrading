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
    # ── 2026 — verify the rest against the NSE holiday circular ──────────
    date(2026, 1, 26),   # Republic Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 12, 25),  # Christmas
    # Festival holidays (Holi, Ram Navami, Mahavir Jayanti, Buddha Purnima,
    # Bakri Id, Muharram, Ganesh Chaturthi, Dussehra, Diwali Laxmi Pujan,
    # Diwali Balipratipada, Guru Nanak Jayanti) move yearly with the lunar
    # calendar — add them here once the NSE circular is published.
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
