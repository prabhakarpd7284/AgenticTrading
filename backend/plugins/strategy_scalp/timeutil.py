"""Tiny time helpers for the scalp engine — IST epoch ⇄ ISO, EOD check.

Kept dependency-free (no Django, no engine imports) so both ``profile``,
``engine`` and ``replay`` can import it without circular imports.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))


def iso_from_epoch(epoch: float) -> str:
    """Epoch seconds (UTC) → IST ISO string ``YYYY-MM-DDTHH:MM:SS+05:30``."""
    return datetime.fromtimestamp(epoch, IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")


def epoch_from_iso(ts: str) -> float:
    """IST ISO string → epoch seconds. Tolerant of ``T``/space and tz suffix."""
    s = ts.strip().replace("T", " ")
    # strip a trailing +05:30 / Z if present, treat the wall-clock as IST
    for suffix in ("+05:30", "+0530", "Z"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
            break
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            naive = datetime.strptime(s, fmt)
            return naive.replace(tzinfo=IST).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unparseable timestamp: {ts!r}")


def is_eod(epoch: float, hour: int, minute: int) -> bool:
    """True if the IST wall-clock of ``epoch`` is at/after ``hour:minute``."""
    dt = datetime.fromtimestamp(epoch, IST)
    return dt.hour > hour or (dt.hour == hour and dt.minute >= minute)


# NSE/BSE session opens 09:15 IST. Intraday candles (and TradingView) anchor
# their grid to the session open, so a 10-min candle is 09:15–09:25, not the
# clock-aligned 09:10–09:20.
SESSION_OPEN_HOUR, SESSION_OPEN_MIN = 9, 15


def session_anchor(epoch: float) -> float:
    """Epoch of 09:15 IST on the same calendar day as ``epoch``."""
    d = datetime.fromtimestamp(epoch, IST).replace(
        hour=SESSION_OPEN_HOUR, minute=SESSION_OPEN_MIN, second=0, microsecond=0)
    return d.timestamp()


def decision_bucket(epoch: float, bucket_secs: int) -> int:
    """Start epoch of the ``bucket_secs`` candle containing ``epoch``,
    anchored to the 09:15 session open (so 10-min bars are 09:15, 09:25, …)."""
    anchor = session_anchor(epoch)
    if epoch < anchor:
        return int(anchor)
    return int(anchor + ((epoch - anchor) // bucket_secs) * bucket_secs)
