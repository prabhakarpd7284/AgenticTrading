"""Bar type + IST session helpers shared by the Pivot Boss modules.

``t`` is epoch seconds (UTC) of the bar *open*. NSE session clock helpers work
on IST minute-of-day so no tz database is needed in hot loops.
"""
from __future__ import annotations

from dataclasses import dataclass

IST_OFFSET_SECS = 19_800  # +05:30
SESSION_OPEN_MIN = 9 * 60 + 15  # 09:15 IST


@dataclass(slots=True)
class Bar:
    t: int
    o: float
    h: float
    l: float  # noqa: E741 — OHLC convention
    c: float
    v: float = 0.0

    @property
    def range(self) -> float:
        return self.h - self.l

    @property
    def body(self) -> float:
        return abs(self.c - self.o)


def ist_minute(t: int) -> int:
    """IST minute-of-day (0..1439) for an epoch-seconds timestamp."""
    return ((t + IST_OFFSET_SECS) % 86_400) // 60


def ist_day(t: int) -> int:
    """IST day number (days since epoch, in IST) — cheap grouping key."""
    return (t + IST_OFFSET_SECS) // 86_400


def hhmm(minute_of_day: int) -> str:
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"
