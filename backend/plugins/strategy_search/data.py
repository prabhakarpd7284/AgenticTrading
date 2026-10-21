"""DB-backed candle source for research — one symbol at a time, OOS-guarded.

* ``market_data.Candle`` rows: ``1m`` Angel bars (NIFTY 50 + indices) and
  ``1d`` NSE bhavcopy (all stocks; ``t`` = IST midnight). ``t`` is aware UTC.
* Bars are ``plugins.strategy_pivot_boss.bars.Bar`` (``t`` = epoch seconds of
  the bar OPEN, UTC). IST helpers: ``ist_minute``, ``ist_day``.
* ``CandleSource(max_date=…)`` refuses any query whose end is past
  ``max_date``; ``max_date`` past IS_END needs ``allow_oos=True``.
* ``load_symbol`` returns one symbol's sessions (plus a few warm-up sessions
  before ``start``). Callers process a symbol, drop it, move on — never hold
  all symbols' 1-minute rows at once (~47k rows per symbol for 6 months).
* Intraday bars are cut at 15:14 IST (no bars 15:15–15:29 except 15:28 since
  2026-08-03 → everything is flat by the 15:14 bar).

No look-ahead: a ``DaySession`` exposes the PRIOR session's bhavcopy bar
(``prev_daily``) and the daily history up to the previous session; the day's
own bhavcopy bar is only known after the close and is deliberately not exposed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, UTC
from zoneinfo import ZoneInfo

from plugins.strategy_pivot_boss.bars import IST_OFFSET_SECS, Bar, ist_day, ist_minute
from plugins.strategy_pivot_boss.triggers import resample as _pb_resample

from .periods import IS_END, check_date

IST = ZoneInfo("Asia/Kolkata")
EPOCH = date(1970, 1, 1)
SESSION_OPEN_MIN = 9 * 60 + 15
LAST_MIN = 15 * 60 + 14
REFERENCE_SYMBOL = "RELIANCE"  # its 1d bhavcopy rows define the session calendar


def day_of(t: int) -> date:
    return EPOCH + timedelta(days=ist_day(t))


def ist_dt(t: int) -> datetime:
    """Naive IST datetime for epoch seconds (what the live screener uses)."""
    return datetime(1970, 1, 1) + timedelta(seconds=t + IST_OFFSET_SECS)


def epoch_of_ist(dt: datetime) -> int:
    """Inverse of ``ist_dt`` for a naive IST datetime."""
    return int((dt - datetime(1970, 1, 1)).total_seconds()) - IST_OFFSET_SECS


def epoch_at(d: date, hh: int, mm: int) -> int:
    """Epoch seconds of ``d hh:mm`` IST."""
    return epoch_of_ist(datetime(d.year, d.month, d.day, hh, mm))


def resample(bars_1m: list[Bar], minutes: int) -> list[Bar]:
    """N-minute bars aligned to 09:15, never spanning sessions; ``t`` = bucket start.
    A missing minute just leaves a thinner bucket (no synthetic bars)."""
    if minutes == 1:
        return [Bar(b.t, b.o, b.h, b.l, b.c, b.v) for b in bars_1m]
    return _pb_resample(bars_1m, minutes, SESSION_OPEN_MIN)


@dataclass
class DaySession:
    day: date
    bars: list[Bar]                 # 1m bars 09:15 → 15:14 IST
    prev_daily: Bar | None          # prior session's bhavcopy (o/h/l/c official)
    daily_history: list[Bar]        # bhavcopy bars strictly before ``day``
    in_period: bool = True          # False for warm-up sessions before ``start``

    def resampled(self, minutes: int) -> list[Bar]:
        return resample(self.bars, minutes)


@dataclass
class SymbolData:
    symbol: str
    sessions: list[DaySession] = field(default_factory=list)

    @property
    def period_sessions(self) -> list[DaySession]:
        return [s for s in self.sessions if s.in_period]

    def all_bars(self) -> list[Bar]:
        return [b for s in self.sessions for b in s.bars]


class CandleSource:
    def __init__(self, max_date: date = IS_END, allow_oos: bool = False):
        check_date(max_date, allow_oos)
        self.max_date = max_date
        self.allow_oos = allow_oos
        self._ids: dict[str, int] = {}
        self._sessions: list[date] | None = None

    # ── low level ──────────────────────────────────────────────────────────

    def _guard(self, end: date) -> None:
        if end > self.max_date:
            from .periods import OOSAccessError
            raise OOSAccessError(f"query end {end} past this source's max_date {self.max_date}")

    def symbol_id(self, name: str, exchange: str = "NSE") -> int | None:
        key = f"{exchange}:{name}"
        if key not in self._ids:
            from django.db import connection
            with connection.cursor() as cur:
                cur.execute("SELECT id FROM market_data_symbol WHERE exchange=%s AND tradingsymbol=%s",
                            [exchange, name])
                row = cur.fetchone()
            self._ids[key] = row[0] if row else None
        return self._ids[key]

    def _rows(self, sid: int, interval: str, start: date | None, end: date) -> list[Bar]:
        from django.db import connection
        self._guard(end)
        hi = datetime(end.year, end.month, end.day, tzinfo=IST) + timedelta(days=1)
        lo = (datetime(start.year, start.month, start.day, tzinfo=IST) if start
              else datetime(2000, 1, 1, tzinfo=UTC))
        with connection.cursor() as cur:
            cur.execute(
                "SELECT extract(epoch from t)::bigint, o::float8, h::float8, l::float8, "
                "c::float8, v::float8 FROM market_data_candle "
                "WHERE symbol_id=%s AND interval=%s AND t >= %s AND t < %s ORDER BY t",
                [sid, interval, lo, hi])
            return [Bar(*r) for r in cur.fetchall()]

    def minute_bars(self, symbol: str, start: date, end: date, *, cut_last: bool = True,
                    exchange: str = "NSE") -> list[Bar]:
        self._guard(end)
        sid = self.symbol_id(symbol, exchange)
        if sid is None:
            return []
        rows = self._rows(sid, "1m", start, end)
        if cut_last:
            rows = [b for b in rows if SESSION_OPEN_MIN <= ist_minute(b.t) <= LAST_MIN]
        return rows

    def daily_bars(self, symbol: str, end: date, start: date | None = None,
                   exchange: str = "NSE") -> list[Bar]:
        self._guard(end)
        sid = self.symbol_id(symbol, exchange)
        return [] if sid is None else self._rows(sid, "1d", start, end)

    # ── calendar ───────────────────────────────────────────────────────────

    def sessions(self, start: date | None = None, end: date | None = None) -> list[date]:
        """NSE sessions (dates with a bhavcopy row for the reference symbol) up to max_date."""
        if self._sessions is None:
            self._sessions = [day_of(b.t) for b in self.daily_bars(REFERENCE_SYMBOL, self.max_date)]
        return [d for d in self._sessions
                if (start is None or d >= start) and (end is None or d <= end)]

    def calendar(self):
        """``is_trading_day(d)``: DB sessions inside the data range (≤ max_date),
        ``market_calendar`` (weekends + listed holidays) beyond it. Never reads
        sessions past max_date, so an IS run cannot learn OOS holidays."""
        known = set(self.sessions())
        first = min(known) if known else None
        last = self.max_date

        def is_trading_day(d: date) -> bool:
            if first and first <= d <= last:
                return d in known
            from apps.market_data.services.market_calendar import is_trading_day as cal
            return cal(d)
        return is_trading_day

    # ── per-symbol load ────────────────────────────────────────────────────

    def load_symbol(self, symbol: str, start: date, end: date, warmup_sessions: int = 3,
                    exchange: str = "NSE") -> SymbolData:
        """One symbol's sessions in [start, end] plus up to ``warmup_sessions``
        earlier sessions (flagged ``in_period=False``) for indicator warm-up."""
        self._guard(end)
        daily = self.daily_bars(symbol, end, exchange=exchange)
        sess = self.sessions(end=end)
        before = [d for d in sess if d < start][-warmup_sessions:] if warmup_sessions else []
        lo = before[0] if before else start
        mins = self.minute_bars(symbol, lo, end, exchange=exchange)
        by_day: dict[date, list[Bar]] = {}
        for b in mins:
            by_day.setdefault(day_of(b.t), []).append(b)
        del mins
        daily_by_day = {day_of(b.t): i for i, b in enumerate(daily)}
        out = SymbolData(symbol=symbol)
        for d in sorted(by_day):
            hist_end = _index_before(daily, daily_by_day, d)
            hist = daily[:hist_end]
            out.sessions.append(DaySession(
                day=d, bars=by_day[d], prev_daily=hist[-1] if hist else None,
                daily_history=hist, in_period=start <= d <= end))
        return out


def _index_before(daily: list[Bar], idx: dict[date, int], d: date) -> int:
    """Number of daily bars strictly before ``d``."""
    if d in idx:
        return idx[d]
    n = 0
    for b in daily:
        if day_of(b.t) < d:
            n += 1
        else:
            break
    return n
