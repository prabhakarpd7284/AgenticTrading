"""Turn the live tick stream into persisted 1-minute candles.

`data_port.candles()` and `place_order.reference_price()` both fall back to the
local `Candle` table when the ltp cache is cold — and that table has always had
zero rows. data_port's own docstring admits the ingestor "isn't running in dev
yet". A pricing fallback that always answers 0.0 does not fail loudly; it turns
a missing price into a fabricated number, which is exactly how the 2026-09-09
EOD square-off booked a flat P&L on a trade that had moved.

Two pieces:

* `CandleAggregator` — pure, no DB, no clock. Folds ticks into 1-minute OHLCV
  and emits a bar when the minute rolls over.
* `persist_bars()` — writes completed bars, creating `Symbol` rows on demand.

Angel's tick volume is cumulative for the trading day, so a bar's volume is the
delta observed across that bar. The first bar of a symbol therefore
under-reports slightly (we never saw the pre-open cumulative) — acceptable, and
far better than restating the whole day's volume into every bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import structlog

log = structlog.get_logger(__name__)


@dataclass
class Bar:
    symbol: str
    t: datetime          # bar open time, truncated to the minute
    o: float
    h: float
    l: float
    c: float
    v: int


@dataclass
class _Open:
    """A bar still being built."""
    t: datetime
    o: float
    h: float
    l: float
    c: float
    v_first: int
    v_last: int

    def to_bar(self, symbol: str) -> Bar:
        return Bar(
            symbol=symbol, t=self.t, o=self.o, h=self.h, l=self.l, c=self.c,
            v=max(0, self.v_last - self.v_first),
        )


class CandleAggregator:
    """Folds ticks into 1-minute bars. Not thread-safe by design — it runs on
    the single tick-stream thread."""

    def __init__(self):
        self._open: dict[str, _Open] = {}

    @property
    def pending_count(self) -> int:
        return len(self._open)

    def add(self, symbol: str, ltp: float, volume: int, ts: datetime) -> Bar | None:
        """Fold one tick in. Returns the previous bar if this tick closed it."""
        if not ltp or ltp <= 0:
            return None

        minute = ts.replace(second=0, microsecond=0)
        cur = self._open.get(symbol)

        if cur is None:
            self._open[symbol] = _Open(
                t=minute, o=ltp, h=ltp, l=ltp, c=ltp,
                v_first=volume, v_last=volume,
            )
            return None

        if minute == cur.t:
            cur.h = max(cur.h, ltp)
            cur.l = min(cur.l, ltp)
            cur.c = ltp
            cur.v_last = max(cur.v_last, volume)
            return None

        if minute < cur.t:
            # Stale tick arriving after the bar rolled — dropping it is safer
            # than back-dating a bar we may already have written.
            return None

        # Minute rolled over (possibly several, for a thin symbol): close the
        # bar we have and start a new one.
        completed = cur.to_bar(symbol)
        self._open[symbol] = _Open(
            t=minute, o=ltp, h=ltp, l=ltp, c=ltp,
            v_first=volume, v_last=volume,
        )
        return completed

    def drain(self) -> list[Bar]:
        """Close every open bar. Call at the session close so the final minute
        of the day is not lost."""
        bars = [op.to_bar(sym) for sym, op in self._open.items()]
        self._open.clear()
        return bars


def persist_bars(bars: list[Bar]) -> int:
    """Write completed bars, creating Symbol rows on demand. Never raises."""
    if not bars:
        return 0
    try:
        from django.db import transaction

        from apps.market_data.models import Candle, Symbol

        written = 0
        with transaction.atomic():
            for bar in bars:
                symbol = _ensure_symbol(Symbol, bar.symbol)
                if symbol is None:
                    continue
                t = _aware(bar.t)
                # unique_together (symbol, interval, t) — a replayed tick must
                # update the bar rather than blow up the whole batch.
                Candle.objects.update_or_create(
                    symbol=symbol, interval="1m", t=t,
                    defaults={"o": bar.o, "h": bar.h, "l": bar.l,
                              "c": bar.c, "v": bar.v},
                )
                written += 1
        return written
    except Exception:  # noqa: BLE001 — ingest must never disturb the tick path.
        log.warning("candle_ingestor.persist_failed", count=len(bars), exc_info=True)
        return 0


def _aware(t: datetime) -> datetime:
    """Attach the local timezone to a naive bar timestamp.

    The aggregator is pure and clock-agnostic, so it happily carries naive
    datetimes. Storing those misorders rows across DST and reads back
    shifted — and `reference_price()` picks the newest row by ``t``.
    """
    from django.utils import timezone as djtz

    if djtz.is_aware(t):
        return t
    return djtz.make_aware(t, djtz.get_current_timezone())


# Derivative segments to search after the NSE equity master misses. BFO carries
# SENSEX options, whose symbols can otherwise look like NFO ones.
_DERIVATIVE_EXCHANGES = ("NFO", "BFO")


def _resolve_symbol_meta(tradingsymbol: str) -> dict | None:
    """Look an instrument up in the symbol master.

    Returns exchange/token/name/segment/lot_size/tick_size, or None on a miss.
    Resolving the real exchange matters: hardcoding NSE writes every option as
    an equity instrument, which mislabels the candle and breaks any later
    lookup that filters by exchange.
    """
    try:
        from trading.services.ticker_service import TickerService

        ts = TickerService()

        info = ts.get_info(tradingsymbol)
        if info:
            return {
                "exchange": info.get("exch_seg") or "NSE",
                "token": str(info.get("token") or ""),
                "name": info.get("name") or tradingsymbol,
                "segment": info.get("instrument_type") or "",
                "lot_size": int(info.get("lot_size") or 1),
                "tick_size": _tick_rupees(info.get("tick_size")),
            }

        for exchange in _DERIVATIVE_EXCHANGES:
            token = ts.get_token(tradingsymbol, exchange)
            if not token:
                continue
            # Read the real contract metadata rather than assuming. Lot sizes
            # are never hardcoded (CLAUDE.md invariant #8) — NIFTY is 65,
            # BANKNIFTY 30, SENSEX 20, and they change by circular.
            raw = _derivative_record(ts, tradingsymbol) or {}
            return {
                "exchange": raw.get("exch_seg") or exchange,
                "token": str(token),
                "name": raw.get("name") or tradingsymbol,
                "segment": raw.get("instrumenttype") or "",
                "lot_size": int(raw.get("lotsize") or 1),
                "tick_size": _tick_rupees(raw.get("tick_size")),
            }
    except Exception:  # noqa: BLE001 — a master miss must not block ingest.
        log.debug("candle_ingestor.resolve_failed symbol=%s",
                  tradingsymbol, exc_info=True)
    return None


def _tick_rupees(raw) -> float:
    """Angel reports tick_size in paise; the model field is rupees.

    Storing the raw value puts 10.0 in a rupee field — a 200x error on a
    5-paise tick. Anchor: NIFTY options tick at Rs 0.05 and the master
    reports 5.000000 for them.
    """
    try:
        paise = float(raw or 0)
    except (TypeError, ValueError):
        paise = 0.0
    return (paise / 100.0) if paise > 0 else 0.05


def _derivative_record(ts, tradingsymbol: str) -> dict | None:
    """Full instrument row for a derivative from the loaded symbol master.

    `get_info` only covers the NSE equity index, and `get_token` returns just a
    token — neither carries lot size, which is the field we must not guess.
    """
    try:
        ts._ensure_loaded()
        for row in ts._instruments or ():
            if row.get("symbol") == tradingsymbol:
                return row
    except Exception:  # noqa: BLE001
        pass
    return None


def _ensure_symbol(Symbol, tradingsymbol: str):
    """Get or create the Symbol row a Candle needs as its FK."""
    row = Symbol.objects.filter(tradingsymbol=tradingsymbol).first()
    if row is not None:
        return row

    meta = _resolve_symbol_meta(tradingsymbol)
    if meta is None:
        # Unknown instrument. Still ingest — dropping market data because the
        # master is stale would be worse than an under-described Symbol row —
        # but say so, since the exchange is a guess.
        log.warning("candle_ingestor.symbol_unresolved symbol=%s", tradingsymbol)
        meta = {"exchange": "", "token": "", "name": tradingsymbol,
                "segment": "", "lot_size": 1, "tick_size": 0.05}

    row, _ = Symbol.objects.get_or_create(
        tradingsymbol=tradingsymbol, defaults=meta,
    )
    return row
