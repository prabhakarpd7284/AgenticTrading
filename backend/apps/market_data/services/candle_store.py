"""Persistent candle store for closed trading sessions.

Once a session is over, its bars **never change** — but we keep re-paying
the broker rate-limit (400ms × N bars) on every cockpit panel that asks
for that session. This module is the source of truth for closed-session
candles: fetch once, cache for 30 days, serve every subsequent request
in <50ms regardless of the broker throttle.

Topology:
- `bars:{interval}:{symbol}:{YYYY-MM-DD}` Redis keys, JSON-serialised
  list of {t,o,h,l,c,v} dicts.
- TTL: 30 days. Long enough for any realistic cockpit time-travel window;
  short enough that Redis doesn't accumulate forever.
- Today's in-progress session is NOT stored here — the existing 30-60s
  service caches handle live data (it changes minute-by-minute).
- `fetch_bars()` is the central read path: store check → broker → store
  write-back if the session is closed.

The pre-warmer (`warm_historical_bars` in tasks/warmers.py) fires at
15:35 IST after market close and populates today's bars for every
watchlist symbol so the very first cockpit hit tomorrow morning is fast.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from django.core.cache import cache

from trading.utils.time_utils import is_market_open

_TTL_CLOSED_SESSION = 60 * 60 * 24 * 30  # 30 days


def _key(interval: str, symbol: str, session: date) -> str:
    """Redis key for a closed session's bars."""
    return f"bars:{interval}:{symbol}:{session.isoformat()}"


def _is_closed_session(session: date) -> bool:
    """True if the session is safely in the past — bars won't change.
    Today is closed once market close (15:30 IST) has passed."""
    today = date.today()
    if session < today:
        return True
    if session == today and not is_market_open():
        # Post-market on the same trading day — bars are final.
        # is_market_open returns False on weekends/holidays too, which
        # means today's "session" never opened — also safe to cache.
        from trading.utils.time_utils import MARKET_CLOSE
        return datetime.now().time() > MARKET_CLOSE
    return False


def get_bars(interval: str, symbol: str, session: date) -> Optional[list[dict]]:
    """Read closed-session bars from the persistent store. None if absent."""
    return cache.get(_key(interval, symbol, session))


def put_bars(interval: str, symbol: str, session: date, bars: list[dict]) -> None:
    """Write bars to the persistent store. Only honoured for closed sessions —
    storing today's in-progress data would freeze a stale snapshot. No-op
    if session is still active or bars list is empty."""
    if not bars or not _is_closed_session(session):
        return
    cache.set(_key(interval, symbol, session), bars, _TTL_CLOSED_SESSION)


def fetch_bars(
    symbol: str,
    session: date,
    interval: str,
    angel_interval: str,
    exchange_resolver,
) -> list[dict]:
    """Persistent-store-aware fetch: read-through cache for closed sessions,
    plain broker fetch for today's live session.

    Args:
      symbol: Trading symbol (e.g. "RELIANCE")
      session: Target trading day
      interval: Short label for cache key ("1m", "5m")
      angel_interval: Angel One SDK interval name ("ONE_MINUTE", "FIVE_MINUTE")
      exchange_resolver: Callable[[symbol], str] — defer to caller so we
                        don't import ticker_service at module load.

    Returns: List of {t,o,h,l,c,v} dicts (possibly empty on failure).
    """
    # 1. Persistent store fast-path for closed sessions
    if _is_closed_session(session):
        hit = get_bars(interval, symbol, session)
        if hit is not None:
            return hit

    # 2. Broker miss path
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service

        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            return []
        start = session.strftime("%Y-%m-%d 09:15")
        end = session.strftime("%Y-%m-%d 15:30")
        exch = exchange_resolver(symbol)
        try:
            raw = broker.fetch_candles(token, start, end, angel_interval, exchange=exch) or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, angel_interval) or []
    except Exception:  # noqa: BLE001 — broker hiccups must not crash the panel
        return []

    bars = [
        {"t": str(r[0]), "o": float(r[1]), "h": float(r[2]),
         "l": float(r[3]), "c": float(r[4]),
         "v": int(r[5]) if len(r) > 5 else 0}
        for r in raw if len(r) >= 5
    ]

    # 3. Write-back: only persist closed sessions (today's bars change)
    if bars:
        put_bars(interval, symbol, session, bars)
    return bars


def fetch_intraday_bars(
    symbol: str,
    interval: str,
    angel_interval: str,
    exchange_resolver,
    short_ttl: int = 30,
) -> list[dict]:
    """One-call replacement for the per-service `_fetch_1m_today` / `_fetch_5m`
    pattern. Handles BOTH closed-session (persistent store) and live-session
    (short-TTL shared cache) paths transparently.

    All cockpit panels that want today/yesterday's intraday bars should call
    this — it deduplicates broker calls across panels (VWAP + Tape + ORB
    rendered together share a single fetch) and gives free time-travel
    speedup via the closed-session store.
    """
    from trading.utils.time_utils import intraday_session_date

    session = intraday_session_date()
    if _is_closed_session(session):
        # Closed sessions: read-through persistent store, no short-TTL layer
        return fetch_bars(symbol, session, interval, angel_interval, exchange_resolver)
    # Live session: short-TTL shared key so panels reusing the same minute
    # avoid hitting the broker N times. Shape mirrors persistent keys so
    # the two layers are visually adjacent in Redis.
    live_key = f"bars_live:{interval}:{symbol}:{session.isoformat()}"
    cached = cache.get(live_key)
    if cached is not None:
        return cached
    bars = fetch_bars(symbol, session, interval, angel_interval, exchange_resolver)
    cache.set(live_key, bars, short_ttl)
    return bars
