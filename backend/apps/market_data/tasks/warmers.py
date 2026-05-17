"""Cache warmers for the slow market-data builders.

Pulse and Rotation are expensive on a cold cache (10-17s) because each rebuild
hits the broker for VIX + NIFTY + 22 sector quotes. Both have short TTLs
(30s pulse, 60s rotation) since the dashboard polls them frequently.

Beat keeps the cache continuously warm so the user-facing GET is always a
sub-100ms cache hit. We force=True at a cadence slightly under each TTL.
"""
from __future__ import annotations

import structlog
from celery import shared_task

log = structlog.get_logger()


@shared_task
def warm_pulse() -> dict:
    from apps.market_data.services.pulse_service import build_pulse
    import time
    t0 = time.time()
    build_pulse(force=True)
    ms = int((time.time() - t0) * 1000)
    log.info("warmer.pulse", ms=ms)
    return {"builder": "pulse", "ms": ms}


@shared_task
def warm_rotation() -> dict:
    from apps.market_data.services.rotation_service import build_rotation
    import time
    t0 = time.time()
    build_rotation(force=True)
    ms = int((time.time() - t0) * 1000)
    log.info("warmer.rotation", ms=ms)
    return {"builder": "rotation", "ms": ms}


@shared_task
def warm_historical_bars(interval: str = "1m") -> dict:
    """Persist today's bars to the candle store after market close.

    Fires once a weekday at 15:35 IST (after the 15:30 close). Iterates
    the watchlist, fetches each symbol's session bars, and stores them
    with a 30-day TTL so tomorrow's cockpit time-travel hits the cache
    instead of the broker rate limit.

    The fetch path itself writes back to the store on success — this
    task just triggers the path for every watchlist symbol in parallel.
    """
    import time
    from apps.market_data.services import candle_store
    from apps.market_data.services._parallel import parallel_symbols
    from trading.models import WatchlistEntry
    from trading.services.ticker_service import ticker_service
    from trading.utils.time_utils import intraday_session_date

    session = intraday_session_date()
    if not candle_store._is_closed_session(session):
        log.warning("warmer.historical_bars.skipped",
                    reason="session-still-open", session=session.isoformat())
        return {"skipped": True, "reason": "session-still-open"}

    angel_interval = {"1m": "ONE_MINUTE", "5m": "FIVE_MINUTE"}.get(interval, "ONE_MINUTE")
    seen: set[str] = set()
    symbols: list[str] = []
    for s in WatchlistEntry.objects.values_list("symbol", flat=True)[:80]:
        s = (s or "").upper()
        if s and s not in seen:
            seen.add(s)
            symbols.append(s)

    t0 = time.time()
    results = parallel_symbols(
        symbols,
        lambda sym: len(candle_store.fetch_bars(
            sym, session, interval, angel_interval,
            ticker_service.resolve_exchange,
        )),
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    populated = sum(1 for n in results if n > 0)
    total_bars = sum(results)
    log.info("warmer.historical_bars",
             session=session.isoformat(), interval=interval,
             symbols=len(symbols), populated=populated,
             total_bars=total_bars, ms=elapsed_ms)
    return {
        "session": session.isoformat(),
        "interval": interval,
        "symbols": len(symbols),
        "populated": populated,
        "total_bars": total_bars,
        "ms": elapsed_ms,
    }
