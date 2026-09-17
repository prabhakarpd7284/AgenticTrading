"""Route ticks: everything to the data path, equities only to the engine.

Streaming the ATM option chain is what finally gives options an ltp cache, a
candle history and working paper fills — the same plumbing equities got. But
the screener's strategies are equity setups (breakout, breakdown, VWAP
reclaim); evaluating them over a NIFTY option would emit confident nonsense
that looks exactly like a real signal.

So the split is by destination, not by feed: the tee already fans every tick to
cache/candles/browser, and this decides which of those ticks the engine also
sees.
"""
from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


def engine_router(engine_on_tick, engine_symbols):
    """Wrap the engine handler so only ``engine_symbols`` reach it.

    Exceptions are contained: this sits *under* the data tee, so a strategy
    blowing up on one symbol must not stop that tick reaching the cache and
    candle store, which is what the rest of the system prices off.
    """
    universe = set(engine_symbols)

    def route(symbol: str, ltp: float, volume: int = 0, *args, **kwargs) -> None:
        if symbol not in universe:
            return
        try:
            engine_on_tick(symbol, ltp, volume, *args, **kwargs)
        except Exception:  # noqa: BLE001
            log.warning("tick_routing.engine_failed", symbol=symbol, exc_info=True)

    return route
