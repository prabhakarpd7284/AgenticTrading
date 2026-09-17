"""Publish live ticks onto the channel layer so browsers can see them.

`TickConsumer` (ws/ticks/) only relays what arrives in the ``ticks.<tenant_id>``
group — it never produced anything itself, so the socket connected and stayed
silent. This is the missing producer: the screener's tick stream calls it for
every tick it receives from Angel One's websocket.

Payload shape is dictated by the existing consumer + frontend contract:
`PositionsPage.tsx` subscribes with *symbols* in a field named ``tokens`` and
keys incoming ticks on ``symbol ?? token``, while `TickConsumer.tick` filters
on ``tick["token"]``. So the symbol goes in ``token`` — renaming either side
would be a wider breaking change than this bridge warrants.
"""
from __future__ import annotations

import logging
from datetime import datetime

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from apps.market_data.services.candle_ingestor import CandleAggregator, persist_bars

log = logging.getLogger(__name__)

# A tick older than this is not a live price. Readers fall back to candles
# on a miss, which is the honest answer when the feed has stopped.
_LTP_TTL_SECONDS = 120


def publish_tick(tenant_id, symbol: str, ltp: float, volume: int = 0) -> None:
    """Fan one tick out to every ws/ticks/ client of ``tenant_id``.

    Never raises. This runs in the tick stream's hot path, and a degraded
    channel layer must not stop ticks reaching the screener engine — the
    engine is what produces signals; the browser feed is a nice-to-have.
    """
    try:
        layer = get_channel_layer()
        if layer is None:
            return
        async_to_sync(layer.group_send)(
            f"ticks.{tenant_id}",
            {
                "type": "tick",
                "tick": {
                    "token": symbol,
                    "symbol": symbol,
                    "ltp": ltp,
                    "volume": volume,
                },
            },
        )
    except Exception:  # noqa: BLE001 — see docstring: must never raise.
        log.debug("tick_publisher.send_failed symbol=%s", symbol, exc_info=True)


def cache_ltp(symbol: str, ltp: float) -> None:
    """Store the last traded price where the rest of the system looks for it.

    `place_order.reference_price()` reads ``ltp:<symbol>`` and is what portfolio
    mark-to-market, position sizing and paper fills all price off. Nothing ever
    wrote this key, so every reader silently fell back to the newest stored
    Candle — stale by minutes, or on a long-idle install by months.

    TTL keeps a dead feed from serving yesterday's price as if it were live;
    readers treat a miss as "unknown" and fall back to candles.
    """
    try:
        from django.core.cache import cache

        cache.set(f"ltp:{symbol}", ltp, timeout=_LTP_TTL_SECONDS)
    except Exception:  # noqa: BLE001 — hot path, never raise.
        log.debug("tick_publisher.cache_failed symbol=%s", symbol, exc_info=True)


def tee_to_browser(on_tick, tenant_id):
    """Wrap a tick handler so one tick feeds the engine, the cache, the
    persisted candle history and the browser.

    Order is deliberate, most to least important:

    1. the engine — signals are the product
    2. the LTP cache — everything prices off it
    3. the candle store — the durable fallback when the cache is cold
    4. the browser feed — the only one of the four that is decoration

    Every step after the first is wrapped so it cannot cost the engine a tick.
    """
    aggregator = CandleAggregator()

    def handler(symbol: str, ltp: float, volume: int = 0, *args, ts=None, **kwargs):
        on_tick(symbol, ltp, volume, *args, **kwargs)
        cache_ltp(symbol, ltp)
        try:
            bar = aggregator.add(
                symbol, ltp, volume, ts or datetime.now(),
            )
            if bar is not None:
                persist_bars([bar])
        except Exception:  # noqa: BLE001 — bookkeeping, never fatal.
            log.debug("tick_publisher.ingest_failed symbol=%s", symbol, exc_info=True)
        publish_tick(tenant_id, symbol, ltp, volume)

    handler.aggregator = aggregator      # exposed so the session can drain it
    return handler
