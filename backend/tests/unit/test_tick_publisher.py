"""Ticks reaching the browser.

`TickConsumer` (ws/ticks/) is a pure relay: it forwards whatever lands in the
``ticks.<tenant_id>`` channel group to clients that subscribed to that key.
Until this publisher existed nothing ever sent to that group, so the socket
connected and then sat silent forever.

The frontend subscribes with *symbols* in a field named ``tokens``
(PositionsPage.tsx) and keys incoming ticks on ``symbol ?? token`` — so the
published payload must carry the symbol in ``token`` for the consumer's
subscription filter to match.
"""
from __future__ import annotations

import pytest
from channels.layers import get_channel_layer


@pytest.fixture
def layer(settings):
    settings.CHANNEL_LAYERS = {
        "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"},
    }
    return get_channel_layer()


async def test_publish_tick_reaches_the_tenant_group(layer):
    import asyncio

    from apps.market_data.services.tick_publisher import publish_tick

    tenant_id = "11111111-1111-1111-1111-111111111111"
    await layer.group_add(f"ticks.{tenant_id}", "test-channel")

    # Called off-thread exactly as production does — TickStream invokes this
    # from its websocket/polling thread, and async_to_sync refuses to run
    # inside a live event loop.
    await asyncio.to_thread(publish_tick, tenant_id, "RELIANCE", 1234.5, 900)

    msg = await layer.receive("test-channel")
    assert msg["type"] == "tick"
    assert msg["tick"]["token"] == "RELIANCE"
    assert msg["tick"]["symbol"] == "RELIANCE"
    assert msg["tick"]["ltp"] == 1234.5
    assert msg["tick"]["volume"] == 900


async def test_publish_tick_does_not_reach_a_different_tenant(layer):
    """Tick fan-out must respect tenant isolation."""
    import asyncio

    from apps.market_data.services.tick_publisher import publish_tick

    await layer.group_add("ticks.tenant-a", "channel-a")
    await asyncio.to_thread(publish_tick, "tenant-b", "INFY", 1500.0, 10)

    # Nothing was sent to tenant-a's group, so the receive must time out.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(layer.receive("channel-a"), timeout=0.2)


def test_publish_tick_never_raises_when_the_layer_is_broken(settings):
    """Tick publishing sits in the screener's hot path — a channel-layer
    failure must never take down the tick stream feeding the engine."""
    from apps.market_data.services.tick_publisher import publish_tick

    settings.CHANNEL_LAYERS = {
        "default": {"BACKEND": "tests.unit.test_tick_publisher.ExplodingLayer"},
    }
    # Must not raise.
    publish_tick("any-tenant", "TCS", 1.0, 1)


class ExplodingLayer:
    """Channel layer stand-in that fails on every send."""

    def __init__(self, *args, **kwargs):
        pass

    async def group_send(self, *args, **kwargs):
        raise RuntimeError("redis is down")


# ---------------------------------------------------------------------------
# tee_to_browser — wraps the engine's on_tick so one tick feeds both the
# screener engine and the browser feed.
# ---------------------------------------------------------------------------
async def test_tee_forwards_to_the_engine_and_the_browser(layer):
    import asyncio

    from apps.market_data.services.tick_publisher import tee_to_browser

    seen = []
    handler = tee_to_browser(
        lambda sym, ltp, vol=0: seen.append((sym, ltp, vol)),
        "22222222-2222-2222-2222-222222222222",
    )
    await layer.group_add("ticks.22222222-2222-2222-2222-222222222222", "ch")

    await asyncio.to_thread(handler, "SBIN", 812.4, 55)

    assert seen == [("SBIN", 812.4, 55)], "engine must still receive the tick"
    msg = await layer.receive("ch")
    assert msg["tick"]["symbol"] == "SBIN"
    assert msg["tick"]["ltp"] == 812.4


def test_tee_still_feeds_the_engine_when_publishing_fails(settings):
    """The engine produces signals; the browser feed is decoration. A broken
    channel layer must not cost us a tick."""
    from apps.market_data.services.tick_publisher import tee_to_browser

    settings.CHANNEL_LAYERS = {
        "default": {"BACKEND": "tests.unit.test_tick_publisher.ExplodingLayer"},
    }
    seen = []
    handler = tee_to_browser(lambda s, l, v=0: seen.append(s), "t")

    handler("TCS", 3000.0, 1)

    assert seen == ["TCS"]


# ---------------------------------------------------------------------------
# Live ticks must also land in the `ltp:<symbol>` cache.
#
# `place_order.reference_price()` reads that key and is what portfolio MTM,
# position sizing and paper fills all price off. Nothing ever wrote it, so
# every reader silently fell back to the most recent stored Candle — stale by
# minutes or, on this install, by months (`Trade.last_ltp` was None on every
# open trade). The tick tee already sees every tick; caching there makes the
# whole system price off live data.
# ---------------------------------------------------------------------------
def test_tee_caches_ltp_for_reference_price(settings):
    from django.core.cache import cache

    from apps.market_data.services.tick_publisher import tee_to_browser
    from apps.trading.services.place_order import reference_price

    settings.CHANNEL_LAYERS = {
        "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"},
    }
    cache.delete("ltp:SBIN")

    handler = tee_to_browser(lambda *a, **k: None, "t")
    handler("SBIN", 812.4, 55)

    assert cache.get("ltp:SBIN") == 812.4
    assert reference_price("SBIN", tenant_id=None) == 812.4


def test_tee_caches_ltp_even_if_channel_publish_fails(settings):
    """Pricing must not depend on the browser feed being healthy."""
    from django.core.cache import cache

    from apps.market_data.services.tick_publisher import tee_to_browser

    settings.CHANNEL_LAYERS = {
        "default": {"BACKEND": "tests.unit.test_tick_publisher.ExplodingLayer"},
    }
    cache.delete("ltp:TCS")

    tee_to_browser(lambda *a, **k: None, "t")("TCS", 3000.0, 1)

    assert cache.get("ltp:TCS") == 3000.0


# ---------------------------------------------------------------------------
# Ticks must also become persisted candles — the fallback the whole system
# prices off when the ltp cache is cold. See candle_ingestor.py.
# ---------------------------------------------------------------------------
def test_tee_folds_ticks_into_candles(settings, db):
    from datetime import datetime

    from apps.market_data.models import Candle
    from apps.market_data.services.tick_publisher import tee_to_browser

    settings.CHANNEL_LAYERS = {
        "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"},
    }
    handler = tee_to_browser(lambda *a, **k: None, "t")

    # Two ticks in one minute, then one in the next — closes the first bar.
    handler("SBIN", 100.0, 1000, ts=datetime(2026, 9, 10, 10, 1, 0))
    handler("SBIN", 105.0, 1400, ts=datetime(2026, 9, 10, 10, 1, 30))
    handler("SBIN", 103.0, 1600, ts=datetime(2026, 9, 10, 10, 2, 0))

    row = Candle.objects.get(symbol__tradingsymbol="SBIN", interval="1m")
    assert (float(row.o), float(row.h), float(row.c)) == (100.0, 105.0, 105.0)
    assert row.v == 400


def test_tee_survives_a_candle_write_failure(settings, db, monkeypatch):
    """Ingest is bookkeeping; it must never cost the engine a tick."""
    import apps.market_data.services.tick_publisher as tp

    settings.CHANNEL_LAYERS = {
        "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"},
    }
    monkeypatch.setattr(
        tp, "persist_bars", lambda bars: (_ for _ in ()).throw(RuntimeError("db")),
    )
    seen = []
    handler = tee_to_browser_local = tp.tee_to_browser(
        lambda s, l, v=0, **k: seen.append(s), "t",
    )
    from datetime import datetime
    handler("TCS", 100.0, 1, ts=datetime(2026, 9, 10, 10, 1, 0))
    handler("TCS", 101.0, 2, ts=datetime(2026, 9, 10, 10, 2, 0))

    assert seen == ["TCS", "TCS"], "engine must still receive every tick"
