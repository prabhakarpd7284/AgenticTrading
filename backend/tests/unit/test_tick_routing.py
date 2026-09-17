"""Options join the data pipeline without joining the equity strategies.

Streaming the ATM chain gives us option ticks → ltp cache → candle store →
paper fills, which is what makes options tradeable by the rest of the system.
But the screener's ten strategies are equity setups; running "Breakout Long"
over a NIFTY option would emit confident nonsense.

So the tick router feeds every tick to the data path and only equity symbols to
the engine.
"""
from __future__ import annotations

import pytest

from apps.market_data.services.tick_routing import engine_router


def test_equity_ticks_reach_the_engine():
    seen = []
    route = engine_router(lambda s, l, v=0: seen.append(s), {"RELIANCE", "SBIN"})

    route("RELIANCE", 100.0, 10)

    assert seen == ["RELIANCE"]


def test_option_ticks_do_not_reach_the_engine():
    seen = []
    route = engine_router(lambda s, l, v=0: seen.append(s), {"RELIANCE"})

    route("NIFTY15SEP2623500CE", 120.0, 5)

    assert seen == [], "equity strategies must not evaluate option contracts"


def test_router_returns_none_so_it_can_wrap_transparently():
    route = engine_router(lambda *a, **k: None, {"RELIANCE"})
    assert route("RELIANCE", 100.0, 10) is None


def test_engine_failure_on_one_symbol_does_not_break_the_stream():
    """The router sits under the data tee; an engine exception must not stop
    the tick reaching the cache and candle store."""
    def boom(symbol, ltp, volume=0):
        raise RuntimeError("strategy blew up")

    route = engine_router(boom, {"RELIANCE"})
    route("RELIANCE", 100.0, 10)      # must not raise


def test_empty_engine_universe_routes_nothing_to_the_engine():
    seen = []
    route = engine_router(lambda s, l, v=0: seen.append(s), set())

    route("RELIANCE", 100.0, 10)

    assert seen == []
