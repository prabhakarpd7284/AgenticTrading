"""Aggregate live ticks into 1-minute candles.

`data_port` and `place_order.reference_price()` both fall back to the local
Candle table when the ltp cache is cold. That table has 0 rows and always has —
the docstring in data_port.py admits the ingestor "isn't running in dev yet".
So the fallback silently returns 0.0 for everything, which is what turned a
missing EOD price into a fabricated flat P&L on 2026-09-09.

Angel's tick volume is *cumulative for the day*, so a bar's volume is the delta
across the bar, not the raw field.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from apps.market_data.services.candle_ingestor import CandleAggregator


def _t(minute: int, second: int = 0) -> datetime:
    return datetime(2026, 9, 10, 10, minute, second)


def test_ticks_in_one_minute_build_a_single_open_bar():
    agg = CandleAggregator()
    assert agg.add("SBIN", 100.0, 1000, _t(1, 0)) is None
    assert agg.add("SBIN", 102.0, 1200, _t(1, 30)) is None
    assert agg.add("SBIN", 101.0, 1500, _t(1, 59)) is None
    assert agg.pending_count == 1


def test_a_new_minute_completes_the_previous_bar():
    agg = CandleAggregator()
    agg.add("SBIN", 100.0, 1000, _t(1, 0))
    agg.add("SBIN", 105.0, 1200, _t(1, 30))
    agg.add("SBIN", 98.0, 1400, _t(1, 45))

    bar = agg.add("SBIN", 99.0, 1600, _t(2, 0))

    assert bar is not None
    assert bar.symbol == "SBIN"
    assert bar.t == _t(1, 0)
    assert (bar.o, bar.h, bar.l, bar.c) == (100.0, 105.0, 98.0, 98.0)


def test_bar_volume_is_the_cumulative_delta():
    """Angel reports volume-for-the-day; a raw copy would restate the whole
    session's volume into every bar."""
    agg = CandleAggregator()
    agg.add("SBIN", 100.0, 5000, _t(1, 0))
    agg.add("SBIN", 101.0, 5400, _t(1, 30))
    bar = agg.add("SBIN", 102.0, 6000, _t(2, 0))

    assert bar.v == 400          # 5400 - 5000, within the bar


def test_symbols_are_aggregated_independently():
    agg = CandleAggregator()
    agg.add("SBIN", 100.0, 10, _t(1, 0))
    agg.add("INFY", 200.0, 20, _t(1, 0))

    sbin_bar = agg.add("SBIN", 110.0, 30, _t(2, 0))
    assert sbin_bar.symbol == "SBIN"
    assert sbin_bar.o == 100.0
    assert agg.pending_count == 2      # INFY's bar is still open


def test_a_skipped_minute_still_completes_the_open_bar():
    """Thin symbols do not tick every minute. The bar that exists must close."""
    agg = CandleAggregator()
    agg.add("PAGEIND", 35000.0, 5, _t(1, 0))

    bar = agg.add("PAGEIND", 35100.0, 9, _t(7, 0))

    assert bar is not None
    assert bar.t == _t(1, 0)
    assert bar.c == 35000.0


def test_out_of_order_tick_does_not_close_a_newer_bar():
    agg = CandleAggregator()
    agg.add("SBIN", 100.0, 10, _t(5, 0))
    late = agg.add("SBIN", 99.0, 11, _t(4, 30))

    assert late is None, "a stale tick must not emit a bar"


def test_drain_closes_every_open_bar():
    """Called at the close so the final minute is not lost."""
    agg = CandleAggregator()
    agg.add("SBIN", 100.0, 10, _t(1, 0))
    agg.add("INFY", 200.0, 20, _t(1, 0))

    bars = agg.drain()

    assert {b.symbol for b in bars} == {"SBIN", "INFY"}
    assert agg.pending_count == 0


def test_zero_price_ticks_are_ignored():
    agg = CandleAggregator()
    assert agg.add("SBIN", 0.0, 10, _t(1, 0)) is None
    assert agg.pending_count == 0
