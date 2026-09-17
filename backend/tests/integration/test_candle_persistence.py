"""Completed bars reach the Candle table, and the pricing fallback then works.

The whole point of ingesting: `reference_price()` falls back to the newest
local Candle when the ltp cache is cold. With 0 rows that fallback returned 0.0
for everything, which is how a missing EOD price became a fabricated flat P&L.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from apps.market_data.models import Candle, Symbol
from apps.market_data.services.candle_ingestor import Bar, persist_bars

pytestmark = pytest.mark.django_db


def _a_live_option_symbol() -> str:
    """A NIFTY option that exists right now.

    Pinning a literal contract (NIFTY15SEP2623500PE) made these tests fail
    two days after that expiry passed — a false red with no code change.
    """
    from trading.services.ticker_service import TickerService
    from trading.utils.expiry_utils import iso_to_angel_long, next_expiry_date

    expiry = iso_to_angel_long(next_expiry_date("NIFTY").isoformat())
    ts = TickerService(); ts._ensure_loaded()
    for row in ts._instruments or ():
        if (row.get("name") == "NIFTY" and row.get("expiry") == expiry
                and "OPT" in (row.get("instrumenttype") or "")):
            return row.get("symbol")
    pytest.skip("symbol master has no NIFTY options for the next expiry")


def _bar(sym="SBIN", minute=1, c=101.0):
    t = datetime(2026, 9, 10, 10, minute, 0)
    return Bar(symbol=sym, t=t, o=100.0, h=102.0, l=99.0, c=c, v=500)


def test_a_bar_is_written_and_creates_its_symbol():
    assert Symbol.objects.count() == 0

    written = persist_bars([_bar()])

    assert written == 1
    assert Symbol.objects.filter(tradingsymbol="SBIN").exists()
    row = Candle.objects.get()
    assert row.interval == "1m"
    assert float(row.c) == 101.0
    assert row.v == 500


def test_replaying_the_same_minute_updates_rather_than_duplicates():
    """unique_together (symbol, interval, t) — a redelivered tick must not
    explode the batch or double-count the bar."""
    persist_bars([_bar(c=101.0)])
    persist_bars([_bar(c=103.0)])

    assert Candle.objects.count() == 1
    assert float(Candle.objects.get().c) == 103.0


def test_persisting_is_never_fatal(monkeypatch):
    """Ingest runs on the tick path; a DB hiccup must not kill the feed."""
    import apps.market_data.services.candle_ingestor as ci

    def boom(*a, **k):
        raise RuntimeError("db gone")

    monkeypatch.setattr(ci, "_ensure_symbol", boom)
    assert persist_bars([_bar()]) == 0      # no exception


def test_reference_price_falls_back_to_the_newest_candle():
    """The behaviour this whole module exists for."""
    from django.core.cache import cache

    from apps.trading.services.place_order import reference_price

    cache.delete("ltp:SBIN")
    assert reference_price("SBIN", None) == 0.0     # nothing to fall back to

    persist_bars([_bar(minute=1, c=101.0), _bar(minute=2, c=104.0)])

    assert reference_price("SBIN", None) == 104.0   # newest bar's close


def test_bar_timestamps_are_stored_timezone_aware():
    """A naive timestamp in a market-data table misorders across DST and
    reads back shifted — and `reference_price` picks the newest row by `t`."""
    from django.utils import timezone as djtz

    persist_bars([_bar()])

    t = Candle.objects.get().t
    assert djtz.is_aware(t), "candle timestamps must carry a timezone"


# ---------------------------------------------------------------------------
# Symbol rows must carry the instrument's real exchange and metadata.
#
# The first cut hardcoded exchange="NSE", which is right for the 98-symbol
# equity screener universe and silently wrong for anything on NFO/BFO. Options
# would be written as NSE instruments, mislabelling every option candle and
# breaking any later lookup that filters by exchange.
# ---------------------------------------------------------------------------
def test_equity_symbol_carries_its_real_metadata(monkeypatch):
    import apps.market_data.services.candle_ingestor as ci

    monkeypatch.setattr(ci, "_resolve_symbol_meta", lambda s: {
        "exchange": "NSE", "token": "2885", "name": "RELIANCE",
        "segment": "EQ", "lot_size": 1, "tick_size": 0.05,
    })
    persist_bars([_bar(sym="RELIANCE")])

    row = Symbol.objects.get(tradingsymbol="RELIANCE")
    assert row.exchange == "NSE"
    assert row.token == "2885"
    assert row.lot_size == 1


def test_option_symbol_is_not_mislabelled_as_nse(monkeypatch):
    import apps.market_data.services.candle_ingestor as ci

    monkeypatch.setattr(ci, "_resolve_symbol_meta", lambda s: {
        "exchange": "NFO", "token": "47298", "name": "NIFTY",
        "segment": "OPTIDX", "lot_size": 65, "tick_size": 0.05,
    })
    persist_bars([_bar(sym="NIFTY15SEP2623500PE")])

    row = Symbol.objects.get(tradingsymbol="NIFTY15SEP2623500PE")
    assert row.exchange == "NFO", "options must not be written as NSE"
    assert row.lot_size == 65


def test_unresolvable_symbol_still_ingests(monkeypatch):
    """A symbol master miss must not silently drop market data."""
    import apps.market_data.services.candle_ingestor as ci

    monkeypatch.setattr(ci, "_resolve_symbol_meta", lambda s: None)

    assert persist_bars([_bar(sym="MYSTERY")]) == 1
    assert Candle.objects.filter(symbol__tradingsymbol="MYSTERY").exists()


def test_existing_symbol_row_is_reused(monkeypatch):
    import apps.market_data.services.candle_ingestor as ci

    monkeypatch.setattr(ci, "_resolve_symbol_meta", lambda s: {
        "exchange": "NSE", "token": "2885", "name": "RELIANCE",
        "segment": "EQ", "lot_size": 1, "tick_size": 0.05,
    })
    persist_bars([_bar(sym="RELIANCE", minute=1)])
    persist_bars([_bar(sym="RELIANCE", minute=2)])

    assert Symbol.objects.filter(tradingsymbol="RELIANCE").count() == 1
    assert Candle.objects.count() == 2


def test_option_lot_size_comes_from_the_symbol_master():
    """CLAUDE.md invariant: lot sizes are never hardcoded. The first cut wrote
    lot_size=1 for every derivative, which is wrong for every index option
    (NIFTY is 65) and silently mis-sizes anything that reads it back."""
    from apps.market_data.services.candle_ingestor import _resolve_symbol_meta

    meta = _resolve_symbol_meta(_a_live_option_symbol())

    assert meta is not None, "the contract must resolve"
    assert meta["exchange"] == "NFO"
    assert meta["lot_size"] == 65, f"expected NIFTY lot 65, got {meta['lot_size']}"


def test_tick_size_is_converted_from_paise_to_rupees():
    """Angel's symbol master reports tick_size in paise. Storing the raw value
    puts 10.0 in a rupee field — a 200x error on a ₹0.05 tick. Anchor: NIFTY
    options tick at ₹0.05 and the master reports 5.000000 for them."""
    from apps.market_data.services.candle_ingestor import _resolve_symbol_meta

    option = _resolve_symbol_meta(_a_live_option_symbol())
    assert option["tick_size"] == 0.05, "5.000000 paise must read as ₹0.05"

    equity = _resolve_symbol_meta("RELIANCE")
    assert equity["tick_size"] < 1.0, (
        f"a rupee tick size below ₹1 is expected, got {equity['tick_size']}"
    )
