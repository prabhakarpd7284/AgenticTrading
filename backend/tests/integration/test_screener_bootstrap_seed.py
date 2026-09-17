"""Bootstrap must seed from the local candle store, and never claim success
while loading nothing.

Observed 2026-09-10 11:44: a mid-session restart logged

    Fetching prev-day 5m candles for 98 symbols...
    Cached 0 symbols of 5m candles
    Seeded 0/98 with today's 1m candles

then logged `bootstrapped` and started streaming as though healthy. Every
broker fetch had failed and the exception handler swallowed it at debug level.
The engine ran blind — indicators need 21 bars, so no signal could fire for
~21 minutes, with nothing in the logs saying why.

We now have a local Candle table fed by the tick ingestor, which already holds
today's 1m bars for the whole universe. Seeding from it is faster than the
broker, costs no rate limit, and survives a restart.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from django.utils import timezone

from apps.market_data.services.candle_ingestor import Bar, persist_bars
from apps.market_data.services.local_seed import (
    local_1m_candles,
    seeded_enough,
)

pytestmark = pytest.mark.django_db


def _write_bars(symbol="SBIN", n=30):
    base = timezone.localtime().replace(hour=9, minute=15, second=0, microsecond=0)
    persist_bars([
        Bar(symbol=symbol, t=(base + timedelta(minutes=i)).replace(tzinfo=None),
            o=100.0 + i, h=101.0 + i, l=99.0 + i, c=100.5 + i, v=10)
        for i in range(n)
    ])


def test_local_seed_returns_todays_bars_in_broker_shape():
    _write_bars(n=5)

    rows = local_1m_candles("SBIN")

    assert len(rows) == 5
    # Angel candle shape: [ts, o, h, l, c, v] — what seed_from_candles expects.
    first = rows[0]
    assert len(first) == 6
    assert first[1] == 100.0 and first[4] == 100.5


def test_local_seed_is_ordered_oldest_first():
    """Indicator warm-up folds forward; reversed bars silently corrupt EMA/RSI."""
    _write_bars(n=10)

    rows = local_1m_candles("SBIN")

    assert rows == sorted(rows, key=lambda r: r[0])


def test_local_seed_is_empty_for_an_unknown_symbol():
    assert local_1m_candles("NOSUCHSYMBOL") == []


def test_seeded_enough_rejects_a_blind_bootstrap():
    """The check that would have caught 'Seeded 0/98' claiming success."""
    assert seeded_enough(seeded=0, total=98) is False
    assert seeded_enough(seeded=3, total=98) is False


def test_seeded_enough_accepts_a_healthy_bootstrap():
    assert seeded_enough(seeded=98, total=98) is True
    assert seeded_enough(seeded=80, total=98) is True


def test_seeded_enough_is_true_when_there_is_nothing_to_seed():
    """Pre-market, an empty universe is not a failure."""
    assert seeded_enough(seeded=0, total=0) is True
