"""Seed a strategy engine from the local candle store instead of the broker.

The screener's bootstrap fetched today's 1m history over REST, one symbol at a
time, 98 calls deep. On a mid-session restart that runs straight into Angel's
rate limit — and the failures were swallowed at debug level, so on 2026-09-10
a restart logged ``Seeded 0/98``, then ``bootstrapped``, then streamed as if
healthy. Indicators need 21 bars, so the engine produced no signal for ~20
minutes with nothing explaining why.

The tick ingestor now writes those same 1m bars to `market_data.Candle` as they
form. Reading them back is one query, costs no rate limit, and survives a
restart — which is exactly the case the broker path handled worst.
"""
from __future__ import annotations

import structlog
from django.utils import timezone

log = structlog.get_logger(__name__)

# Below this share of the universe, a bootstrap has not really happened. Set
# loose enough to tolerate genuinely illiquid symbols that have not printed.
MIN_SEED_RATIO = 0.5


def local_1m_candles(symbol: str, day=None) -> list[list]:
    """Today's stored 1m bars for ``symbol``, in Angel's candle shape.

    ``[timestamp, open, high, low, close, volume]``, oldest first — the order
    `seed_from_candles` folds forward. Reversed bars would silently corrupt
    every EMA and RSI derived from them.
    """
    try:
        from apps.market_data.models import Candle

        day = day or timezone.localtime().date()
        rows = (
            Candle.objects
            .filter(symbol__tradingsymbol=symbol, interval="1m", t__date=day)
            .order_by("t")
            .values_list("t", "o", "h", "l", "c", "v")
        )
        return [
            [t.isoformat(), float(o), float(h), float(l), float(c), int(v)]
            for t, o, h, l, c, v in rows
        ]
    except Exception:  # noqa: BLE001 — a seed miss must not stop the engine.
        log.warning("local_seed.read_failed", symbol=symbol, exc_info=True)
        return []


def seeded_enough(*, seeded: int, total: int) -> bool:
    """Did the bootstrap actually load history?

    Exists so a blind start fails loudly instead of logging success. An engine
    with no history is not merely slow to warm up — it silently emits nothing,
    which reads identically to "the market offered no setups".
    """
    if total <= 0:
        return True
    return (seeded / total) >= MIN_SEED_RATIO
