"""Default MarketDataPort implementation — reads from local cache, falls back to broker."""
from __future__ import annotations

from apps.market_data.models import Candle, Symbol


class DefaultMarketData:
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id

    def ltp(self, symbol: str) -> float:
        # Real impl: Redis LTP key populated by the tick ingestor.
        from django.core.cache import cache
        v = cache.get(f"ltp:{symbol}")
        if v is not None:
            return float(v)
        row = Candle.objects.filter(symbol__tradingsymbol=symbol).order_by("-t").first()
        return float(row.c) if row else 0.0

    def candles(self, symbol: str, interval: str, n: int) -> list[dict]:
        sym = Symbol.objects.filter(tradingsymbol=symbol).first()
        if not sym:
            return []
        return list(
            Candle.objects.filter(symbol=sym, interval=interval)
            .order_by("-t")[:n]
            .values("t", "o", "h", "l", "c", "v")
        )[::-1]

    def options_chain(self, underlying: str, expiry: str) -> dict:
        # Placeholder — real impl queries broker / NSE
        return {"underlying": underlying, "expiry": expiry, "strikes": []}
