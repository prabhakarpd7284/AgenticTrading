"""Default MarketDataPort implementation — reads from local cache, falls back to broker.

The local Candle table is populated by a tick ingestor that's not running
in dev yet, so until that lands the `candles()` and `ltp()` calls go
straight to the broker when the cache is cold. This is what makes
`/setup/<symbol>` and friends actually render numbers in dev without
manually pre-warming the DB.
"""
from __future__ import annotations

import logging

from apps.market_data.models import Candle, Symbol

log = logging.getLogger(__name__)

# Angel One interval name for each `data_port` interval string. Anything
# not mapped falls back to FIVE_MINUTE — same default the legacy services use.
_BROKER_INTERVAL = {
    "1m":  "ONE_MINUTE",
    "3m":  "THREE_MINUTE",
    "5m":  "FIVE_MINUTE",
    "10m": "TEN_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h":  "ONE_HOUR",
    "1d":  "ONE_DAY",
}

_INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 60, "1d": 1440,
}


class DefaultMarketData:
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id

    def ltp(self, symbol: str) -> float:
        # 1. Redis LTP key populated by the tick ingestor (when running).
        from django.core.cache import cache
        v = cache.get(f"ltp:{symbol}")
        if v is not None:
            return float(v)
        # 2. Most-recent local Candle close.
        row = Candle.objects.filter(symbol__tradingsymbol=symbol).order_by("-t").first()
        if row:
            return float(row.c)
        # 3. Broker — single-symbol LTP via BrokerClient cache.
        try:
            from trading.services.data_service import BrokerClient
            from trading.services.ticker_service import ticker_service
            token = ticker_service.get_token(symbol)
            if not token:
                return 0.0
            broker = BrokerClient.get_instance()
            broker.ensure_login()
            data = broker.ltp("NSE", symbol, token)
            return float(data.get("ltp") or 0.0)
        except Exception as exc:  # noqa: BLE001 — non-blocking
            log.warning("ltp broker fallback failed for %s: %s", symbol, exc)
            return 0.0

    def candles(self, symbol: str, interval: str, n: int) -> list[dict]:
        # 1. Local Candle cache.
        sym = Symbol.objects.filter(tradingsymbol=symbol).first()
        if sym:
            cached = list(
                Candle.objects.filter(symbol=sym, interval=interval)
                .order_by("-t")[:n]
                .values("t", "o", "h", "l", "c", "v")
            )[::-1]
            if cached:
                return cached
        # 2. Broker fallback — fetches a window wide enough to cover ~n bars.
        try:
            from datetime import datetime, timedelta
            from trading.services.data_service import BrokerClient
            from trading.services.ticker_service import ticker_service
            from trading.utils.time_utils import last_trading_day

            token = ticker_service.get_token(symbol)
            if not token:
                return []

            broker_interval = _BROKER_INTERVAL.get(interval, "FIVE_MINUTE")
            minutes = max(n, 1) * _INTERVAL_MINUTES.get(interval, 5)
            # 30% headroom — broker sometimes returns fewer bars than the
            # wall-clock window suggests (holidays, lunch break, etc.).
            window = timedelta(minutes=int(minutes * 1.3))

            # If today is a trading day and market is mid-session use "now"
            # as the end; otherwise rewind to the last completed session close.
            now = datetime.now()
            open_t  = datetime.strptime("09:15", "%H:%M").time()
            close_t = datetime.strptime("15:30", "%H:%M").time()
            if now.time() < open_t or now.time() > close_t or now.weekday() >= 5:
                end_date = last_trading_day(now)
                end = datetime.combine(end_date, close_t)
            else:
                end = now
            start = end - window

            broker = BrokerClient.get_instance()
            broker.ensure_login()
            raw = broker.fetch_candles(
                token,
                start.strftime("%Y-%m-%d %H:%M"),
                end.strftime("%Y-%m-%d %H:%M"),
                broker_interval,
            ) or []
            # raw rows: [ts_iso, open, high, low, close, volume]; reshape to
            # the dict shape data_port consumers expect (matches the cache).
            return [
                {"t": r[0], "o": r[1], "h": r[2], "l": r[3], "c": r[4],
                 "v": r[5] if len(r) > 5 else 0}
                for r in raw[-n:]
            ]
        except Exception as exc:  # noqa: BLE001 — non-blocking; setup degrades gracefully
            log.warning("candles broker fallback failed for %s: %s", symbol, exc)
            return []

    def options_chain(self, underlying: str, expiry: str) -> dict:
        # Placeholder — real impl queries broker / NSE
        return {"underlying": underlying, "expiry": expiry, "strikes": []}
