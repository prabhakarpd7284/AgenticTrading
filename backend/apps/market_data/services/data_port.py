"""Default MarketDataPort implementation.

Resolution order for both `ltp()` and `candles()`:
  1. Redis cache (`ltp:<SYMBOL>`) populated by the tick ingestor — instant.
  2. Local Candle table — works when the v2 ingestor has snapshotted.
  3. **Live broker call via the legacy Angel One SmartAPI** — the actual
     source of truth most of the time during development. The legacy
     BrokerClient is a singleton with built-in rate-limit throttling so
     this is safe to call per request.

This makes the v2 `GET /api/v1/market-data/candles/?symbol=NIFTY&interval=5m&n=200`
endpoint return real 5-min OHLCV that React lightweight-charts can render
directly — no run-result roundtrip needed.
"""
from __future__ import annotations

from datetime import date, timedelta

from apps.market_data.models import Candle, Symbol


# Frontend interval label → (Angel One name, minutes-per-bar, default lookback days)
_INTERVAL_MAP = {
    "1m":  ("ONE_MINUTE",     1,  2),
    "3m":  ("THREE_MINUTE",   3,  5),
    "5m":  ("FIVE_MINUTE",    5,  5),
    "10m": ("TEN_MINUTE",    10,  7),
    "15m": ("FIFTEEN_MINUTE",15, 10),
    "30m": ("THIRTY_MINUTE", 30, 15),
    "1h":  ("ONE_HOUR",      60, 30),
    "1d":  ("ONE_DAY",     1440, 90),
}


class DefaultMarketData:
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id

    def ltp(self, symbol: str) -> float:
        # 1. Redis LTP key
        from django.core.cache import cache
        v = cache.get(f"ltp:{symbol}")
        if v is not None:
            return float(v)
        # 2. Cached Candle table
        row = Candle.objects.filter(symbol__tradingsymbol=symbol).order_by("-t").first()
        if row:
            return float(row.c)
        # 3. Live broker call
        return _broker_ltp(symbol)

    def candles(self, symbol: str, interval: str, n: int) -> list[dict]:
        """Return n most-recent candles as JSON-safe dicts."""
        sym = Symbol.objects.filter(tradingsymbol=symbol).first()
        if sym:
            rows = list(
                Candle.objects.filter(symbol=sym, interval=interval)
                .order_by("-t")[:n]
                .values("t", "o", "h", "l", "c", "v")
            )[::-1]
            if rows:
                return rows
        # Live broker fallback — actually returns data even on a fresh DB.
        return _broker_candles(symbol, interval, n)

    def options_chain(self, underlying: str, expiry: str) -> dict:
        return {"underlying": underlying, "expiry": expiry, "strikes": []}


# ─────────────────────────────────────────────────────────────────────────
# Live broker helpers — wrap the legacy stack so we don't duplicate auth.
# ─────────────────────────────────────────────────────────────────────────
def _broker_ltp(symbol: str) -> float:
    try:
        if symbol in ("NIFTY", "BANKNIFTY", "SENSEX"):
            from trading.options.data_service import OptionsDataService
            ods = OptionsDataService()
            if symbol == "BANKNIFTY":
                return float(ods.fetch_banknifty_spot().get("ltp", 0) or 0)
            return float(ods.fetch_nifty_spot().get("ltp", 0) or 0)
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service
        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            return 0.0
        r = broker.ltp("NSE", symbol, token)
        return float(r.get("ltp", 0) if isinstance(r, dict) else 0)
    except Exception:  # noqa: BLE001
        return 0.0


def _broker_candles(symbol: str, interval: str, n: int) -> list[dict]:
    """Call Angel One via the legacy BrokerClient for `n` candles of `interval`.

    Normalises Angel One's [ts, o, h, l, c, v] rows into [{t, o, h, l, c, v}]
    dicts so the frontend chart code stays consistent regardless of source.
    """
    angel_interval, minutes, default_days = _INTERVAL_MAP.get(interval, ("FIVE_MINUTE", 5, 5))

    bars_per_day = max(1, int((6 * 60 + 15) / minutes)) if minutes < 1440 else 1
    days = max(default_days, int(n / bars_per_day) + 2)

    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service
        from trading.options.data_service import (
            NIFTY_SPOT_TOKEN, INDIA_VIX_TOKEN, BANKNIFTY_SPOT_TOKEN,
        )

        broker = BrokerClient.get_instance(); broker.ensure_login()
        today = date.today()
        start = (today - timedelta(days=days)).strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")

        if symbol == "NIFTY":
            token, exchange = NIFTY_SPOT_TOKEN, "NSE"
        elif symbol == "BANKNIFTY":
            token, exchange = BANKNIFTY_SPOT_TOKEN, "NSE"
        elif symbol == "INDIAVIX":
            token, exchange = INDIA_VIX_TOKEN, "NSE"
        else:
            token = ticker_service.get_token(symbol)
            exchange = "NSE"

        if not token:
            return []

        try:
            raw = broker.fetch_candles(token, start, end, angel_interval, exchange=exchange) or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, angel_interval) or []

        out: list[dict] = []
        for row in raw[-n:]:
            try:
                out.append({
                    "t": str(row[0]), "o": float(row[1]), "h": float(row[2]),
                    "l": float(row[3]), "c": float(row[4]),
                    "v": int(row[5]) if len(row) > 5 else 0,
                })
            except (IndexError, ValueError, TypeError):
                continue
        return out
    except Exception:  # noqa: BLE001
        return []
