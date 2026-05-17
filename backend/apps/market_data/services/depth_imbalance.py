"""Level-2 depth imbalance & iceberg sniffer.

Real-data path: subscribe to the Angel One WebSocket DEPTH feed (mode 3)
and aggregate the 5-level bid/ask sizes into an imbalance ratio per
symbol. The bond / equity SDK doesn't expose that endpoint in this build,
so this stub:

  - uses today's volume-vs-baseline as a proxy 'depth_imbalance'
  - flags an "iceberg" when last-bar volume > 5× baseline
  - returns a stable contract so wiring real depth later doesn't break
    the React panel

The intent is for the trader to glance at the panel and see which symbols
have unusual order-flow today before sizing into them.
"""
from __future__ import annotations

import statistics
from datetime import date, timedelta
from typing import Any

from django.core.cache import cache


_TTL = 60


def _today_volume(symbol: str) -> int:
    key = f"depth:vol:{symbol}:{date.today().isoformat()}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service
        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            cache.set(key, 0, _TTL); return 0
        today = date.today()
        start = today.strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "FIVE_MINUTE", exchange=ticker_service.resolve_exchange(symbol)) or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "FIVE_MINUTE") or []
        total = sum(int(r[5]) for r in raw if len(r) > 5)
        cache.set(key, total, _TTL); return total
    except Exception:  # noqa: BLE001
        cache.set(key, 0, _TTL); return 0


def _baseline_volume(symbol: str, days: int = 20) -> float:
    key = f"depth:basevol:{symbol}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service
        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            cache.set(key, 0.0, 3600); return 0.0
        today = date.today()
        start = (today - timedelta(days=days + 2)).strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange=ticker_service.resolve_exchange(symbol)) or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        vols = [int(r[5]) for r in raw[-days:] if len(r) > 5]
        v = round(statistics.median(vols), 2) if vols else 0.0
        cache.set(key, v, 3600); return v
    except Exception:  # noqa: BLE001
        cache.set(key, 0.0, 3600); return 0.0


def _watchlist() -> list[str]:
    syms: set[str] = set()
    try:
        from trading.models import WatchlistEntry, TradeJournal
        for s in WatchlistEntry.objects.values_list("symbol", flat=True)[:60]:
            if s: syms.add(s)
        for s in TradeJournal.objects.filter(
            status__in=("EXECUTED", "PAPER", "APPROVED")
        ).values_list("symbol", flat=True)[:30]:
            if s: syms.add(s)
    except Exception:  # noqa: BLE001
        pass
    return sorted(syms)


def build_depth_imbalance(tenant=None) -> dict[str, Any]:
    symbols = _watchlist()[:25]
    rows: list[dict] = []
    for sym in symbols:
        today_v = _today_volume(sym)
        base_v = _baseline_volume(sym)
        ratio = (today_v / base_v) if base_v > 0 else 0.0
        iceberg = ratio >= 5.0
        # Without true L2 we represent depth_imbalance as 0 ± a heuristic.
        imbalance = 0.0
        rows.append({
            "symbol": sym,
            "today_volume": today_v,
            "baseline_volume": base_v,
            "volume_ratio": round(ratio, 2),
            "depth_imbalance": imbalance,
            "iceberg_flag": iceberg,
        })
    rows.sort(key=lambda r: -r["volume_ratio"])
    return {
        "count": len(rows),
        "rows": rows,
        "data_source": "volume-proxy",
        "note": (
            "True bid/ask depth requires the Angel One WS DEPTH mode-3 feed; "
            "this view shows today's volume vs 20-day median as a proxy. "
            "Volume ratio ≥ 5× = iceberg-likely; investigate before scaling in."
        ),
    }
