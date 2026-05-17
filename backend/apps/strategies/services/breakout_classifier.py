"""Fresh-Breakout vs Extended Classifier.

For each watchlist symbol (or any list passed in), classify the current
position as `fresh` or `extended` so chasing climactic moves can be
blocked at @RiskGuard.

Metrics:
  pct_from_pivot   distance from the recent 60-day high
  pct_from_20dma   distance from the 20-day SMA
  base_depth_pct   most recent base depth (peak to trough since pivot)

Rules (cumulative, first that fires wins):
  base_too_shallow  base_depth < 5%   — no real launchpad
  fresh             |pct_from_pivot| < 3 AND pct_from_20dma <= 5
  extended          pct_from_20dma > 10  OR pct_from_pivot > 8
  consolidating     pct_from_pivot < -8  — pulled back, not fresh
  neutral           everything else
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from django.core.cache import cache

_TTL = 600


def _fetch_daily(symbol: str, days: int = 80) -> list[dict]:
    key = f"breakout:daily:{symbol}:{days}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service
        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            cache.set(key, [], _TTL); return []
        today = date.today()
        start = (today - timedelta(days=days + 7)).strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange="NSE") or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        rows = [
            {"h": float(r[2]), "l": float(r[3]), "c": float(r[4])}
            for r in raw if len(r) >= 5
        ]
        cache.set(key, rows, _TTL); return rows
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def _classify_one(symbol: str) -> dict:
    daily = _fetch_daily(symbol)
    if len(daily) < 25:
        return {"symbol": symbol, "state": "no_data",
                "pct_from_pivot": 0.0, "pct_from_20dma": 0.0, "base_depth_pct": 0.0}

    highs = [b["h"] for b in daily]
    lows = [b["l"] for b in daily]
    closes = [b["c"] for b in daily]
    close = closes[-1]
    pivot_idx = max(range(max(0, len(highs) - 60), len(highs)), key=lambda i: highs[i])
    pivot = highs[pivot_idx]
    post_pivot_lows = lows[pivot_idx:]
    base_depth = ((pivot - min(post_pivot_lows)) / pivot * 100.0) if pivot > 0 else 0.0

    sma20 = sum(closes[-20:]) / 20.0
    pct_from_pivot = (close - pivot) / pivot * 100.0 if pivot > 0 else 0.0
    pct_from_20dma = (close - sma20) / sma20 * 100.0 if sma20 > 0 else 0.0

    if base_depth < 5.0:
        state = "base_too_shallow"
    elif abs(pct_from_pivot) < 3.0 and pct_from_20dma <= 5.0:
        state = "fresh"
    elif pct_from_20dma > 10.0 or pct_from_pivot > 8.0:
        state = "extended"
    elif pct_from_pivot < -8.0:
        state = "consolidating"
    else:
        state = "neutral"

    return {
        "symbol": symbol,
        "close": round(close, 2),
        "pivot": round(pivot, 2),
        "sma20": round(sma20, 2),
        "pct_from_pivot": round(pct_from_pivot, 2),
        "pct_from_20dma": round(pct_from_20dma, 2),
        "base_depth_pct": round(base_depth, 2),
        "state": state,
    }


def _watchlist() -> list[str]:
    syms: set[str] = set()
    try:
        from trading.models import WatchlistEntry
        for s in WatchlistEntry.objects.values_list("symbol", flat=True)[:60]:
            if s: syms.add(s)
    except Exception:  # noqa: BLE001
        pass
    return sorted(syms)


def build_breakout_classifier(symbols: list[str] | None = None) -> dict[str, Any]:
    syms = (symbols or _watchlist())[:30]
    rows = [_classify_one(s) for s in syms]
    # Surface freshest first, extended last — that's the order the trader cares about.
    state_order = {"fresh": 0, "neutral": 1, "consolidating": 2, "base_too_shallow": 3, "extended": 4, "no_data": 5}
    rows.sort(key=lambda r: (state_order.get(r["state"], 6), -r.get("pct_from_pivot", 0)))
    return {
        "count": len(rows),
        "rows": rows,
        "note": (
            "fresh = within 3% of pivot AND ≤5% above 20DMA (the ideal "
            "buy-stop zone). extended = >10% above 20DMA or >8% past pivot "
            "— chasing here pays for someone else's exit."
        ),
    }
