"""Gap-Fill Probability dashboard.

For each watchlist symbol with a meaningful overnight gap (|gap| > 0.3%
of prev close), surface:

  gap_pct           today's open vs prior close
  filled_today      bool — did intraday LTP touch the prior close?
  historical_fill_p empirical rate gaps of similar size fill same-day
                    over the last 90 trading days
  status            'filled' / 'open' / 'no_gap'

Gives the operator a quick "is this gap likely to fill?" read so they
can decide whether to fade or chase.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from django.core.cache import cache

_TTL = 300


def _watchlist() -> list[str]:
    try:
        from trading.models import WatchlistEntry
        return [s for s in WatchlistEntry.objects.values_list("symbol", flat=True)[:50] if s]
    except Exception:  # noqa: BLE001
        return []


def _daily(symbol: str, days: int = 100) -> list[dict]:
    key = f"gapfill:daily:{symbol}:{days}"
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
        start = (today - timedelta(days=days + 14)).strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange=ticker_service.resolve_exchange(symbol)) or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        rows = [
            {"o": float(r[1]), "h": float(r[2]), "l": float(r[3]), "c": float(r[4])}
            for r in raw if len(r) >= 5
        ]
        cache.set(key, rows, _TTL); return rows
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def _historical_fill_rate(daily: list[dict], gap_threshold: float = 0.3) -> float:
    """Of all past sessions with |gap| >= threshold, how many filled same day?"""
    if len(daily) < 5:
        return 0.0
    hits = 0
    total = 0
    for i in range(1, len(daily)):
        prev_c = daily[i - 1]["c"]
        if prev_c <= 0: continue
        o = daily[i]["o"]
        gap = (o - prev_c) / prev_c * 100.0
        if abs(gap) >= gap_threshold:
            total += 1
            if (gap > 0 and daily[i]["l"] <= prev_c) or (gap < 0 and daily[i]["h"] >= prev_c):
                hits += 1
    return round(hits / total, 2) if total > 0 else 0.0


def build_gap_fill(tenant=None) -> dict[str, Any]:
    symbols = _watchlist()[:30]
    rows: list[dict] = []
    for sym in symbols:
        daily = _daily(sym)
        if len(daily) < 2:
            rows.append({"symbol": sym, "status": "no_data", "gap_pct": 0.0,
                         "filled_today": False, "historical_fill_p": 0.0})
            continue

        # Today's bar is the LAST element if market open, else also last
        today_bar = daily[-1]
        prev_close = daily[-2]["c"] if len(daily) >= 2 else 0.0
        gap_pct = ((today_bar["o"] - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0

        if abs(gap_pct) < 0.3:
            status = "no_gap"; filled = False
        else:
            if gap_pct > 0:
                filled = today_bar["l"] <= prev_close
            else:
                filled = today_bar["h"] >= prev_close
            status = "filled" if filled else "open"

        rows.append({
            "symbol": sym,
            "prev_close": round(prev_close, 2),
            "open": round(today_bar["o"], 2),
            "high": round(today_bar["h"], 2),
            "low": round(today_bar["l"], 2),
            "close": round(today_bar["c"], 2),
            "gap_pct": round(gap_pct, 2),
            "status": status,
            "filled_today": filled,
            "historical_fill_p": _historical_fill_rate(daily[:-1]),  # exclude today
        })

    # Surface most actionable first — open gaps with high historical fill prob
    rows.sort(key=lambda r: (
        0 if r["status"] == "open" else 1 if r["status"] == "filled" else 2,
        -r.get("historical_fill_p", 0),
    ))
    return {
        "count": len(rows),
        "rows": rows,
        "note": (
            "An 'open' gap with historical fill > 60% is a high-conviction "
            "fade. A gap that ALREADY filled tells you the open's bias is "
            "done — switch to the new intraday trend."
        ),
    }
