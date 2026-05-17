"""Base-Quality scorer (depth · length · tightness · volume dry-up).

For a single equity symbol, identify the most recent base (the high to
its most recent meaningful pullback) and grade it 0-100:

  depth_pct      pullback depth (closer to 15-30% scores best)
  length_weeks   how many weeks the base has formed
  tightness_pct  std-dev of last 3 weekly closes / mean (smaller = better)
  volume_dryup   last 5-day avg volume / 50-day avg (lower = drier base)

Weighted sum produces a 0-100 score plus a pattern_tag:
  VCP_TIGHT      tightness < 2%, dryup < 0.6
  FLAT_BASE      depth 8-18%, length > 5w
  DEEP_BASE      depth > 30%
  RAW_RANGE      anything else
"""
from __future__ import annotations

import statistics
from datetime import date, timedelta
from typing import Any

from django.core.cache import cache

_TTL = 600    # base shape is daily-bar driven; recompute every 10 min


def _fetch_daily(symbol: str, days: int = 180) -> list[dict]:
    key = f"basequality:daily:{symbol}:{days}"
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
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange="NSE") or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        rows = [
            {"t": str(r[0]), "h": float(r[2]), "l": float(r[3]),
             "c": float(r[4]), "v": int(r[5]) if len(r) > 5 else 0}
            for r in raw if len(r) >= 5
        ]
        cache.set(key, rows, _TTL); return rows
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def _weekly_closes(daily: list[dict], weeks: int = 6) -> list[float]:
    """Pick the close of every 5th bar from the tail to approximate weekly closes."""
    if not daily:
        return []
    closes = [b["c"] for b in daily[-weeks * 5:]]
    return closes[::5]


def _score_one(symbol: str) -> dict[str, Any]:
    daily = _fetch_daily(symbol)
    if len(daily) < 40:
        return {
            "symbol": symbol, "score": 0, "pattern_tag": "NO_DATA",
            "depth_pct": 0.0, "length_weeks": 0,
            "tightness_pct": 0.0, "volume_dryup": 0.0,
            "note": "not enough daily bars",
        }

    highs = [b["h"] for b in daily]
    lows = [b["l"] for b in daily]
    closes = [b["c"] for b in daily]
    vols = [b["v"] for b in daily]

    # Pivot = highest high in last 60 bars
    pivot_idx = max(range(max(0, len(highs) - 60), len(highs)), key=lambda i: highs[i])
    pivot = highs[pivot_idx]
    bars_since_pivot = len(highs) - 1 - pivot_idx
    length_weeks = round(bars_since_pivot / 5.0, 1)

    # Depth = (pivot − lowest_low since pivot) / pivot
    post_pivot_lows = lows[pivot_idx:]
    depth_pct = round(((pivot - min(post_pivot_lows)) / pivot) * 100.0, 2) if pivot > 0 else 0.0

    # Tightness = stdev / mean of last 3 weekly closes
    wk = _weekly_closes(daily, weeks=3)
    if len(wk) >= 3 and statistics.mean(wk) > 0:
        tightness_pct = round((statistics.stdev(wk) / statistics.mean(wk)) * 100.0, 2)
    else:
        tightness_pct = 0.0

    # Dryup = last-5 vol mean / last-50 vol mean
    if len(vols) >= 50 and statistics.mean(vols[-50:]) > 0:
        dryup = round(statistics.mean(vols[-5:]) / statistics.mean(vols[-50:]), 2)
    else:
        dryup = 0.0

    # Score (each component 0-25)
    depth_score = max(0, 25 - abs(20 - depth_pct))             # peak at 20% depth
    length_score = min(25, length_weeks * 3.5)                  # 7w+ saturates
    tightness_score = max(0, 25 - tightness_pct * 6)            # <2% ≈ 13/25
    dryup_score = max(0, 25 - (dryup - 0.5) * 50) if dryup > 0 else 0
    score = round(max(0, min(100, depth_score + length_score + tightness_score + dryup_score)))

    if tightness_pct < 2.0 and dryup > 0 and dryup < 0.6 and length_weeks >= 3:
        tag = "VCP_TIGHT"
    elif 8.0 <= depth_pct <= 18.0 and length_weeks >= 5:
        tag = "FLAT_BASE"
    elif depth_pct > 30:
        tag = "DEEP_BASE"
    else:
        tag = "RAW_RANGE"

    return {
        "symbol": symbol,
        "score": int(score),
        "pattern_tag": tag,
        "pivot": round(pivot, 2),
        "depth_pct": depth_pct,
        "length_weeks": length_weeks,
        "tightness_pct": tightness_pct,
        "volume_dryup": dryup,
        "last_close": round(closes[-1], 2),
        "pct_from_pivot": round((closes[-1] - pivot) / pivot * 100.0, 2) if pivot > 0 else 0.0,
    }


def build_base_quality(symbols: list[str] | None = None) -> dict[str, Any]:
    """Score one or many symbols. Caller can pass an explicit list,
    otherwise pulls from the legacy watchlist."""
    if not symbols:
        try:
            from trading.models import WatchlistEntry
            symbols = [s for s in WatchlistEntry.objects.values_list("symbol", flat=True)[:40] if s]
        except Exception:  # noqa: BLE001
            symbols = []
    symbols = (symbols or [])[:30]
    rows = [_score_one(s) for s in symbols]
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {
        "count": len(rows),
        "rows": rows,
        "note": (
            "Score blends depth (peak at 20%), length (saturates 7w+), "
            "tightness (last-3-weekly-closes stdev), and volume dry-up "
            "(last-5 vs last-50). 80+ = high-quality VCP / flat base."
        ),
    }
