"""First-5-Minute candle profile / day-type classifier.

At 9:20 IST each session, the 9:15–9:20 candle predicts the day. We
classify each watchlist symbol's opening candle into one of:

  wide_range_trend  body >= 70% of range AND gap_pct >= 0.3
  gap_and_go        gap_pct >= 0.5 AND closes near the open
  gap_and_fade      gap_pct >= 0.5 BUT closes back inside prev range
  inside_bar_coil   range < 0.4 × ATR(14)
  doji              body < 20% of range
  normal            everything else

Each row also carries a day_type_tag (TREND_DAY / RANGE_DAY / FADE_DAY /
COIL_DAY / UNKNOWN) the operator can grep against.
"""
from __future__ import annotations

import statistics
from datetime import date, datetime, time, timedelta
from typing import Any

from django.core.cache import cache

_TTL = 60
_FIRST_BAR_END = time(9, 20)


def _watchlist() -> list[str]:
    syms: set[str] = set()
    try:
        from trading.models import WatchlistEntry, TradeJournal
        for s in WatchlistEntry.objects.values_list("symbol", flat=True)[:80]:
            if s: syms.add(s)
        for s in TradeJournal.objects.filter(
            status__in=("EXECUTED", "PAPER", "APPROVED")
        ).values_list("symbol", flat=True)[:80]:
            if s: syms.add(s)
    except Exception:  # noqa: BLE001
        pass
    return sorted(syms)


def _parse_minute(ts: str) -> time | None:
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(ts, fmt).time()
        except ValueError:
            continue
    return None


def _fetch_first_5min(symbol: str) -> list[dict]:
    """09:15-09:20 IST 5-min bar(s) for today. Cached 60s."""
    key = f"first5:{symbol}:{date.today().isoformat()}"
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
        start = today.strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 09:20")
        try:
            raw = broker.fetch_candles(token, start, end, "FIVE_MINUTE", exchange="NSE") or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "FIVE_MINUTE") or []
        rows = [
            {"t": str(r[0]), "o": float(r[1]), "h": float(r[2]),
             "l": float(r[3]), "c": float(r[4]),
             "v": int(r[5]) if len(r) > 5 else 0}
            for r in raw if len(r) >= 5
        ]
        cache.set(key, rows, _TTL)
        return rows
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def _prev_close_and_atr(symbol: str) -> tuple[float, float, float]:
    """Returns (prev_close, atr14, prev_range_high). All daily-derived."""
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service
        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            return 0.0, 0.0, 0.0
        today = date.today()
        start = (today - timedelta(days=25)).strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange="NSE") or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        rows = [(float(r[2]), float(r[3]), float(r[4])) for r in raw if len(r) >= 5]
        if len(rows) < 2:
            return 0.0, 0.0, 0.0
        prev_close = rows[-2][2] if today.strftime("%Y-%m-%d") in str(raw[-1][0]) else rows[-1][2]
        prev_h = rows[-2][0] if today.strftime("%Y-%m-%d") in str(raw[-1][0]) else rows[-1][0]
        trs = []
        for i in range(1, len(rows)):
            h, l, _ = rows[i]; pc = rows[i - 1][2]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        atr = round(statistics.mean(trs[-14:]), 2)
        return prev_close, atr, prev_h
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0


def _classify(bar: dict, prev_close: float, atr: float, prev_h: float) -> tuple[str, str]:
    rng = bar["h"] - bar["l"]
    if rng <= 0:
        return ("doji", "UNKNOWN")
    body = abs(bar["c"] - bar["o"])
    body_pct = body / rng
    gap_pct = ((bar["o"] - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0

    if rng < 0.4 * atr and atr > 0:
        return ("inside_bar_coil", "COIL_DAY")
    if body_pct < 0.2:
        return ("doji", "RANGE_DAY")
    if abs(gap_pct) >= 0.5:
        # Did the bar close near the open (continuation) or back inside prev range (fade)?
        if prev_h > 0 and ((gap_pct > 0 and bar["c"] < prev_h) or (gap_pct < 0 and bar["c"] > prev_h)):
            return ("gap_and_fade", "FADE_DAY")
        return ("gap_and_go", "TREND_DAY")
    if body_pct >= 0.7 and abs(gap_pct) >= 0.3:
        return ("wide_range_trend", "TREND_DAY")
    return ("normal", "RANGE_DAY")


def _fetch_1030_bar(symbol: str) -> dict | None:
    """Return the 10:25-10:30 5-min bar, used by the market-profile refinement.
    Cached 60s, single API call per symbol."""
    key = f"first5:1030:{symbol}:{date.today().isoformat()}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service
        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            cache.set(key, None, _TTL); return None
        today = date.today()
        start = today.strftime("%Y-%m-%d 10:25")
        end = today.strftime("%Y-%m-%d 10:30")
        try:
            raw = broker.fetch_candles(token, start, end, "FIVE_MINUTE", exchange="NSE") or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "FIVE_MINUTE") or []
        bar = None
        for r in raw:
            if len(r) >= 5:
                bar = {"t": str(r[0]), "o": float(r[1]), "h": float(r[2]),
                       "l": float(r[3]), "c": float(r[4]),
                       "v": int(r[5]) if len(r) > 5 else 0}
                break
        cache.set(key, bar, _TTL); return bar
    except Exception:  # noqa: BLE001
        cache.set(key, None, _TTL); return None


def _market_profile_refinement(first_bar: dict, bar_1030: dict | None) -> dict:
    """At 10:30, evaluate whether the first-bar classification still holds.

    NORMAL_DAY     — price stays within 1× first-bar range
    DOUBLE_DIST    — second distribution forming above/below first-bar
    TREND_DAY      — price has cleanly broken away (> 1.5× first-bar range)
    ROTATION       — bouncing between extremes (chop)
    """
    if not bar_1030:
        return {"refined_at": "10:30", "label": "PENDING",
                "displacement_x": 0.0, "note": "no 10:30 bar yet"}
    first_range = max(first_bar["h"] - first_bar["l"], 1e-9)
    mid_first = (first_bar["h"] + first_bar["l"]) / 2
    displacement = (bar_1030["c"] - mid_first) / first_range  # in first-bar-ranges
    if abs(displacement) > 1.5:
        label = "TREND_DAY"
    elif abs(displacement) > 0.5:
        label = "DOUBLE_DIST"
    elif bar_1030["h"] > first_bar["h"] and bar_1030["l"] < first_bar["l"]:
        label = "ROTATION"
    else:
        label = "NORMAL_DAY"
    return {
        "refined_at": "10:30",
        "label": label,
        "displacement_x": round(displacement, 2),
        "note": ("TREND_DAY = ride continuation. DOUBLE_DIST = new value "
                  "developing, trade the second range. ROTATION = scalp inside "
                  "first-bar extremes. NORMAL_DAY = original first-5-min plan still valid."),
    }


def build_first_5min(tenant=None) -> dict[str, Any]:
    symbols = _watchlist()[:30]
    rows: list[dict] = []
    for sym in symbols:
        bars = _fetch_first_5min(sym)
        if not bars:
            rows.append({
                "symbol": sym, "classification": "no_data", "day_type_tag": "UNKNOWN",
                "gap_pct": 0.0, "body_pct": 0.0, "vol": 0,
                "refinement_1030": {"refined_at": "10:30", "label": "PENDING", "displacement_x": 0.0},
            })
            continue
        bar = bars[0]
        prev_close, atr, prev_h = _prev_close_and_atr(sym)
        classification, tag = _classify(bar, prev_close, atr, prev_h)
        rng = max(bar["h"] - bar["l"], 1e-9)
        bar_1030 = _fetch_1030_bar(sym)
        rows.append({
            "symbol": sym,
            "open": bar["o"], "high": bar["h"], "low": bar["l"], "close": bar["c"],
            "vol": bar["v"],
            "gap_pct": round(((bar["o"] - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0, 2),
            "body_pct": round(abs(bar["c"] - bar["o"]) / rng * 100.0, 1),
            "range_atr": round((bar["h"] - bar["l"]) / atr, 2) if atr > 0 else 0.0,
            "classification": classification,
            "day_type_tag": tag,
            "refinement_1030": _market_profile_refinement(bar, bar_1030),
        })
    return {
        "count": len(rows),
        "rows": rows,
        "note": (
            "Classifies the 09:15-09:20 IST 5-min bar per symbol + refines at "
            "10:30 with a market-profile read (NORMAL_DAY / DOUBLE_DIST / "
            "TREND_DAY / ROTATION). The 10:30 refinement supersedes the 9:20 "
            "first read when displacement_x > 1."
        ),
    }
