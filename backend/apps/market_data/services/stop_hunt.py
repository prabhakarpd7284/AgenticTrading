"""Intraday Stop-Hunt & Liquidity-Sweep Detector.

Scans the watchlist looking for bars that:

  - take out the previous-day high (PDH) or low (PDL)
  - take out the opening-range high (ORH) or low (ORL)
  - touch a round-number level (multiples of 100 for NIFTY, 10 for cheap stocks)

then immediately reverse (next bar closes back inside the swept level).
That's the classic stop-hunt: liquidity is taken out, then the move dies
and reverses — fade it.

Returns per-symbol detected events with the level swept + the reversal
strength so the trader can rank the cleanest fades.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from django.core.cache import cache


_TTL = 60


def _today_1m(symbol: str) -> list[dict]:
    key = f"sweep:1m:{symbol}:{date.today().isoformat()}"
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
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_MINUTE", exchange="NSE") or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_MINUTE") or []
        out = [
            {"t": str(r[0]), "o": float(r[1]), "h": float(r[2]),
             "l": float(r[3]), "c": float(r[4])}
            for r in raw if len(r) >= 5
        ]
        cache.set(key, out, _TTL); return out
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def _prev_day_hi_lo(symbol: str) -> tuple[float, float]:
    """Returns (PDH, PDL) — yesterday's daily high/low."""
    key = f"sweep:pdhl:{symbol}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service
        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            cache.set(key, (0.0, 0.0), 3600); return (0.0, 0.0)
        today = date.today()
        start = (today - timedelta(days=5)).strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange="NSE") or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        if len(raw) < 2:
            cache.set(key, (0.0, 0.0), 3600); return (0.0, 0.0)
        prev = raw[-2] if str(raw[-1][0])[:10] == today.isoformat() else raw[-1]
        out = (float(prev[2]), float(prev[3]))
        cache.set(key, out, 3600); return out
    except Exception:  # noqa: BLE001
        cache.set(key, (0.0, 0.0), 3600); return (0.0, 0.0)


def _round_levels_near(price: float) -> list[float]:
    """Return round levels near `price`. NIFTY-style hundreds for prices > 1000,
    tens for cheap stocks."""
    if price <= 0:
        return []
    step = 100 if price > 1000 else 10 if price > 100 else 1
    base = int(price // step) * step
    return [float(base - step), float(base), float(base + step), float(base + 2 * step)]


def _opening_range_hi_lo(bars: list[dict]) -> tuple[float, float]:
    """First 15-min OR — bars between 09:15 and 09:30."""
    ohi, olo = 0.0, float("inf")
    for b in bars[:15]:
        if b["h"] > ohi: ohi = b["h"]
        if b["l"] < olo: olo = b["l"]
    return ohi, 0.0 if olo == float("inf") else olo


def _detect_sweeps(bars: list[dict], levels: list[tuple[str, float]]) -> list[dict]:
    """For each bar, check if any level was swept (wick beyond level) AND
    the next bar closes back on the original side → confirmed reversal."""
    events: list[dict] = []
    if len(bars) < 2:
        return events
    for i in range(len(bars) - 1):
        bar = bars[i]; nxt = bars[i + 1]
        for name, lvl in levels:
            if lvl <= 0: continue
            # Up-sweep: bar high pierces above lvl, next bar closes back below
            if bar["h"] > lvl and bar["c"] > lvl and nxt["c"] < lvl:
                events.append({
                    "t": bar["t"], "level": name, "level_price": round(lvl, 2),
                    "direction": "up_sweep", "wick_pct": round((bar["h"] - lvl) / lvl * 100, 2),
                    "reversal_strength_pct": round((nxt["c"] - bar["h"]) / bar["h"] * 100, 2),
                })
            # Down-sweep: bar low pierces below lvl, next bar closes back above
            if bar["l"] < lvl and bar["c"] < lvl and nxt["c"] > lvl:
                events.append({
                    "t": bar["t"], "level": name, "level_price": round(lvl, 2),
                    "direction": "down_sweep", "wick_pct": round((lvl - bar["l"]) / lvl * 100, 2),
                    "reversal_strength_pct": round((nxt["c"] - bar["l"]) / bar["l"] * 100, 2),
                })
    return events


def _watchlist() -> list[str]:
    syms: set[str] = set()
    try:
        from trading.models import WatchlistEntry, TradeJournal
        for s in WatchlistEntry.objects.values_list("symbol", flat=True)[:50]:
            if s: syms.add(s.upper())
        for s in TradeJournal.objects.filter(
            status__in=("EXECUTED", "PAPER", "APPROVED")
        ).values_list("symbol", flat=True)[:30]:
            if s: syms.add(s.upper())
    except Exception:  # noqa: BLE001
        pass
    return sorted(syms)


def build_stop_hunt(tenant=None) -> dict[str, Any]:
    symbols = _watchlist()[:25]
    rows: list[dict] = []
    for sym in symbols:
        bars = _today_1m(sym)
        if not bars:
            rows.append({"symbol": sym, "events": [], "event_count": 0, "note": "no_bars"})
            continue
        pdh, pdl = _prev_day_hi_lo(sym)
        orh, orl = _opening_range_hi_lo(bars)
        last_close = bars[-1]["c"]
        round_lvls = _round_levels_near(last_close)

        levels: list[tuple[str, float]] = [
            ("PDH", pdh), ("PDL", pdl), ("ORH", orh), ("ORL", orl),
        ] + [(f"ROUND_{int(x)}", x) for x in round_lvls]
        events = _detect_sweeps(bars, levels)
        # Sort strongest reversals first
        events.sort(key=lambda e: -abs(e["reversal_strength_pct"]))
        rows.append({
            "symbol": sym,
            "pdh": round(pdh, 2), "pdl": round(pdl, 2),
            "orh": round(orh, 2), "orl": round(orl, 2),
            "last_close": round(last_close, 2),
            "events": events[:5],   # top-5 cleanest reversals per symbol
            "event_count": len(events),
        })

    rows.sort(key=lambda r: -r["event_count"])
    total_events = sum(r["event_count"] for r in rows)
    return {
        "count": len(rows),
        "rows": rows,
        "total_events": total_events,
        "note": (
            "Detects when a bar wicks past a key level (PDH/PDL/ORH/ORL/round "
            "number) AND the next bar closes back inside — classic stop-hunt "
            "fade setup. reversal_strength_pct ranks the cleanest setups. "
            "Trade against the sweep direction with the swept level as stop."
        ),
    }
