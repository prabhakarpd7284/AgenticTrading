"""Opening-Range Breakout tracker.

For every watchlist + open-position symbol, fetch today's 1-min candles
from 9:15 to 9:30 IST, compute the 15-min opening range high/low, then
walk the rest of the session to classify:

  state                  inside | breakout_up | breakout_down | failed_breakout
  breakout_time          first minute spot closes outside the OR
  retests                number of bars that closed back inside after a breakout
  or_width_atr           OR-width divided by ATR(14) on daily for context

Cached 60s — the OR itself never changes after 9:30, only state evolves.
"""
from __future__ import annotations

import statistics
from datetime import date, datetime, time, timedelta
from typing import Any

from django.core.cache import cache

_TTL = 60
_OR_END = time(9, 30)


def _watchlist_symbols() -> list[str]:
    syms: set[str] = set()
    try:
        from trading.models import WatchlistEntry, TradeJournal
        for s in WatchlistEntry.objects.values_list("symbol", flat=True)[:80]:
            if s:
                syms.add(s)
        for s in TradeJournal.objects.filter(
            status__in=("EXECUTED", "PAPER", "APPROVED")
        ).values_list("symbol", flat=True)[:80]:
            if s:
                syms.add(s)
    except Exception:  # noqa: BLE001
        pass
    return sorted(syms)


def _fetch_1m(symbol: str) -> list[dict]:
    """Today's 1-min candles. Caller deals with empty lists."""
    key = f"orb:1m:{symbol}:{date.today().isoformat()}"
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
        cache.set(key, out, _TTL)
        return out
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL)
        return []


def _parse_minute(ts: str) -> time | None:
    """Tolerate Angel One's slightly-variable timestamp formats."""
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(ts, fmt).time()
        except ValueError:
            continue
    return None


def _atr14_daily(symbol: str) -> float:
    """Average true range over last 14 daily bars; cached 10 min."""
    key = f"orb:atr:{symbol}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service

        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            cache.set(key, 0.0, 600); return 0.0
        today = date.today()
        start = (today - timedelta(days=25)).strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange="NSE") or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        rows = [(float(r[2]), float(r[3]), float(r[4])) for r in raw if len(r) >= 5]
        if len(rows) < 2:
            cache.set(key, 0.0, 600); return 0.0
        trs = []
        for i in range(1, len(rows)):
            h, l, _ = rows[i]
            pc = rows[i - 1][2]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        atr = round(statistics.mean(trs[-14:]), 2)
        cache.set(key, atr, 600)
        return atr
    except Exception:  # noqa: BLE001
        cache.set(key, 0.0, 600); return 0.0


def _classify(candles: list[dict]) -> dict:
    # split into OR window (≤09:30) vs rest
    or_bars, rest = [], []
    for c in candles:
        t = _parse_minute(c["t"])
        if t is None:
            continue
        (or_bars if t <= _OR_END else rest).append(c)

    if not or_bars:
        return {"or_high": 0.0, "or_low": 0.0, "or_width": 0.0,
                "state": "pre_open", "breakout_time": None, "retests": 0,
                "state_transitions": []}

    or_high = max(b["h"] for b in or_bars)
    or_low = min(b["l"] for b in or_bars)
    or_width = round(or_high - or_low, 2)

    state, breakout_time, broke_up = "inside", None, None
    retests = 0
    transitions: list[dict] = [{"t": rest[0]["t"] if rest else "", "from": "pre", "to": "inside"}]
    for b in rest:
        prev_state = state
        if state == "inside":
            if b["c"] > or_high:
                state, breakout_time, broke_up = "breakout_up", b["t"], True
            elif b["c"] < or_low:
                state, breakout_time, broke_up = "breakout_down", b["t"], False
        else:
            if broke_up and b["c"] <= or_high:
                retests += 1
                if b["c"] < or_low:
                    state = "failed_breakout"
            elif not broke_up and b["c"] >= or_low:
                retests += 1
                if b["c"] > or_high:
                    state = "failed_breakout"
        if state != prev_state:
            transitions.append({"t": b["t"], "from": prev_state, "to": state})

    return {
        "or_high": round(or_high, 2),
        "or_low": round(or_low, 2),
        "or_width": or_width,
        "state": state,
        "breakout_time": breakout_time,
        "retests": retests,
        "state_transitions": transitions,
    }


def build_orb(tenant=None) -> dict[str, Any]:
    symbols = _watchlist_symbols()[:30]
    rows: list[dict] = []
    for sym in symbols:
        c = _fetch_1m(sym)
        cl = _classify(c)
        atr = _atr14_daily(sym)
        cl["or_width_atr"] = round(cl["or_width"] / atr, 2) if atr > 0 else 0.0
        cl["atr14"] = atr
        cl["symbol"] = sym
        rows.append(cl)
    return {
        "count": len(rows),
        "rows": rows,
        "note": (
            "Opening range = high/low of 09:15-09:30 IST 1-min bars. "
            "Width / ATR > 0.6 = wide-open day (trend-friendly); < 0.3 = tight "
            "coil (mean-reversion edge). Retest count rising = breakout failing."
        ),
    }
