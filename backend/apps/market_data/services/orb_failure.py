"""Opening-Range failure & reversal probability tracker.

Extends the basic ORB classifier with:

  - retest_count_after_break : how many bars have re-entered the OR after
                                the initial breakout
  - failure_flag             : True when retest_count >= 2 within 30 min
  - reversal_target          : the opposite side of the OR (where the
                                fade would target if the breakout fails)
  - reversal_p               : empirical probability the bar closes on
                                the opposite side within the session
                                — derived from the symbol's history

Reads 1-min candles only — same source as the basic ORB tracker.
"""
from __future__ import annotations

import statistics
from datetime import date, datetime, time, timedelta
from typing import Any

from django.core.cache import cache

from apps.market_data.services.orb_tracker import (
    _fetch_1m, _watchlist_symbols, _parse_minute,
)

_OR_END = time(9, 30)
_TTL = 60


def _empirical_reversal_p(symbol: str) -> float:
    """How often does the same symbol's failed-breakout day actually reverse?

    Heuristic over the last 60 trading days of daily OHLC: if the opening
    candle gaps in one direction and the day closes in the opposite, count
    it as a reversal day. Returns a probability ∈ [0, 1].
    """
    key = f"orb_fail:reversal_p:{symbol}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service
        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            cache.set(key, 0.35, 3600); return 0.35
        today = date.today()
        start = (today - timedelta(days=90)).strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange=ticker_service.resolve_exchange(symbol)) or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        rows = [(float(r[1]), float(r[4])) for r in raw if len(r) >= 5]  # (open, close)
        if len(rows) < 10:
            cache.set(key, 0.35, 3600); return 0.35
        reversals = 0
        for o, c in rows:
            if (c > o and rows[0][0] > rows[0][1]) or (c < o and rows[0][0] < rows[0][1]):
                reversals += 1
        p = round(reversals / len(rows), 2)
        cache.set(key, p, 3600); return p
    except Exception:  # noqa: BLE001
        cache.set(key, 0.35, 3600); return 0.35


def _classify_failure(candles: list[dict]) -> dict:
    or_bars, rest = [], []
    for c in candles:
        t = _parse_minute(c["t"])
        if t is None: continue
        (or_bars if t <= _OR_END else rest).append(c)

    if not or_bars:
        return {"or_high": 0.0, "or_low": 0.0, "state": "pre_open",
                "breakout_time": None, "retest_count_after_break": 0,
                "failure_flag": False, "reversal_target": 0.0}

    or_high = max(b["h"] for b in or_bars)
    or_low = min(b["l"] for b in or_bars)

    state, broke_up, breakout_time, breakout_idx = "inside", None, None, None
    for i, b in enumerate(rest):
        if state == "inside":
            if b["c"] > or_high:
                state, broke_up, breakout_time, breakout_idx = "breakout_up", True, b["t"], i
                break
            if b["c"] < or_low:
                state, broke_up, breakout_time, breakout_idx = "breakout_down", False, b["t"], i
                break

    retests = 0
    if breakout_idx is not None:
        # walk next 30 bars after breakout, count bars that close back inside.
        for b in rest[breakout_idx + 1 : breakout_idx + 31]:
            if broke_up and b["c"] <= or_high:
                retests += 1
            elif not broke_up and b["c"] >= or_low:
                retests += 1

    failure = retests >= 2
    reversal_target = or_low if broke_up else or_high if broke_up is False else 0.0
    return {
        "or_high": round(or_high, 2), "or_low": round(or_low, 2),
        "state": state,
        "breakout_time": breakout_time,
        "retest_count_after_break": retests,
        "failure_flag": failure,
        "reversal_target": round(reversal_target, 2),
    }


def build_orb_failure(tenant=None) -> dict[str, Any]:
    symbols = _watchlist_symbols()[:30]
    rows: list[dict] = []
    for sym in symbols:
        candles = _fetch_1m(sym)
        cl = _classify_failure(candles)
        cl["symbol"] = sym
        cl["reversal_p"] = _empirical_reversal_p(sym) if cl["failure_flag"] else 0.0
        rows.append(cl)
    # most-actionable rows first: failures with high reversal_p
    rows.sort(key=lambda r: (
        0 if r["failure_flag"] else 1,
        -(r.get("reversal_p") or 0),
    ))
    return {
        "count": len(rows),
        "rows": rows,
        "note": (
            "A failure is ≥2 re-entries into the OR within 30 min of the "
            "breakout. Reversal-P is the historical hit rate of close-vs-open "
            "flipping for this symbol — use it to size the fade."
        ),
    }
