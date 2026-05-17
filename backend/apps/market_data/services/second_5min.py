"""Second-5-Minute (09:20-09:25 IST) continuation-vs-reversal classifier.

The 9:20 candle is the 'test' of the 9:15 candle. We compare them:

  continuation  : same direction as the first candle and breaks its high/low
  reversal      : opposite direction and breaks the first candle's open
  consolidation : inside the first candle's range
  weak          : same direction but doesn't break the first bar's high/low

The day_type_tag is the same taxonomy as first-5-min so a planner can chain
both into a single 'TREND_DAY confirmed' / 'TREND_DAY weakening' signal.
"""
from __future__ import annotations

from datetime import date, time, timedelta
from typing import Any

from django.core.cache import cache

from trading.utils.time_utils import intraday_session_date

_TTL = 60
_FIRST_END = time(9, 20)
_SECOND_END = time(9, 25)


def _watchlist() -> list[str]:
    try:
        from trading.models import WatchlistEntry
        return [s for s in WatchlistEntry.objects.values_list("symbol", flat=True)[:50] if s]
    except Exception:  # noqa: BLE001
        return []


def _fetch_first_two(symbol: str) -> list[dict]:
    key = f"second5:bars:{symbol}:{intraday_session_date().isoformat()}"
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
        today = intraday_session_date()
        start = today.strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 09:25")
        try:
            raw = broker.fetch_candles(token, start, end, "FIVE_MINUTE", exchange=ticker_service.resolve_exchange(symbol)) or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "FIVE_MINUTE") or []
        rows = [
            {"t": str(r[0]), "o": float(r[1]), "h": float(r[2]),
             "l": float(r[3]), "c": float(r[4]), "v": int(r[5]) if len(r) > 5 else 0}
            for r in raw if len(r) >= 5
        ][:2]
        cache.set(key, rows, _TTL); return rows
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def _classify(first: dict, second: dict) -> tuple[str, str]:
    f_dir = 1 if first["c"] > first["o"] else -1 if first["c"] < first["o"] else 0
    s_dir = 1 if second["c"] > second["o"] else -1 if second["c"] < second["o"] else 0

    # Continuation: same direction AND breaks the first bar's extreme
    if f_dir > 0 and s_dir > 0 and second["c"] > first["h"]:
        return "continuation", "TREND_CONFIRMED"
    if f_dir < 0 and s_dir < 0 and second["c"] < first["l"]:
        return "continuation", "TREND_CONFIRMED"
    # Reversal: opposite direction AND breaks the first bar's open
    if f_dir > 0 and s_dir < 0 and second["c"] < first["o"]:
        return "reversal", "FAIL_DAY"
    if f_dir < 0 and s_dir > 0 and second["c"] > first["o"]:
        return "reversal", "FAIL_DAY"
    # Consolidation: stays inside the first bar's range
    if first["l"] <= second["c"] <= first["h"] and first["l"] <= second["o"] <= first["h"]:
        return "consolidation", "COIL_DAY"
    return "weak", "WEAKENING"


def build_second_5min(tenant=None) -> dict[str, Any]:
    from apps.market_data.services._parallel import parallel_symbols
    symbols = _watchlist()[:30]

    def _row(sym: str) -> dict:
        bars = _fetch_first_two(sym)
        if len(bars) < 2:
            return {"symbol": sym, "classification": "no_data",
                    "day_type_tag": "UNKNOWN",
                    "first_bar": None, "second_bar": None,
                    "vol_ratio": 0.0}
        first, second = bars[0], bars[1]
        cls, tag = _classify(first, second)
        vol_ratio = (second["v"] / first["v"]) if first["v"] > 0 else 0.0
        return {
            "symbol": sym,
            "first_bar": {"o": first["o"], "h": first["h"], "l": first["l"], "c": first["c"], "v": first["v"]},
            "second_bar": {"o": second["o"], "h": second["h"], "l": second["l"], "c": second["c"], "v": second["v"]},
            "classification": cls,
            "day_type_tag": tag,
            "vol_ratio": round(vol_ratio, 2),
        }

    rows = parallel_symbols(symbols, _row)
    return {
        "count": len(rows),
        "rows": rows,
        "note": (
            "Compares 09:20-09:25 bar to 09:15-09:20. Continuation on rising "
            "vol = the strongest 'go' signal. Reversal = first 5-min was a "
            "trap; size the fade. Consolidation = wait for the 09:30 break."
        ),
    }
