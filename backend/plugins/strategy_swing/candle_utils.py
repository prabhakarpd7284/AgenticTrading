"""Candle aggregation utilities used by the OK swing scanner / backtest.

Originally lived in `dashboard_utils.candle_cache` (Streamlit era). When
the dashboard was dropped, the swing module's two consumers were left
pointing at a deleted file — this restores the single helper they need.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable


def _parse_ts(raw) -> date:
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str):
        # Tolerate ISO date or full datetime strings.
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    raise ValueError(f"unrecognised timestamp: {raw!r}")


def _aggregate_to_weekly(daily: Iterable[dict]) -> list[dict]:
    """Roll a list of daily candles into weekly OHLCV bars.

    Expects each row to have ``timestamp`` (ISO string, date, or datetime),
    ``open``, ``high``, ``low``, ``close``, ``volume``. Weeks are bucketed
    by ISO week (Mon–Sun). Output bars are emitted in chronological order
    and timestamped at the bucket's Monday.
    """
    buckets: dict[tuple[int, int], dict] = {}

    for row in daily:
        d = _parse_ts(row["timestamp"])
        iso_year, iso_week, _ = d.isocalendar()
        key = (iso_year, iso_week)
        bar = buckets.get(key)
        if bar is None:
            buckets[key] = {
                "timestamp": d.isoformat(),
                "_monday": d - __import__("datetime").timedelta(days=d.weekday()),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row.get("volume", 0),
            }
        else:
            bar["high"] = max(bar["high"], row["high"])
            bar["low"] = min(bar["low"], row["low"])
            bar["close"] = row["close"]
            bar["volume"] += row.get("volume", 0)

    out: list[dict] = []
    for (_iy, _iw), bar in sorted(buckets.items()):
        bar["timestamp"] = bar.pop("_monday").isoformat()
        out.append(bar)
    return out
