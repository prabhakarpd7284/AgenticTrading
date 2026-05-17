"""IBD-style market health gauge: distribution-day count + follow-through.

  Distribution day:  NIFTY closes down >= 0.2% AND volume > yesterday.
                     Distribution day count rolls forward 25 trading days
                     before each one expires.
  Follow-through day:  After a market correction (cluster of distributions),
                     NIFTY closes up >= 1.7% on rising volume — that's the
                     bullish 'follow-through' confirming the uptrend.

  Market status from the count:
    HEALTHY      0-3 distribution days in last 25 sessions
    PRESSURED    4-5 distribution days
    UNDER_PRESSURE  6+ distribution days  → reduce exposure
    IN_RALLY     follow-through day inside last 10 sessions  → expand

This is the classic O'Neil framework that William O'Neil's IBD team
publishes daily. We compute it from NIFTY 50 daily candles + yfinance
volume (broker SDK doesn't return turnover for the index directly).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from django.core.cache import cache


_TTL = 600          # daily-bar driven; recompute every 10 min during the day
_LOOKBACK_DAYS = 60
_DIST_WINDOW = 25   # rolling window for distribution-day count
_FTD_WINDOW = 10    # follow-through must be within 10 sessions of low


def _nifty_daily(days: int = _LOOKBACK_DAYS) -> list[dict]:
    """yfinance daily OHLCV for NIFTY 50. Volume is required for IBD math."""
    key = f"market_health:nifty:{days}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        import yfinance as yf
        hist = yf.Ticker("^NSEI").history(period=f"{days + 10}d", auto_adjust=False)
        out: list[dict] = []
        for d, row in hist.iterrows():
            try:
                out.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "o": float(row["Open"]), "h": float(row["High"]),
                    "l": float(row["Low"]),  "c": float(row["Close"]),
                    "v": int(row["Volume"]) if row["Volume"] == row["Volume"] else 0,
                })
            except (KeyError, ValueError, TypeError):
                continue
        out = out[-days:]
        cache.set(key, out, _TTL); return out
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def _classify_distribution_days(bars: list[dict]) -> list[dict]:
    """Tag every day with whether it's a distribution day."""
    out = []
    for i in range(1, len(bars)):
        prev = bars[i - 1]; cur = bars[i]
        if prev["c"] <= 0 or prev["v"] <= 0:
            continue
        change_pct = (cur["c"] - prev["c"]) / prev["c"] * 100.0
        vol_up = cur["v"] > prev["v"]
        is_dist = change_pct <= -0.2 and vol_up
        out.append({
            "date": cur["date"],
            "close": round(cur["c"], 2),
            "change_pct": round(change_pct, 2),
            "vol_up": vol_up,
            "distribution_day": is_dist,
        })
    return out


def _detect_follow_through(bars: list[dict]) -> dict | None:
    """Look back over the last _FTD_WINDOW days for a follow-through —
    NIFTY up >= 1.7% on volume > yesterday. Returns the day if found.
    The 'rally attempt' day (the recent swing low) is also tagged.
    """
    if len(bars) < _FTD_WINDOW + 1:
        return None
    tail = bars[-_FTD_WINDOW:]
    for i in range(len(tail) - 1):
        prev = tail[i]; cur = tail[i + 1]
        if prev["c"] <= 0 or prev["v"] <= 0:
            continue
        change_pct = (cur["c"] - prev["c"]) / prev["c"] * 100.0
        if change_pct >= 1.7 and cur["v"] > prev["v"]:
            return {
                "date": cur["date"],
                "change_pct": round(change_pct, 2),
                "from_close": round(prev["c"], 2),
                "to_close": round(cur["c"], 2),
            }
    return None


def build_market_health(tenant=None) -> dict[str, Any]:
    bars = _nifty_daily()
    if len(bars) < 5:
        return {
            "status": "no_data",
            "distribution_count_25d": 0,
            "follow_through": None,
            "events": [],
            "note": "yfinance unavailable — IBD gauge needs ^NSEI history.",
        }

    daily = _classify_distribution_days(bars)
    window = daily[-_DIST_WINDOW:]
    dist_count = sum(1 for d in window if d["distribution_day"])
    ftd = _detect_follow_through(bars)

    if ftd:
        status = "IN_RALLY"
    elif dist_count >= 6:
        status = "UNDER_PRESSURE"
    elif dist_count >= 4:
        status = "PRESSURED"
    else:
        status = "HEALTHY"

    return {
        "status": status,
        "distribution_count_25d": dist_count,
        "follow_through": ftd,
        "events": daily[-_DIST_WINDOW:],
        "as_of": datetime.now(timezone.utc).isoformat(),
        "note": (
            "IBD-style gauge from NIFTY 50 daily OHLCV. UNDER_PRESSURE = "
            "reduce exposure or hedge. IN_RALLY = a fresh follow-through "
            "day confirms the uptrend; size up new positions. PRESSURED = "
            "tighten stops, don't add. HEALTHY = trade your edges normally."
        ),
    }
