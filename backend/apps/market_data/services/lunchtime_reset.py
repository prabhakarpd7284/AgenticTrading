"""Lunchtime Chop Filter + 13:15 PM Afternoon-Session Reset Briefing.

Indian intraday flow has a distinct rhythm:
  09:15-11:00  morning trend; high volume
  11:00-13:00  lunchtime chop; volume collapses, mean-reversion edge
  13:15-15:00  afternoon trend; HFT + institutional positioning
  15:00-15:30  closing-auction drift

At 13:15 the operator wants a "reset briefing" — what the morning did,
whether to fade or follow, key levels for the afternoon, and a CHOP
filter that suppresses signals 11:00-13:00.

  GET /market-data/lunchtime-reset/?underlying=NIFTY  → {
    morning_summary: {open, high, low, last, change_pct, type},
    chop_window: { active, until_ist },
    afternoon_briefing: { fade_or_follow, key_levels, plan },
    note: "..."
  }
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from django.core.cache import cache

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    _IST = timezone(timedelta(hours=5, minutes=30))


_TTL = 60
_LUNCH_START = time(11, 0)
_LUNCH_END = time(13, 15)


def _fetch_today_5m(symbol: str) -> list[dict]:
    key = f"lunch:5m:{symbol}:{date.today().isoformat()}"
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
            raw = broker.fetch_candles(token, start, end, "FIVE_MINUTE", exchange="NSE") or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "FIVE_MINUTE") or []
        out = [
            {"t": str(r[0]), "o": float(r[1]), "h": float(r[2]),
             "l": float(r[3]), "c": float(r[4])}
            for r in raw if len(r) >= 5
        ]
        cache.set(key, out, _TTL); return out
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def _parse_minute(ts: str) -> time | None:
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(ts, fmt).time()
        except ValueError:
            continue
    return None


def build_lunchtime_reset(tenant=None, *, underlying: str = "NIFTY") -> dict[str, Any]:
    sym = (underlying or "NIFTY").upper()
    bars = _fetch_today_5m(sym)
    now_ist = datetime.now(tz=_IST).time()

    chop_active = _LUNCH_START <= now_ist < _LUNCH_END
    chop_until = "13:15 IST" if chop_active else None

    # Morning summary — bars between 09:15 and 11:00
    morning_bars = []
    for b in bars:
        t = _parse_minute(b["t"])
        if t is None: continue
        if time(9, 15) <= t < _LUNCH_START:
            morning_bars.append(b)
    if morning_bars:
        m_open = morning_bars[0]["o"]
        m_high = max(b["h"] for b in morning_bars)
        m_low = min(b["l"] for b in morning_bars)
        m_last = morning_bars[-1]["c"]
        change_pct = round((m_last - m_open) / m_open * 100, 2) if m_open else 0.0
        if abs(change_pct) >= 0.5 and (m_last - m_open) * (m_high - m_low) > 0:
            morning_type = "TRENDED_UP" if change_pct > 0 else "TRENDED_DOWN"
        else:
            morning_type = "ROTATED"
    else:
        m_open = m_high = m_low = m_last = 0.0
        change_pct = 0.0; morning_type = "NO_DATA"

    # Afternoon briefing — opinion derived from morning type
    if morning_type == "TRENDED_UP":
        plan = "Follow the trend — buy pullbacks to 13:15 swing low or VWAP."
        fade_or_follow = "FOLLOW"
        levels = [m_high, m_low, round((m_high + m_low) / 2, 2)]
    elif morning_type == "TRENDED_DOWN":
        plan = "Follow the trend — short rallies to VWAP or 13:15 swing high."
        fade_or_follow = "FOLLOW"
        levels = [m_low, m_high, round((m_high + m_low) / 2, 2)]
    elif morning_type == "ROTATED":
        plan = "Fade the extremes — sell near morning high, buy near morning low. Skip the middle."
        fade_or_follow = "FADE"
        levels = [m_high, m_low]
    else:
        plan = "Wait for the 13:15 break of morning extremes before sizing in."
        fade_or_follow = "WAIT"
        levels = []

    return {
        "underlying": sym,
        "morning_summary": {
            "open": round(m_open, 2), "high": round(m_high, 2),
            "low": round(m_low, 2),  "last": round(m_last, 2),
            "change_pct": change_pct, "type": morning_type,
        },
        "chop_window": {
            "active": chop_active,
            "starts": "11:00 IST", "ends": "13:15 IST",
            "until_ist": chop_until,
        },
        "afternoon_briefing": {
            "fade_or_follow": fade_or_follow,
            "key_levels": levels,
            "plan": plan,
        },
        "note": (
            "11:00-13:15 = chop window. RiskGuard / planner should suppress "
            "breakout signals during this window. At 13:15 the afternoon "
            "briefing tells you whether to fade or follow based on morning type."
        ),
    }
