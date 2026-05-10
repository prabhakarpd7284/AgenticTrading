"""
Candle data caching layer for the dashboard.

Fetches 7 days of intraday candles + 90 days of daily candles in minimal
API calls, caches in st.session_state, and serves sliced data for any
timeframe (5m intraday, daily, weekly, monthly).

Handles after-hours / weekends automatically via time_utils.
"""
import streamlit as st
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from logzero import logger


# Cache TTL: 5 minutes for intraday, 1 hour for daily
INTRADAY_TTL = 3000
DAILY_TTL = 36000

# Session state key prefix
_KEY = "candle_cache"


def _cache_key(symbol: str) -> str:
    return f"{_KEY}_{symbol}"


def _is_stale(cached: dict, ttl: int) -> bool:
    ts = cached.get("fetched_at")
    if not ts:
        return True
    return (datetime.now() - ts).total_seconds() > ttl


def _parse_raw_candles(raw: list) -> list:
    """Convert Angel One raw candle rows to dicts with timestamp."""
    candles = []
    for r in raw:
        ts = r[0] if isinstance(r[0], str) else str(r[0])
        candles.append({
            "timestamp": ts,
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": int(r[5]),
        })
    return candles


def _aggregate_to_daily(intraday_by_date: Dict[str, list]) -> list:
    """Aggregate intraday candles into daily OHLCV bars."""
    daily = []
    for day_str in sorted(intraday_by_date.keys()):
        candles = intraday_by_date[day_str]
        if not candles:
            continue
        daily.append({
            "timestamp": f"{day_str}T00:00:00",
            "open": candles[0]["open"],
            "high": max(c["high"] for c in candles),
            "low": min(c["low"] for c in candles),
            "close": candles[-1]["close"],
            "volume": sum(c["volume"] for c in candles),
        })
    return daily


def _aggregate_to_weekly(daily_candles: list) -> list:
    """Aggregate daily candles into weekly OHLCV bars (Mon-Fri)."""
    if not daily_candles:
        return []
    weeks = {}
    for c in daily_candles:
        d = datetime.fromisoformat(c["timestamp"].replace("+05:30", "")).date()
        # Week key = Monday of that week
        week_start = d - timedelta(days=d.weekday())
        key = week_start.isoformat()
        if key not in weeks:
            weeks[key] = []
        weeks[key].append(c)

    result = []
    for week_key in sorted(weeks.keys()):
        candles = weeks[week_key]
        result.append({
            "timestamp": f"{week_key}T00:00:00",
            "open": candles[0]["open"],
            "high": max(c["high"] for c in candles),
            "low": min(c["low"] for c in candles),
            "close": candles[-1]["close"],
            "volume": sum(c["volume"] for c in candles),
        })
    return result


def _aggregate_to_monthly(daily_candles: list) -> list:
    """Aggregate daily candles into monthly OHLCV bars."""
    if not daily_candles:
        return []
    months = {}
    for c in daily_candles:
        d = datetime.fromisoformat(c["timestamp"].replace("+05:30", "")).date()
        key = f"{d.year}-{d.month:02d}"
        if key not in months:
            months[key] = []
        months[key].append(c)

    result = []
    for month_key in sorted(months.keys()):
        candles = months[month_key]
        result.append({
            "timestamp": f"{month_key}-01T00:00:00",
            "open": candles[0]["open"],
            "high": max(c["high"] for c in candles),
            "low": min(c["low"] for c in candles),
            "close": candles[-1]["close"],
            "volume": sum(c["volume"] for c in candles),
        })
    return result


def fetch_and_cache_candles(symbol: str, force: bool = False) -> dict:
    """
    Fetch candle data for a symbol with smart caching.

    Makes 2 API calls:
    1. ONE_DAY candles for last 90 days (1 call)
    2. FIVE_MINUTE candles for today/last trading day (1 call)

    Returns cached data dict with keys:
        intraday_5m, daily, weekly, monthly, prev_day, pivots,
        intraday_date, fetched_at
    """
    cache_key = _cache_key(symbol)
    cached = st.session_state.get(cache_key)

    # Check cache freshness
    if cached and not force:
        from trading.utils.time_utils import is_market_open
        ttl = INTRADAY_TTL if is_market_open() else DAILY_TTL
        if not _is_stale(cached, ttl):
            return cached

    from trading.services.data_service import BrokerClient, DataService
    from trading.services.ticker_service import ticker_service
    from trading.utils.time_utils import get_candle_date_range, cap_end_time
    from trading.utils.indicators import camarilla_pivots

    token = ticker_service.get_token(symbol)
    if not token:
        return {"error": f"Token not found for {symbol}"}

    b = BrokerClient.get_instance()
    b.ensure_login()
    ds = DataService()

    intraday_date, history_start, history_end = get_candle_date_range()

    # ── API Call 1: Daily candles (90 days, 1 call) ──
    daily_candles = []
    raw_daily = b.fetch_candles(
        token,
        f"{history_start.isoformat()} 09:15",
        f"{history_end.isoformat()} 15:30",
        "ONE_DAY",
    )
    if raw_daily:
        daily_candles = _parse_raw_candles(raw_daily)

    # ── API Call 2: Intraday 5-min candles (multi-day, 1 call) ──
    # Fetch 7 calendar days of 5m candles (~5 trading days ≈ 375 bars).
    # This ensures we always have enough bars even when:
    #   - Market just opened (few bars today)
    #   - Some symbols have sparse early-session data
    #   - Viewing after hours / weekends
    intraday_5m = []
    intraday_start = (intraday_date - timedelta(days=6)).isoformat()
    end_time = cap_end_time(intraday_date.isoformat())
    raw_intraday = b.fetch_candles(
        token,
        f"{intraday_start} 09:15",
        end_time,
        "FIVE_MINUTE",
    )
    if raw_intraday:
        intraday_5m = _parse_raw_candles(raw_intraday)

    # Fallback: if today yields nothing, widen to last trading day
    if not intraday_5m:
        from trading.utils.time_utils import last_trading_day
        fallback_date = last_trading_day()
        fallback_start = (fallback_date - timedelta(days=6)).isoformat()
        raw_fallback = b.fetch_candles(
            token,
            f"{fallback_start} 09:15",
            f"{fallback_date.isoformat()} 15:30",
            "FIVE_MINUTE",
        )
        if raw_fallback:
            intraday_5m = _parse_raw_candles(raw_fallback)
            intraday_date = fallback_date
            logger.info(f"  {symbol}: Fallback 5m candles from {fallback_date} ({len(intraday_5m)} bars)")

    # ── Derive higher timeframes from daily ──
    weekly = _aggregate_to_weekly(daily_candles)
    monthly = _aggregate_to_monthly(daily_candles)

    # ── Prev day for pivots ──
    prev_day = None
    pivots = {}
    if len(daily_candles) >= 2:
        # Last complete trading day (not today's partial)
        prev_day = daily_candles[-1] if intraday_date != date.today() else (
            daily_candles[-1] if len(daily_candles) >= 1 else None
        )
        # If we have intraday for today, prev_day is second-to-last daily
        if intraday_date == date.today() and len(daily_candles) >= 2:
            prev_day = daily_candles[-1]
        elif intraday_date == date.today() and len(daily_candles) >= 1:
            prev_day = daily_candles[-1]

        if prev_day:
            pivots = camarilla_pivots(
                prev_day["high"], prev_day["low"], prev_day["close"]
            )
    elif len(daily_candles) == 1:
        prev_day = daily_candles[0]
        pivots = camarilla_pivots(
            prev_day["high"], prev_day["low"], prev_day["close"]
        )

    result = {
        "intraday_5m": intraday_5m,
        "daily": daily_candles,
        "weekly": weekly,
        "monthly": monthly,
        "prev_day": prev_day,
        "pivots": pivots,
        "intraday_date": intraday_date.isoformat(),
        "fetched_at": datetime.now(),
    }

    st.session_state[cache_key] = result
    logger.info(
        f"Cached {symbol}: {len(intraday_5m)} 5m candles, "
        f"{len(daily_candles)} daily, {len(weekly)} weekly, {len(monthly)} monthly"
    )
    return result


def get_candles_for_timeframe(symbol: str, timeframe: str) -> list:
    """
    Get cached candles for a specific timeframe.
    Timeframes: '5m', 'daily', 'weekly', 'monthly'
    """
    cached = st.session_state.get(_cache_key(symbol))
    if not cached:
        return []

    mapping = {
        "5m": "intraday_5m",
        "daily": "daily",
        "weekly": "weekly",
        "monthly": "monthly",
    }
    key = mapping.get(timeframe, "intraday_5m")
    return cached.get(key, [])


def invalidate_cache(symbol: str = None):
    """Clear candle cache for a symbol or all symbols."""
    if symbol:
        key = _cache_key(symbol)
        if key in st.session_state:
            del st.session_state[key]
    else:
        keys = [k for k in st.session_state if k.startswith(_KEY)]
        for k in keys:
            del st.session_state[k]
