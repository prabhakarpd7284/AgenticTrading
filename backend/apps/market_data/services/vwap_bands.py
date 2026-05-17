"""VWAP + deviation bands.

For a single symbol, build the intraday VWAP from 09:15 onward plus
±1σ / ±2σ rolling bands. The σ is computed across minute returns from
session start — same convention TradingView calls "anchored VWAP".

Tags the latest bar as `stretched_up` if price > +2σ, `stretched_down`
if < −2σ, otherwise `neutral`. The state lets callers wire a one-line
mean-reversion alert without re-implementing the math.
"""
from __future__ import annotations

import math
import statistics
from datetime import date, datetime
from typing import Any

from django.core.cache import cache

from trading.utils.time_utils import intraday_session_date

_TTL = 30


def _fetch_1m_today(symbol: str) -> list[dict]:
    """1-min OHLCV bars for the active intraday session — closed sessions
    read from the persistent candle_store (30-day Redis TTL), today/live
    sessions share a short-TTL key with all other intraday panels."""
    from apps.market_data.services import candle_store
    from trading.services.ticker_service import ticker_service
    return candle_store.fetch_intraday_bars(
        symbol, "1m", "ONE_MINUTE",
        ticker_service.resolve_exchange,
        short_ttl=_TTL,
    )


def _build_anchored_vwap(bars: list[dict]) -> list[dict]:
    """Walk bars cumulatively producing {t, c, vwap, sigma1, sigma2}."""
    if not bars:
        return []
    cum_vol = 0
    cum_pv = 0.0
    closes: list[float] = []
    out: list[dict] = []
    for b in bars:
        typical = (b["h"] + b["l"] + b["c"]) / 3.0
        v = max(b["v"], 1)
        cum_vol += v
        cum_pv += typical * v
        vwap = cum_pv / cum_vol
        closes.append(b["c"])
        if len(closes) >= 5:
            sd = statistics.pstdev(closes[-30:]) if len(closes) >= 5 else 0.0
        else:
            sd = 0.0
        out.append({
            "t": b["t"], "c": b["c"], "vwap": round(vwap, 2),
            "sigma1_up": round(vwap + sd, 2),
            "sigma1_dn": round(vwap - sd, 2),
            "sigma2_up": round(vwap + 2 * sd, 2),
            "sigma2_dn": round(vwap - 2 * sd, 2),
            "sd": round(sd, 3),
        })
    return out


def build_vwap_bands(symbol: str) -> dict[str, Any]:
    sym = (symbol or "").upper()
    if not sym:
        return {"symbol": "", "series": [], "state": "neutral", "error": "symbol required"}

    bars = _fetch_1m_today(sym)
    series = _build_anchored_vwap(bars)
    if not series:
        return {
            "symbol": sym, "series": [], "vwap": 0.0,
            "sigma1_up": 0.0, "sigma1_dn": 0.0,
            "sigma2_up": 0.0, "sigma2_dn": 0.0,
            "dist_sigma": 0.0, "state": "no_data",
            "note": (
                f"No 1-min bars for {sym} on {intraday_session_date().isoformat()} "
                "— symbol may be unknown or the session had no trades."
            ),
        }

    last = series[-1]
    sd = last["sd"]
    dist_sigma = round((last["c"] - last["vwap"]) / sd, 2) if sd > 0 else 0.0
    # z-score is the same as dist_sigma but clamped for alert purposes.
    dist_zscore = dist_sigma   # explicit field so the FE can switch terminology
    state = (
        "stretched_up" if dist_sigma >= 2.0
        else "stretched_down" if dist_sigma <= -2.0
        else "neutral"
    )
    # Alert level: 1 = warning (|z| >= 1.5), 2 = critical (|z| >= 2.5)
    abs_z = abs(dist_sigma)
    alert_level = 2 if abs_z >= 2.5 else 1 if abs_z >= 1.5 else 0
    return {
        "symbol": sym,
        "vwap": last["vwap"],
        "sigma1_up": last["sigma1_up"], "sigma1_dn": last["sigma1_dn"],
        "sigma2_up": last["sigma2_up"], "sigma2_dn": last["sigma2_dn"],
        "dist_sigma": dist_sigma,
        "dist_zscore": dist_zscore,
        "alert_level": alert_level,
        "state": state,
        "last_close": last["c"],
        "series": series[-200:],   # cap payload
        "bar_count": len(series),
    }
