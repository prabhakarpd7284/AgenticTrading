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

_TTL = 30


def _fetch_1m_today(symbol: str) -> list[dict]:
    """Today's 1-min OHLCV bars. Cached 30s so a 10-symbol scan stays cheap."""
    key = f"vwap:1m:{symbol}:{date.today().isoformat()}"
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
             "l": float(r[3]), "c": float(r[4]),
             "v": int(r[5]) if len(r) > 5 else 0}
            for r in raw if len(r) >= 5
        ]
        cache.set(key, out, _TTL)
        return out
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL)
        return []


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
            "note": "No 1-min bars yet (market closed or symbol unknown).",
        }

    last = series[-1]
    sd = last["sd"]
    dist_sigma = round((last["c"] - last["vwap"]) / sd, 2) if sd > 0 else 0.0
    state = (
        "stretched_up" if dist_sigma >= 2.0
        else "stretched_down" if dist_sigma <= -2.0
        else "neutral"
    )
    return {
        "symbol": sym,
        "vwap": last["vwap"],
        "sigma1_up": last["sigma1_up"], "sigma1_dn": last["sigma1_dn"],
        "sigma2_up": last["sigma2_up"], "sigma2_dn": last["sigma2_dn"],
        "dist_sigma": dist_sigma,
        "state": state,
        "last_close": last["c"],
        "series": series[-200:],   # cap payload
        "bar_count": len(series),
    }
