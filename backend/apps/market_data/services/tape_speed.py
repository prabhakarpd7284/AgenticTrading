"""Tape-Speed & Volatility-Per-Minute gauge.

For a single symbol, walk today's 1-min bars and surface:

  trades_per_sec     proxy: bar volume / 60
  rupees_per_min     bar volume × bar close (≈ turnover)
  baseline_rupees    20-day same-time-of-day median turnover
  realised_vol_pm    1-min stdev of returns × √375 (intraday vol units)
  state              cold | normal | hot | shock

State is derived from rupees_per_min / baseline:
  < 0.5×   → cold
  0.5-1.5  → normal
  1.5-2.5  → hot
  > 2.5    → shock
"""
from __future__ import annotations

import math
import statistics
from datetime import date, datetime, timedelta
from typing import Any

from django.core.cache import cache


_TTL = 30


def _fetch_1m_today(symbol: str) -> list[dict]:
    key = f"tape:1m:{symbol}:{date.today().isoformat()}"
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
            {"t": str(r[0]), "c": float(r[4]), "v": int(r[5]) if len(r) > 5 else 0}
            for r in raw if len(r) >= 5
        ]
        cache.set(key, out, _TTL); return out
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def _baseline_turnover(symbol: str, days: int = 20) -> float:
    """Median per-minute turnover over the last N daily bars. Cached 1 hour."""
    key = f"tape:base:{symbol}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service
        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            cache.set(key, 0.0, 3600); return 0.0
        today = date.today()
        start = (today - timedelta(days=days + 2)).strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange="NSE") or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        turnovers = []
        for r in raw[-days:]:
            if len(r) >= 6:
                c = float(r[4]); v = int(r[5] or 0)
                if v > 0 and c > 0:
                    turnovers.append(c * v / 375.0)  # per-minute
        v = round(statistics.median(turnovers), 2) if turnovers else 0.0
        cache.set(key, v, 3600); return v
    except Exception:  # noqa: BLE001
        cache.set(key, 0.0, 3600); return 0.0


def build_tape_speed(symbol: str) -> dict[str, Any]:
    sym = (symbol or "").upper()
    if not sym:
        return {"symbol": "", "series": [], "current_state": "no_data", "error": "symbol required"}

    bars = _fetch_1m_today(sym)
    if not bars:
        return {"symbol": sym, "series": [], "current_state": "no_data",
                "baseline_rupees_per_min": 0.0, "bar_count": 0,
                "current_cum_delta": 0,
                "note": "No 1-min bars yet."}

    baseline = _baseline_turnover(sym)
    closes = [b["c"] for b in bars]
    rets = [0.0] + [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
    # 10s-equivalent EMA over per-minute samples: with α=1/6 ≈ 10-second
    # decay relative to 1-min bars (since 60/6 = 10). Lets the trader spot
    # intra-minute intensity shifts that a raw bar series hides.
    alpha = 1.0 / 6.0
    ema_tps = 0.0
    ema_ratio = 0.0
    # Cumulative delta proxy: sign(close-prev_close) × volume per bar,
    # then running sum. True aggressor delta needs tick data we don't get
    # from the SDK; bar-direction × volume is the standard cheap proxy.
    cum_delta = 0
    series: list[dict] = []
    for i, b in enumerate(bars):
        tps = round(b["v"] / 60.0, 1)
        rupees_pm = round(b["c"] * b["v"], 2) if b["v"] > 0 else 0.0
        window = rets[max(0, i - 19) : i + 1]
        sd = statistics.pstdev(window) if len(window) >= 3 else 0.0
        realised_vol_pm = round(sd * math.sqrt(375) * 100.0, 3)   # %
        ratio = (rupees_pm / baseline) if baseline > 0 else 0.0
        ema_tps = alpha * tps + (1 - alpha) * ema_tps if i else tps
        ema_ratio = alpha * ratio + (1 - alpha) * ema_ratio if i else ratio
        # Bar-direction × bar-volume → signed aggressor proxy
        bar_delta = 0
        if i > 0:
            sign = 1 if b["c"] > closes[i - 1] else -1 if b["c"] < closes[i - 1] else 0
            bar_delta = sign * b["v"]
            cum_delta += bar_delta
        if ratio == 0:
            state = "cold"
        elif ratio > 2.5:
            state = "shock"
        elif ratio > 1.5:
            state = "hot"
        elif ratio < 0.5:
            state = "cold"
        else:
            state = "normal"
        series.append({
            "t": b["t"], "c": round(b["c"], 2),
            "trades_per_sec": tps,
            "trades_per_sec_ema10s": round(ema_tps, 2),
            "rupees_per_min": rupees_pm,
            "ratio_to_baseline": round(ratio, 2),
            "ratio_ema10s": round(ema_ratio, 2),
            "realised_vol_pm": realised_vol_pm,
            "bar_delta": bar_delta,
            "cum_delta": cum_delta,
            "state": state,
        })

    last = series[-1] if series else {}
    return {
        "symbol": sym,
        "baseline_rupees_per_min": baseline,
        "bar_count": len(series),
        "current_state": last.get("state", "no_data"),
        "current_tps": last.get("trades_per_sec", 0.0),
        "current_rupees_per_min": last.get("rupees_per_min", 0.0),
        "current_realised_vol_pm": last.get("realised_vol_pm", 0.0),
        "current_cum_delta": last.get("cum_delta", 0),
        "series": series[-180:],
        "note": (
            "Hot tape ≥ 1.5× baseline turnover = institutional flow active; "
            "size up breakouts. Cold tape ≤ 0.5× = stand aside, your setup "
            "won't follow through. Shock = stop everything, wait for vol to "
            "normalise."
        ),
    }
