"""Per-Minute volatility & tape-regime strip.

For a single symbol, walk today's 1-min bars and compute per minute:

  realised_vol_ann  σ of recent 20-minute returns × √(252×375)   (annualised)
  trades_per_sec    proxy: bar volume / 60
  regime_tag        TREND  vol high, abs(drift) > 0.7σ
                    CHOP   vol normal, drift ≈ 0
                    DEAD   vol < 0.4 × mean, volume < 0.4 × mean
                    SHOCK  vol > 2× mean

The strip is what the trader scans pre-entry to know whether tape conditions
support their setup. A SHOCK regime invalidates breakouts; DEAD invalidates
mean-reversion plays.
"""
from __future__ import annotations

import math
import statistics
from datetime import date
from typing import Any

from django.core.cache import cache


_TTL = 30
_VOL_WINDOW = 20
_ANN_BARS = 252 * 375    # trading-days × intraday-minutes


def _fetch_1m(symbol: str) -> list[dict]:
    key = f"vol_strip:1m:{symbol}:{date.today().isoformat()}"
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
            raw = broker.fetch_candles(token, start, end, "ONE_MINUTE", exchange=ticker_service.resolve_exchange(symbol)) or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_MINUTE") or []
        rows = [
            {"t": str(r[0]), "c": float(r[4]), "v": int(r[5]) if len(r) > 5 else 0}
            for r in raw if len(r) >= 5
        ]
        cache.set(key, rows, _TTL); return rows
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def build_vol_regime(symbol: str) -> dict[str, Any]:
    sym = (symbol or "").upper()
    if not sym:
        return {"symbol": "", "series": [], "error": "symbol required"}

    bars = _fetch_1m(sym)
    if not bars:
        return {"symbol": sym, "series": [], "current_regime": "no_data",
                "bar_count": 0, "note": "No 1-min bars yet."}

    closes = [b["c"] for b in bars]
    rets = [0.0] + [(closes[i] - closes[i - 1]) / closes[i - 1]
                     for i in range(1, len(closes))]
    vols = [int(b["v"]) for b in bars]
    mean_vol = (statistics.mean(vols) if vols else 0.0) or 1.0

    series: list[dict] = []
    vol_history: list[float] = []
    for i, b in enumerate(bars):
        window = rets[max(0, i - _VOL_WINDOW + 1) : i + 1]
        sd = statistics.pstdev(window) if len(window) >= 3 else 0.0
        realised_vol_ann = sd * math.sqrt(_ANN_BARS) * 100.0   # %
        vol_history.append(realised_vol_ann)
        mean_realised = statistics.mean(vol_history[-30:]) if len(vol_history) >= 3 else 0.0

        # 15-minute rolling vol — a smoother view than the per-minute σ.
        win15 = rets[max(0, i - 14) : i + 1]
        sd_15 = statistics.pstdev(win15) if len(win15) >= 3 else 0.0
        vol_15m_ann = round(sd_15 * math.sqrt(_ANN_BARS) * 100.0, 2)

        # Day percentile = where this minute's vol ranks vs the rest of today.
        if len(vol_history) >= 5:
            sorted_vols = sorted(vol_history)
            rank = sum(1 for v in sorted_vols if v <= realised_vol_ann)
            day_percentile = round(rank / len(sorted_vols) * 100.0, 1)
        else:
            day_percentile = 0.0

        drift = sum(window) if window else 0.0
        bar_v = vols[i]
        tps = round(bar_v / 60.0, 1)

        if realised_vol_ann > mean_realised * 2 and mean_realised > 0:
            regime = "SHOCK"
        elif realised_vol_ann < mean_realised * 0.4 and bar_v < mean_vol * 0.4:
            regime = "DEAD"
        elif mean_realised > 0 and sd > 0 and abs(drift) > sd * 0.7:
            regime = "TREND"
        else:
            regime = "CHOP"

        series.append({
            "t": b["t"],
            "c": round(b["c"], 2),
            "realised_vol_ann": round(realised_vol_ann, 2),
            "vol_15m_ann": vol_15m_ann,
            "day_percentile": day_percentile,
            "trades_per_sec": tps,
            "regime": regime,
        })

    last = series[-1] if series else {"regime": "no_data"}
    return {
        "symbol": sym,
        "bar_count": len(series),
        "current_regime": last["regime"],
        "current_vol_ann": last.get("realised_vol_ann", 0.0),
        "current_tps": last.get("trades_per_sec", 0.0),
        "series": series[-200:],
        "note": (
            "Per-minute regime tag. TREND = take direction signals; "
            "CHOP = mean-reversion or skip; DEAD = stand aside; "
            "SHOCK = cut size 50% and wait for vol to normalise."
        ),
    }
