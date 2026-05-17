"""Multi-timeframe stage scanner.

For each watchlist symbol, classify Daily / Weekly / Monthly each into one
of four Oliver-Kell / Stan-Weinstein-style phases:

  STAGE_1  base — consolidation under flat 30-period MA
  STAGE_2  uptrend — price > rising 30-period MA, slope positive
  STAGE_3  top — price stalls, MA flattens after uptrend
  STAGE_4  downtrend — price < falling 30-period MA

A symbol is "aligned" when all three timeframes agree on STAGE_2 (best
long set-up) or STAGE_4 (best short). Anything mixed is flagged so the
trader can avoid fighting the higher-TF current.
"""
from __future__ import annotations

import statistics
from datetime import date, timedelta
from typing import Any

from django.core.cache import cache

_TTL = 600


def _fetch_daily(symbol: str, days: int = 400) -> list[dict]:
    """Daily bars. We need ~400 to get 80 weekly + 18 monthly."""
    key = f"mtf:daily:{symbol}:{days}"
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
        start = (today - timedelta(days=days + 14)).strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange=ticker_service.resolve_exchange(symbol)) or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        rows = [
            {"t": str(r[0]), "h": float(r[2]), "l": float(r[3]), "c": float(r[4])}
            for r in raw if len(r) >= 5
        ]
        cache.set(key, rows, _TTL); return rows
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def _resample_weekly(daily: list[dict]) -> list[float]:
    """Take every 5th close — close-enough weekly proxy."""
    return [b["c"] for b in daily[::5]][-200:]


def _resample_monthly(daily: list[dict]) -> list[float]:
    """Every 21st close."""
    return [b["c"] for b in daily[::21]][-100:]


def _sma(series: list[float], window: int) -> float | None:
    if len(series) < window:
        return None
    return sum(series[-window:]) / window


def _classify_phase(closes: list[float], ma_window: int = 30) -> dict:
    """Return {stage, ma, slope_pct, close, gap_to_ma_pct}."""
    if len(closes) < ma_window + 5:
        return {"stage": "UNKNOWN", "ma": 0.0, "slope_pct": 0.0,
                "close": closes[-1] if closes else 0.0, "gap_pct": 0.0}
    ma_now = sum(closes[-ma_window:]) / ma_window
    ma_then = sum(closes[-ma_window - 5:-5]) / ma_window
    slope_pct = (ma_now - ma_then) / ma_then * 100.0 if ma_then > 0 else 0.0
    close = closes[-1]
    gap_pct = (close - ma_now) / ma_now * 100.0 if ma_now > 0 else 0.0

    if close > ma_now and slope_pct > 0.5:
        stage = "STAGE_2"
    elif close < ma_now and slope_pct < -0.5:
        stage = "STAGE_4"
    elif abs(slope_pct) < 0.3 and abs(gap_pct) < 3:
        stage = "STAGE_1"
    elif close > ma_now and slope_pct <= 0.5 and slope_pct >= -0.3:
        stage = "STAGE_3"
    else:
        stage = "STAGE_3" if close > ma_now else "STAGE_1"
    return {
        "stage": stage,
        "ma": round(ma_now, 2),
        "slope_pct": round(slope_pct, 2),
        "close": round(close, 2),
        "gap_pct": round(gap_pct, 2),
    }


def _watchlist() -> list[str]:
    syms: set[str] = set()
    try:
        from trading.models import WatchlistEntry
        for s in WatchlistEntry.objects.values_list("symbol", flat=True)[:60]:
            if s: syms.add(s)
    except Exception:  # noqa: BLE001
        pass
    return sorted(syms)


def build_mtf_stage_scanner(symbols: list[str] | None = None) -> dict[str, Any]:
    symbols = (symbols or _watchlist())[:30]
    rows: list[dict] = []
    for sym in symbols:
        daily = _fetch_daily(sym)
        if len(daily) < 40:
            rows.append({"symbol": sym, "daily": None, "weekly": None,
                         "monthly": None, "alignment": "no_data",
                         "stage2_aligned": False})
            continue
        d_closes = [b["c"] for b in daily]
        w_closes = _resample_weekly(daily)
        m_closes = _resample_monthly(daily)
        d_phase = _classify_phase(d_closes)
        w_phase = _classify_phase(w_closes)
        m_phase = _classify_phase(m_closes)

        stages = {d_phase["stage"], w_phase["stage"], m_phase["stage"]}
        if stages == {"STAGE_2"}:
            alignment = "long_aligned"
        elif stages == {"STAGE_4"}:
            alignment = "short_aligned"
        elif "STAGE_2" in stages and "STAGE_4" in stages:
            alignment = "conflict"
        else:
            alignment = "mixed"

        # Weinstein "Stage 2 alignment" specifically requires daily AND
        # weekly to be Stage 2 (monthly STAGE_1 or STAGE_2 acceptable). This
        # is the trader's classic textbook long set-up — break it out for
        # easy filtering in the FE.
        weinstein_stage2 = (
            d_phase["stage"] == "STAGE_2" and w_phase["stage"] == "STAGE_2"
            and m_phase["stage"] in ("STAGE_1", "STAGE_2")
        )

        rows.append({
            "symbol": sym,
            "daily": d_phase, "weekly": w_phase, "monthly": m_phase,
            "alignment": alignment,
            "stage2_aligned": weinstein_stage2,
        })

    rows.sort(key=lambda r: (
        0 if r["alignment"] == "long_aligned" else 1 if r["alignment"] == "short_aligned" else 2,
        r["symbol"],
    ))
    return {
        "count": len(rows),
        "rows": rows,
        "note": (
            "Daily/Weekly/Monthly each classified into Stage 1-4 using a "
            "30-period MA + slope. Long-aligned = all three on STAGE_2 — "
            "the strongest swing-long set-up. Conflict = some TFs say long, "
            "others say short — avoid."
        ),
    }
