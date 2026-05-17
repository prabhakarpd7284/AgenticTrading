"""FII / DII cash + F&O flow tape with NIFTY overlay.

Real-data path: scrape NSE's provisional cash + FII derivative pages
(nseindia.com/api/fiidiiTradeReact, nseindia.com/api/fiiderivatives).
That's blocked by NSE's anti-bot headers most days, so this stub
returns the structural payload + a NIFTY closing series so the React
chart still renders.

When the scraper lands, only `_fetch_flow()` needs to change.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from django.core.cache import cache


_TTL = 1800


def _nifty_close_series(days: int) -> list[dict]:
    """Live daily closes of NIFTY 50 via yfinance — used as overlay."""
    key = f"flow:nifty:{days}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        import yfinance as yf
        hist = yf.Ticker("^NSEI").history(period=f"{days + 5}d", auto_adjust=False)
        out = [
            {"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 2)}
            for d, c in zip(hist.index, hist["Close"].dropna().tolist())
        ][-days:]
        cache.set(key, out, _TTL); return out
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def build_fii_dii_flow(tenant=None, *, days: int = 30) -> dict[str, Any]:
    nifty = _nifty_close_series(days)
    return {
        "days": days,
        "fii_cash": [],
        "dii_cash": [],
        "fii_futures_oi": [],
        "fii_options_premium": [],
        "nifty_close": nifty,
        "data_source": "stub",
        "note": (
            "NIFTY overlay is live; FII/DII series are empty because the NSE "
            "provisional endpoint requires a session-cookie scraper. Wire "
            "apps/market_data/services/fii_dii_flow.py:_fetch_flow() to "
            "populate fii_cash, dii_cash, fii_futures_oi, fii_options_premium."
        ),
    }
