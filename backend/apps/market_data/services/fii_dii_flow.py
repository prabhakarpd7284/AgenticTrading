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


def _classify_regime(series: list[dict], window: int) -> str:
    """Return a regime tag derived from the rolling N-day sum of `value`.

      BULLISH_FII        rolling sum > 0 for window/2+ days
      BEARISH_FII        rolling sum < 0 for window/2+ days
      DISTRIBUTING       net sells through up days (correlation with nifty)
      ACCUMULATING       net buys through down days
      MIXED              everything else
    """
    if not series or len(series) < window:
        return "no_data"
    recent = series[-window:]
    vals = [float(p.get("value", 0) or 0) for p in recent]
    net = sum(vals)
    pos_days = sum(1 for v in vals if v > 0)
    neg_days = sum(1 for v in vals if v < 0)
    if pos_days >= window * 0.6:
        return "BULLISH_FII"
    if neg_days >= window * 0.6:
        return "BEARISH_FII"
    return "ACCUMULATING" if net > 0 else "DISTRIBUTING" if net < 0 else "MIXED"


def build_fii_dii_flow(tenant=None, *, days: int = 30) -> dict[str, Any]:
    nifty = _nifty_close_series(days)
    fii_cash: list[dict] = []
    dii_cash: list[dict] = []
    fii_futures_oi: list[dict] = []
    fii_options_premium: list[dict] = []

    return {
        "days": days,
        "fii_cash": fii_cash,
        "dii_cash": dii_cash,
        "fii_futures_oi": fii_futures_oi,
        "fii_options_premium": fii_options_premium,
        "nifty_close": nifty,
        "regimes": {
            "fii_cash_5d":   _classify_regime(fii_cash, 5),
            "fii_cash_20d":  _classify_regime(fii_cash, 20),
            "dii_cash_5d":   _classify_regime(dii_cash, 5),
            "dii_cash_20d":  _classify_regime(dii_cash, 20),
            "fii_futures_oi_5d":  _classify_regime(fii_futures_oi, 5),
        },
        "data_source": "stub" if not fii_cash else "live",
        "note": (
            "NIFTY overlay is live; FII/DII series are empty until the NSE "
            "provisional scraper is wired. Once populated, the regimes block "
            "will tag 5d + 20d rolling buckets as BULLISH_FII / BEARISH_FII / "
            "ACCUMULATING / DISTRIBUTING / MIXED."
        ),
    }
