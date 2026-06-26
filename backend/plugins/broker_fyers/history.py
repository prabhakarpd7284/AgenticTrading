"""Fyers v3 historical candles — seconds resolution.

Angel One's historical floor is 1-minute; Fyers serves seconds resolutions
(5S/10S/15S/30S/45S, ~30 trading-day retention) which the scalp strategy needs
for intra-candle pressure. This is a thin wrapper over ``fyersModel.history`` so
the rest of the codebase never touches the SDK response envelope directly.
"""
from __future__ import annotations

from .adapter import _raise_if_error

# Fyers max span per request: 100 days for minutes, ~30 trading days for seconds.
SECONDS_RESOLUTIONS = {"5S", "10S", "15S", "30S", "45S"}


def fetch_history(api, symbol: str, resolution: str = "5S", range_from: str = "",
                  range_to: str = "", cont_flag: int = 1) -> list[list]:
    """Return raw Fyers candles ``[[epoch, o, h, l, c, v], …]`` (epoch seconds).

    ``range_from`` / ``range_to`` are ``YYYY-MM-DD`` (date_format=1). A single
    trading day fits one request for any resolution, which is all the MVP needs.
    """
    data = {
        "symbol": symbol,
        "resolution": str(resolution),
        "date_format": "1",
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": str(cont_flag),
    }
    resp = api.history(data=data)
    _raise_if_error(resp, "history")
    candles = (resp or {}).get("candles") or []
    if not isinstance(candles, list):
        raise RuntimeError(f"Fyers history: unexpected candles shape for {symbol}")
    return candles
