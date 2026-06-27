"""Real-data seam for the scalp strategy — Fyers seconds candles.

The simulator falls back to a synthetic sample when this is unavailable, so the
strategy is demoable without a broker; with a linked Fyers account it streams
real intra-candle data. Resolution order: tenant's default ACTIVE Fyers
``BrokerLink`` → symbol-master ticker → ``adapter.history``.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path

from .engine import Candle
from .timeutil import IST

logger = logging.getLogger(__name__)

# Completed sessions are immutable, so we disk-cache the raw seconds candles
# (keyed by symbol+date+resolution) — re-running the same day costs no network.
_CANDLE_CACHE = Path(os.environ.get("SCALP_CANDLE_CACHE", "/tmp/alphadesk_scalp_candles"))


def _cache_path(symbol: str, day: str, resolution: str) -> Path:
    safe = symbol.replace(":", "_").replace("/", "_")
    return _CANDLE_CACHE / f"{safe}_{day}_{resolution}.json"


def _cache_get(symbol: str, day: str, resolution: str):
    p = _cache_path(symbol, day, resolution)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _cache_put(symbol: str, day: str, resolution: str, raw: list) -> None:
    try:
        _CANDLE_CACHE.mkdir(parents=True, exist_ok=True)
        p = _cache_path(symbol, day, resolution)
        # Write to a unique temp then atomically replace — a concurrent reader
        # never sees a torn/half-written JSON file (which forced a refetch).
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(raw))
        tmp.replace(p)
    except OSError as e:
        logger.debug("scalp.candle_cache.write_failed: %s", e)


def default_session_date() -> str:
    """Last COMPLETED trading day (today after close, else previous weekday)."""
    from trading.utils.time_utils import last_trading_day
    return last_trading_day().isoformat()


def resolve_expiry(underlying: str, expiry: str, on_date):
    """Explicit ``DDMMMYY`` expiry, or the nearest listed weekly/monthly."""
    expiry = (expiry or "").strip()
    if expiry:
        return _parse_expiry(expiry)
    from plugins.broker_fyers.symbols import nearest_expiry
    return nearest_expiry(underlying, on_date)


def resolve_session(underlying: str, expiry: str, date: str):
    """Resolve (date_str, expiry_date) applying the defaults. Pure resolution —
    no broker call — so callers can surface the chosen values in the UI."""
    day = (date or "").strip() or default_session_date()
    exp = resolve_expiry(underlying, expiry, datetime.strptime(day, "%Y-%m-%d").date())
    return day, exp


def fetch_scalp_candles(*, underlying: str, strike: int, opt_type: str, expiry: str,
                        date: str, resolution: str, tenant=None) -> list[Candle]:
    """Fetch seconds candles for one option for a single session date.

    Empty ``date`` → last completed trading day; empty ``expiry`` → nearest
    listed weekly (NIFTY/SENSEX) or monthly (BANKNIFTY). Raises with an
    actionable message on failure so the caller can fall back to the sample.
    """
    from apps.market_data.adapters.factory import build_adapter
    from plugins.broker_fyers.symbols import resolve_option_symbol

    day, exp = resolve_session(underlying, expiry, date)
    symbol = resolve_option_symbol(underlying, exp, int(strike), opt_type)

    # Cache only COMPLETED sessions (a past day's data never changes). Today's
    # session is still forming, so always fetch it fresh.
    is_past = day < datetime.now(IST).strftime("%Y-%m-%d")
    if is_past:
        cached = _cache_get(symbol, day, resolution)
        if cached:
            logger.info("scalp.candle_cache.hit symbol=%s day=%s res=%s", symbol, day, resolution)
            return [Candle.from_fyers(r) for r in cached]

    link = _fyers_link(tenant)
    adapter = build_adapter(link)
    if not adapter.authenticate():
        raise RuntimeError("Fyers daily re-login required — re-auth the Fyers broker link.")

    raw = adapter.history(symbol=symbol, resolution=resolution, range_from=day, range_to=day)
    if not raw:
        raise RuntimeError(f"Fyers returned no candles for {symbol} on {day} ({resolution}).")
    if is_past:
        _cache_put(symbol, day, resolution, raw)
    return [Candle.from_fyers(r) for r in raw]


def _parse_expiry(expiry: str):
    expiry = (expiry or "").strip()
    if not expiry:
        raise ValueError("expiry is required for live Fyers data (e.g. 07JUL26).")
    for fmt in ("%d%b%y", "%d%b%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(expiry.upper(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unparseable expiry {expiry!r} — use DDMMMYY (e.g. 07JUL26).")


def _fyers_link(tenant):
    from apps.market_data.models import BrokerLink

    qs = BrokerLink.objects.filter(broker_name="fyers", status=BrokerLink.Status.ACTIVE)
    if tenant is not None:
        qs = qs.filter(tenant=tenant)
    link = qs.order_by("-is_default").first()
    if not link:
        raise LookupError("No active Fyers BrokerLink — link Fyers in broker settings first.")
    return link
