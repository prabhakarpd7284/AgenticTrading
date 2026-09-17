"""Angel One scrip master — cached on disk, refreshed daily.

The scrip master is a ~50MB JSON file Angel publishes that maps every
tradable instrument (cash equity, NFO/BFO options, futures, currencies)
to a `symboltoken` used in every quote and order API.

We cache it on disk and refresh once a day. Module-level lookup
functions return ready-to-use views — option strike lists, underlying
spot tokens, nearest-expiry resolution. Everything is keyed by
underlying name in uppercase (NIFTY / BANKNIFTY / SENSEX / INDIAVIX),
so the AngelOneAdapter and any future broker (Zerodha, Fyers) can
share the same lookup surface even though the upstream master file
differs per broker.

Note: the public Angel master URL has been served as JSON without auth
since 2022, so we don't need broker credentials to refresh.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


SCRIP_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
CACHE_TTL_SEC = 24 * 3600
DEFAULT_CACHE_PATH = Path(os.environ.get(
    "ANGEL_SCRIP_CACHE",
    "/tmp/alphadesk_angel_scrip_master.json",
))

# Index-level tokens — these don't change and are stable across master refreshes.
INDEX_TOKENS: dict[str, tuple[str, str]] = {
    # underlying → (exch_seg, symboltoken)
    "NIFTY":     ("NSE", "99926000"),
    "BANKNIFTY": ("NSE", "99926009"),
    "FINNIFTY":  ("NSE", "99926037"),
    "MIDCPNIFTY":("NSE", "99926074"),
    "INDIAVIX":  ("NSE", "99926017"),
    "SENSEX":    ("BSE", "99919000"),
    "BANKEX":    ("BSE", "99919012"),
}


_scrip_cache: list[dict] | None = None


def _load_master(path: Path = DEFAULT_CACHE_PATH, force: bool = False) -> list[dict]:
    """Load the scrip master from cache, refreshing if stale."""
    global _scrip_cache
    if _scrip_cache is not None and not force:
        return _scrip_cache
    if path.exists() and not force and (time.time() - path.stat().st_mtime) < CACHE_TTL_SEC:
        try:
            _scrip_cache = json.loads(path.read_text())
            return _scrip_cache
        except Exception as e:
            logger.warning("scrip_master.cache_read_failed: %s — refreshing", e)
    # Refresh
    logger.info("scrip_master.downloading url=%s", SCRIP_URL)
    r = requests.get(SCRIP_URL, timeout=60)
    r.raise_for_status()
    data = r.json()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    except Exception as e:
        logger.warning("scrip_master.cache_write_failed: %s", e)
    _scrip_cache = data
    return _scrip_cache


# ── Public lookups ───────────────────────────────────────────────────

def get_underlying_token(underlying: str) -> Optional[str]:
    """Return the spot-token for an index. Falls back to None on unknown."""
    entry = INDEX_TOKENS.get(underlying.upper())
    return entry[1] if entry else None


def get_underlying_exchange(underlying: str) -> str:
    """Return NSE or BSE for an index. Defaults to NSE."""
    entry = INDEX_TOKENS.get(underlying.upper())
    return entry[0] if entry else "NSE"


def list_option_strikes(underlying: str, expiry: str) -> list[dict]:
    """Return all option legs for (underlying, expiry).

    Each entry has: {token, symbol, strike, opt, exch_seg, expiry}.
    `strike` is in rupees (not paise). `opt` is "CE" | "PE".

    expiry should be Angel's canonical DDMMMYYYY uppercase, e.g. "28APR2026".
    """
    out: list[dict] = []
    u = underlying.upper()
    for s in _load_master():
        if s.get("name") != u:
            continue
        if s.get("instrumenttype") not in ("OPTIDX", "OPTIONS"):
            continue
        if s.get("expiry") != expiry:
            continue
        sym = s.get("symbol", "")
        if not (sym.endswith("CE") or sym.endswith("PE")):
            continue
        try:
            # Angel master stores strike in paise as decimal string e.g. "2270000.000000"
            strike = int(s.get("strike", "0").split(".")[0]) // 100
        except (ValueError, AttributeError):
            continue
        out.append({
            "token": s.get("token"),
            "symbol": sym,
            "strike": strike,
            "opt": sym[-2:],
            "exch_seg": s.get("exch_seg"),
            "expiry": expiry,
        })
    return out


def nearest_expiry(underlying: str, on_or_after: datetime | None = None) -> Optional[str]:
    """Return the canonical-format expiry string for the nearest weekly/monthly
    option series ≥ `on_or_after` (default: today). Returns None if no series
    exists for this underlying."""
    target = (on_or_after or datetime.now()).date()
    u = underlying.upper()
    expiries: set[str] = set()
    for s in _load_master():
        if s.get("name") != u:
            continue
        if s.get("instrumenttype") not in ("OPTIDX", "OPTIONS"):
            continue
        e = s.get("expiry")
        if e:
            expiries.add(e)
    if not expiries:
        return None
    def _parse(e: str) -> datetime:
        return datetime.strptime(e, "%d%b%Y")
    future = sorted([e for e in expiries if _parse(e).date() >= target],
                    key=_parse)
    return future[0] if future else None


def list_expiries(underlying: str) -> list[str]:
    """All available expiries for an underlying, sorted ascending."""
    u = underlying.upper()
    expiries: set[str] = set()
    for s in _load_master():
        if s.get("name") != u:
            continue
        if s.get("instrumenttype") not in ("OPTIDX", "OPTIONS"):
            continue
        e = s.get("expiry")
        if e:
            expiries.add(e)
    return sorted(expiries, key=lambda x: datetime.strptime(x, "%d%b%Y"))


def refresh_master(force: bool = True) -> int:
    """Force a master refresh; returns instrument count."""
    return len(_load_master(force=force))
