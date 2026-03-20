"""
Options Data Service — NFO (Futures & Options) data from Angel One SmartAPI.

Reuses BrokerClient from trading.services.data_service (single auth session).
Adds options-specific methods:
  - NIFTY 50 spot price + 5-min candles
  - India VIX
  - NFO option LTP by token
  - NFO instrument token search from Angel One scrip master
"""
import json
import urllib.request
import os
import re
import time
import threading
from pathlib import Path
from datetime import date
from typing import Optional, Tuple

from trading.utils.time_utils import can_fetch_candles, cap_end_time
from trading.utils.expiry_utils import normalize_expiry

from logzero import logger
from dotenv import load_dotenv

from trading.services.data_service import BrokerClient

load_dotenv()

# ── Token constants (stable across expiries)
NIFTY_SPOT_TOKEN    = "99926000"
INDIA_VIX_TOKEN     = "99926017"
BANKNIFTY_SPOT_TOKEN = "99926009"

# ── NFO scrip master — in-process cache
_nfo_master: Optional[list] = None


# ──────────────────────────────────────────────
# NFO instrument master loader (disk-cached per day)
# ──────────────────────────────────────────────
def _nfo_cache_path() -> Path:
    return Path(f"/tmp/nfo_master_{date.today().isoformat()}.json")


def _load_nfo_master() -> list:
    """
    Load Angel One NFO scrip master.
    Order of preference:
      1. In-process memory cache (instant)
      2. Disk cache for today (fast, ~10ms)
      3. Fresh download (~5s, saved to disk for reuse)
    """
    global _nfo_master
    if _nfo_master is not None:
        return _nfo_master

    cache = _nfo_cache_path()
    if cache.exists():
        try:
            loaded = json.loads(cache.read_text())
            _nfo_master = loaded
            logger.info(f"NFO master loaded from disk cache: {len(loaded)} instruments")
            return loaded
        except Exception as e:
            logger.warning(f"NFO master disk cache corrupted, re-downloading: {e}")
            cache.unlink(missing_ok=True)

    nfo_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    logger.info("Downloading NFO instrument master from Angel One...")
    try:
        resp = urllib.request.urlopen(nfo_url, timeout=30)
        downloaded = json.loads(resp.read().decode("utf-8"))
        _nfo_master = downloaded
        logger.info(f"NFO master downloaded: {len(downloaded)} instruments")
        try:
            cache.write_text(json.dumps(downloaded))
            logger.info(f"NFO master cached to disk: {cache}")
        except Exception as e:
            logger.warning(f"NFO master disk cache write failed (non-fatal): {e}")
    except Exception as e:
        logger.error(f"NFO master download failed: {e}")
        _nfo_master = []

    return _nfo_master or []


# ──────────────────────────────────────────────
# Token lookup for NIFTY/BANKNIFTY options
# ──────────────────────────────────────────────
def find_atm_strike(spot: float, step: int = 50) -> int:
    """
    Find the ATM strike closest to spot price.

    For a short straddle, the ideal strike is the one nearest to spot
    to minimize initial delta skew.

    Args:
        spot: Current NIFTY/BANKNIFTY spot price
        step: Strike interval (50 for NIFTY, 100 for BANKNIFTY)

    Returns:
        Strike price closest to spot (e.g. 23150 when spot is 23171)
    """
    lower = int(spot // step) * step          # e.g. 23150
    upper = lower + step                       # e.g. 23200
    # Pick whichever is closer to spot
    if abs(spot - lower) <= abs(spot - upper):
        return lower
    return upper


def find_option_token(
    underlying: str,
    strike: int,
    expiry_str: str,
    option_type: str,
) -> Optional[Tuple[str, str]]:
    """
    Find the Angel One NFO token for a NIFTY/BANKNIFTY option.

    Delegates to the centralized ticker_service which keeps a daily-refreshed
    in-memory index of all NFO instruments.

    Args:
        underlying: "NIFTY" or "BANKNIFTY"
        strike: 24200
        expiry_str: "10MAR26" or "10MAR2026" (both accepted)
        option_type: "CE" or "PE"

    Returns:
        (symbol, token) tuple, or None if not found.
    """
    from trading.services.ticker_service import ticker_service

    normalized = normalize_expiry(expiry_str)
    if not normalized:
        logger.warning(f"Cannot parse expiry '{expiry_str}'. Expected e.g. '10MAR26' or '2026-03-10'.")
        return None

    result = ticker_service.get_nfo_options(underlying, strike, normalized)
    if option_type in result:
        return result[option_type]

    logger.warning(f"Token not found: {underlying} {strike} {normalized} {option_type}")
    return None


# ──────────────────────────────────────────────
# Options Data Service
# ──────────────────────────────────────────────
class OptionsDataService:
    """
    Fetches live NFO options data and NIFTY/VIX spot data.
    Wraps BrokerClient — reuses the single authenticated session.

    Use the module-level singleton _svc (created at import time in graph.py)
    rather than instantiating a new one per request.
    """

    def __init__(self, cache_ttl: int = 5):
        self._broker: Optional[BrokerClient] = None
        self._login_lock = threading.Lock()
        # Note: LTP caching is now handled by BrokerClient.ltp() with built-in TTL.
        # Keep a reference for backward compat (_ensure_broker, _broker access).

    def _ensure_broker(self) -> None:
        """Get the singleton broker and ensure it's logged in."""
        with self._login_lock:
            if self._broker is None:
                self._broker = BrokerClient.get_instance()
            self._broker.ensure_login()
        assert self._broker is not None

    # ── NIFTY 50 Spot ──
    def fetch_nifty_spot(self) -> dict:
        self._ensure_broker()
        return self._broker.ltp("NSE", "Nifty 50", NIFTY_SPOT_TOKEN)

    # ── India VIX ──
    def fetch_vix(self) -> dict:
        self._ensure_broker()
        return self._broker.ltp("NSE", "India VIX", INDIA_VIX_TOKEN)

    # ── BANKNIFTY Spot ──
    def fetch_banknifty_spot(self) -> dict:
        self._ensure_broker()
        return self._broker.ltp("NSE", "Nifty Bank", BANKNIFTY_SPOT_TOKEN)

    # ── VIX intraday candles ──
    def fetch_vix_candles(self, date_str: str, interval: str = "FIVE_MINUTE") -> list:
        if date_str == date.today().isoformat() and not can_fetch_candles():
            return []
        self._ensure_broker()
        try:
            end_str = cap_end_time(date_str)
            result = self._broker.fetch_candles(
                symbol_token=INDIA_VIX_TOKEN,
                start=f"{date_str} 09:15",
                end=end_str,
                interval=interval,
                exchange="NSE",
            )
            return result if result is not None else []
        except Exception as e:
            logger.error(f"VIX candle fetch failed: {e}")
            return []

    # ── Option LTP by token ──
    def fetch_option_ltp(self, symbol: str, token: str) -> dict:
        """Fetch option LTP via centralized broker (cached, rate-limited)."""
        self._ensure_broker()
        return self._broker.ltp("NFO", symbol, token)

    # ── NIFTY 5-min candles ──
    def fetch_nifty_candles(self, date_str: str, interval: str = "FIVE_MINUTE") -> list:
        if date_str == date.today().isoformat() and not can_fetch_candles():
            return []
        self._ensure_broker()
        try:
            end_str = cap_end_time(date_str)
            result = self._broker.fetch_candles(
                symbol_token=NIFTY_SPOT_TOKEN,
                start=f"{date_str} 09:15",
                end=end_str,
                interval=interval,
                exchange="NSE",
            )
            return result if result is not None else []
        except Exception as e:
            logger.error(f"NIFTY candle fetch failed: {e}")
            return []

    # ── Full market snapshot ──
    def fetch_straddle_snapshot(
        self,
        ce_symbol: str,
        ce_token: str,
        pe_symbol: str,
        pe_token: str,
        date_str: str,
        include_candles: bool = True,
    ) -> dict:
        """
        Fetch all data for one straddle management cycle.

        All calls go through BrokerClient which handles rate limiting (0.4s
        min gap) and LTP caching (5s TTL). NIFTY/VIX are usually cache hits.

        Returns:
            {nifty, vix, ce, pe, candles}
        """
        self._ensure_broker()

        results: dict = {}

        for key, fn in [
            ("nifty", lambda: self.fetch_nifty_spot()),
            ("vix",   lambda: self.fetch_vix()),
            ("ce",    lambda: self.fetch_option_ltp(ce_symbol, ce_token)),
            ("pe",    lambda: self.fetch_option_ltp(pe_symbol, pe_token)),
        ]:
            try:
                results[key] = fn()
            except Exception as e:
                logger.error(f"Snapshot fetch '{key}' failed: {e}")
                results[key] = {}

        if include_candles:
            try:
                results["candles"] = self.fetch_nifty_candles(date_str)
            except Exception as e:
                logger.error(f"Candle fetch failed: {e}")
                results["candles"] = []
        else:
            results["candles"] = []

        return results
