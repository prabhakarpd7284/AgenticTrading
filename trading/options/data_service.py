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
from typing import Optional, Tuple

from logzero import logger
from dotenv import load_dotenv

# Reuse the authenticated BrokerClient from the equity data service
from trading.services.data_service import BrokerClient

load_dotenv()

# ── Token constants (stable across expiries)
NIFTY_SPOT_TOKEN  = "99926000"
INDIA_VIX_TOKEN   = "99926017"

# ── NFO scrip master (downloaded once per session)
_nfo_master: Optional[list] = None


# ──────────────────────────────────────────────
# NFO instrument master loader
# ──────────────────────────────────────────────
def _load_nfo_master() -> list:
    """Load or download Angel One NFO scrip master JSON."""
    global _nfo_master
    if _nfo_master is not None:
        return _nfo_master

    nfo_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    logger.info("Downloading NFO instrument master from Angel One...")
    try:
        resp = urllib.request.urlopen(nfo_url, timeout=30)
        _nfo_master = json.loads(resp.read().decode("utf-8"))
        logger.info(f"NFO master loaded: {len(_nfo_master)} instruments")
    except Exception as e:
        logger.error(f"NFO master download failed: {e}")
        _nfo_master = []

    return _nfo_master


# ──────────────────────────────────────────────
# Token lookup for NIFTY options
# ──────────────────────────────────────────────
def find_option_token(
    underlying: str,
    strike: int,
    expiry_str: str,
    option_type: str,
) -> Optional[Tuple[str, str]]:
    """
    Find the Angel One NFO token for a NIFTY option.

    Args:
        underlying: "NIFTY" or "BANKNIFTY"
        strike: 24200
        expiry_str: "10MAR2026" (Angel One format)
        option_type: "CE" or "PE"

    Returns:
        (symbol, token) tuple, or None if not found.
    """
    master = _load_nfo_master()
    if not master:
        return None

    # Angel One symbol format: NIFTY10MAR2624200CE
    # But the year is 2-digit in the symbol (26 not 2026)
    target_year_2d = expiry_str[-4:-2] if len(expiry_str) >= 4 else expiry_str[-4:]
    target_month   = expiry_str[:5] if len(expiry_str) >= 9 else expiry_str[:3]  # "10MAR"
    target_day     = expiry_str[:2] if expiry_str[0].isdigit() else ""

    # Build the expected symbol substring
    expected_suffix = f"{strike}{option_type}"
    expected_expiry = f"{expiry_str[:len(expiry_str)-4]}{target_year_2d}"  # e.g. "10MAR26"

    for inst in master:
        sym = inst.get("symbol", "")
        if (
            inst.get("exch_seg") == "NFO"
            and inst.get("name") == underlying
            and inst.get("instrumenttype") == "OPTIDX"
            and option_type in sym
            and str(strike) in sym
            and sym.startswith(underlying)
        ):
            # Parse expiry from symbol: NIFTY10MAR2624200CE → expiry part = 10MAR26
            after_name = sym[len(underlying):]  # "10MAR2624200CE"
            # expiry is everything before the strike number
            strike_pos = after_name.find(str(strike))
            if strike_pos > 0:
                sym_expiry = after_name[:strike_pos]  # "10MAR26"
                sym_option = after_name[strike_pos + len(str(strike)):]  # "CE" or "PE"
                if sym_option == option_type and sym_expiry.upper() == expected_expiry.upper():
                    return sym, inst.get("token", "")

    logger.warning(f"Token not found: {underlying} {strike} {expiry_str} {option_type}")
    return None


# ──────────────────────────────────────────────
# Options Data Service
# ──────────────────────────────────────────────
class OptionsDataService:
    """
    Fetches live NFO options data and NIFTY/VIX spot data.
    Wraps BrokerClient — reuses the single authenticated session.
    """

    def __init__(self):
        self._broker: Optional[BrokerClient] = None

    def _ensure_broker(self):
        if self._broker is None:
            self._broker = BrokerClient()
        if not self._broker.is_logged_in:
            self._broker.login()

    # ── NIFTY 50 Spot ──
    def fetch_nifty_spot(self) -> dict:
        """
        Fetch NIFTY 50 index LTP + OHLC.

        Returns:
            {"ltp": 24028.05, "open": 23868.05, "high": 24078.15,
             "low": 23697.80, "prev_close": 24450.45}
        """
        self._ensure_broker()
        try:
            r = self._broker.smart_api.ltpData("NSE", "Nifty 50", NIFTY_SPOT_TOKEN)
            d = r.get("data", {})
            return {
                "ltp":        float(d.get("ltp", 0)),
                "open":       float(d.get("open", 0)),
                "high":       float(d.get("high", 0)),
                "low":        float(d.get("low", 0)),
                "prev_close": float(d.get("close", 0)),
            }
        except Exception as e:
            logger.error(f"NIFTY spot fetch failed: {e}")
            return {}

    # ── India VIX ──
    def fetch_vix(self) -> dict:
        """
        Fetch India VIX.

        Returns:
            {"ltp": 23.36, "prev_close": 19.88, "high": 24.49, "low": 19.22}
        """
        self._ensure_broker()
        try:
            r = self._broker.smart_api.ltpData("NSE", "India VIX", INDIA_VIX_TOKEN)
            d = r.get("data", {})
            return {
                "ltp":        float(d.get("ltp", 0)),
                "prev_close": float(d.get("close", 0)),
                "high":       float(d.get("high", 0)),
                "low":        float(d.get("low", 0)),
            }
        except Exception as e:
            logger.error(f"VIX fetch failed: {e}")
            return {}

    # ── Option LTP by token ──
    def fetch_option_ltp(self, symbol: str, token: str) -> dict:
        """
        Fetch LTP + OHLC for a specific NFO option.

        Args:
            symbol: "NIFTY10MAR2624200CE"
            token:  "45482"

        Returns:
            {"ltp": 96.40, "open": 99.55, "high": 148.00, "low": 50.85, "prev_close": 394.85}
        """
        self._ensure_broker()
        try:
            r = self._broker.smart_api.ltpData("NFO", symbol, token)
            d = r.get("data", {})
            return {
                "ltp":        float(d.get("ltp", 0)),
                "open":       float(d.get("open", 0)),
                "high":       float(d.get("high", 0)),
                "low":        float(d.get("low", 0)),
                "prev_close": float(d.get("close", 0)),
            }
        except Exception as e:
            logger.error(f"Option LTP fetch failed ({symbol}): {e}")
            return {}

    # ── NIFTY 5-min candles ──
    def fetch_nifty_candles(self, date_str: str, interval: str = "FIVE_MINUTE") -> list:
        """
        Fetch intraday 5-min candles for NIFTY 50 index.

        Args:
            date_str: "2026-03-09"
            interval: "FIVE_MINUTE" (default), "FIFTEEN_MINUTE", "ONE_HOUR"

        Returns:
            List of [timestamp, open, high, low, close, volume]
        """
        self._ensure_broker()
        try:
            return self._broker.fetch_candles(
                symbol_token=NIFTY_SPOT_TOKEN,
                start=f"{date_str} 09:15",
                end=f"{date_str} 15:30",
                interval=interval,
                exchange="NSE",
            )
        except Exception as e:
            logger.error(f"NIFTY candle fetch failed: {e}")
            return []

    # ── Full market snapshot (single call for straddle cycle) ──
    def fetch_straddle_snapshot(
        self,
        ce_symbol: str,
        ce_token: str,
        pe_symbol: str,
        pe_token: str,
        date_str: str,
    ) -> dict:
        """
        Fetch all data needed for one straddle management cycle.

        Returns:
            {
                "nifty": {...},      # spot OHLC + LTP
                "vix": {...},        # VIX OHLC + LTP
                "ce": {...},         # CE option OHLC + LTP
                "pe": {...},         # PE option OHLC + LTP
                "candles": [...],    # NIFTY 5-min candles
            }
        """
        nifty   = self.fetch_nifty_spot()
        vix     = self.fetch_vix()
        ce      = self.fetch_option_ltp(ce_symbol, ce_token)
        pe      = self.fetch_option_ltp(pe_symbol, pe_token)
        candles = self.fetch_nifty_candles(date_str)

        return {
            "nifty":   nifty,
            "vix":     vix,
            "ce":      ce,
            "pe":      pe,
            "candles": candles,
        }
