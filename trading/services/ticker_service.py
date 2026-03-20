"""
Ticker Service — centralized symbol master management + ticker lookup.

Single source of truth for:
  - Angel One token ↔ ticker mapping (NSE, NFO, BSE)
  - Instrument metadata (lot size, tick size, instrument type, expiry)
  - NIFTY 50 / BANKNIFTY component validation
  - Auto-download + disk-cache of symbol masters (daily refresh)

Usage:
    from trading.services.ticker_service import ticker_service

    # Token lookup
    tok = ticker_service.get_token("RELIANCE")          # "2885"
    tok = ticker_service.get_token("RELIANCE", "BSE")   # BSE token

    # Full instrument info
    info = ticker_service.get_info("RELIANCE")
    # → {"token": "2885", "symbol": "RELIANCE-EQ", "name": "RELIANCE",
    #    "exch_seg": "NSE", "lot_size": 1, "tick_size": 10.0, ...}

    # Reverse lookup
    name = ticker_service.token_to_name("2885", "NSE")  # "RELIANCE"

    # Validate if ticker exists
    ticker_service.exists("TATAMOTORS")  # False (demerged)

    # Search
    ticker_service.search("TATA")  # ["TATASTEEL", "TATAPOWER", "TATACONSUM", ...]

    # NFO option chain helpers
    tokens = ticker_service.get_nfo_options("NIFTY", 23200, "17MAR26")
    # → {"CE": ("NIFTY17MAR2623200CE", "57710"), "PE": ("NIFTY17MAR2623200PE", "57711")}
"""
import json
import os
import time
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from logzero import logger

# ── Download URLs ──
_SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

# ── Cache directory ──
_CACHE_DIR = Path("/tmp/angel_one_masters")


class TickerService:
    """
    Centralized ticker info service backed by Angel One symbol masters.

    Lazy-loaded: masters are downloaded on first access, then cached to disk
    for the rest of the day. In-memory lookups are O(1) dict access.
    """

    def __init__(self):
        self._instruments: Optional[list] = None
        self._nse_by_name: Dict[str, dict] = {}   # "RELIANCE" → full instrument dict
        self._nse_by_token: Dict[str, dict] = {}   # "2885" → full instrument dict
        self._nfo_by_key: Dict[str, dict] = {}     # "NFO:NIFTY17MAR2623200CE" → inst
        self._all_by_key: Dict[str, str] = {}      # "NSE:RELIANCE-EQ" → "2885"
        self._loaded = False
        self._load_date: Optional[str] = None

    # ──────────────────────────────────────────────
    # Loading + caching
    # ──────────────────────────────────────────────
    def _ensure_loaded(self):
        """Load instruments if not already loaded or if date changed."""
        today = date.today().isoformat()
        if self._loaded and self._load_date == today:
            return
        self._load_master(today)

    def _load_master(self, today: str):
        """Load from disk cache or download fresh."""
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _CACHE_DIR / f"scrip_master_{today}.json"

        instruments = None

        # Try disk cache first
        if cache_path.exists():
            try:
                instruments = json.loads(cache_path.read_text())
                logger.info(f"Symbol master loaded from cache: {len(instruments)} instruments")
            except Exception as e:
                logger.warning(f"Cache corrupted, re-downloading: {e}")
                cache_path.unlink(missing_ok=True)

        # Download if needed
        if instruments is None:
            instruments = self._download_master(cache_path)

        if not instruments:
            logger.error("Symbol master unavailable — ticker lookups will fail")
            return

        self._instruments = instruments
        self._build_indexes(instruments)
        self._loaded = True
        self._load_date = today

    def _download_master(self, cache_path: Path) -> list:
        """Download full OpenAPIScripMaster.json from Angel One."""
        logger.info("Downloading symbol master from Angel One...")
        try:
            resp = urllib.request.urlopen(_SCRIP_MASTER_URL, timeout=60)
            raw = resp.read().decode("utf-8")
            instruments = json.loads(raw)
            logger.info(f"Symbol master downloaded: {len(instruments)} instruments")

            # Save to disk cache
            try:
                cache_path.write_text(raw)
                logger.info(f"Cached to {cache_path}")
            except Exception as e:
                logger.warning(f"Cache write failed (non-fatal): {e}")

            # Also save a local copy for other tools (NSE CM format)
            self._save_nse_cm_extract(instruments)

            return instruments
        except Exception as e:
            logger.error(f"Symbol master download failed: {e}")
            return []

    def _save_nse_cm_extract(self, instruments: list):
        """Save NSE cash-market instruments as NSE_CM_sym_master.json for backward compat."""
        project_root = Path(__file__).resolve().parent.parent.parent
        nse_cm_path = project_root / "NSE_CM_sym_master.json"
        try:
            nse_cm = [i for i in instruments if i.get("exch_seg") == "NSE"]
            with open(nse_cm_path, "w") as f:
                json.dump(nse_cm, f)
            logger.info(f"NSE CM extract: {len(nse_cm)} instruments → {nse_cm_path}")
        except Exception as e:
            logger.warning(f"NSE CM extract failed (non-fatal): {e}")

    def _build_indexes(self, instruments: list):
        """Build O(1) lookup dicts from raw instrument list."""
        self._nse_by_name.clear()
        self._nse_by_token.clear()
        self._nfo_by_key.clear()
        self._all_by_key.clear()

        for inst in instruments:
            seg = inst.get("exch_seg", "")
            sym = inst.get("symbol", "")
            name = inst.get("name", "")
            tok = inst.get("token", "")

            if not (seg and sym and tok):
                continue

            # Universal key-based lookup
            self._all_by_key[f"{seg}:{sym}"] = tok

            # NSE equities: index by clean name
            if seg == "NSE" and sym.endswith("-EQ"):
                parsed = {
                    "token": tok,
                    "symbol": sym,
                    "name": name,
                    "exch_seg": seg,
                    "lot_size": int(inst.get("lotsize", "1")),
                    "tick_size": float(inst.get("tick_size", "0")),
                    "instrument_type": inst.get("instrumenttype", ""),
                    "expiry": inst.get("expiry", ""),
                }
                self._nse_by_name[name] = parsed
                self._nse_by_token[tok] = parsed

            # NFO instruments: index by full key for option lookups
            elif seg == "NFO":
                self._nfo_by_key[f"NFO:{sym}"] = inst

    # ──────────────────────────────────────────────
    # Token lookups
    # ──────────────────────────────────────────────
    def get_token(self, ticker: str, exchange: str = "NSE") -> Optional[str]:
        """
        Get Angel One token for a ticker.

        Args:
            ticker: "RELIANCE", "HDFCBANK", etc.
            exchange: "NSE" (default), "BSE", "NFO"

        Returns:
            Token string or None if not found.
        """
        self._ensure_loaded()

        if exchange == "NSE":
            info = self._nse_by_name.get(ticker)
            if info:
                return info["token"]
            # Fallback: try with -EQ suffix
            return self._all_by_key.get(f"NSE:{ticker}-EQ")

        return self._all_by_key.get(f"{exchange}:{ticker}")

    def get_info(self, ticker: str) -> Optional[dict]:
        """
        Get full instrument info for an NSE equity ticker.

        Returns:
            dict with: token, symbol, name, exch_seg, lot_size, tick_size, instrument_type
            None if not found.
        """
        self._ensure_loaded()
        return self._nse_by_name.get(ticker)

    def token_to_name(self, token: str, exchange: str = "NSE") -> Optional[str]:
        """Reverse lookup: token → ticker name."""
        self._ensure_loaded()
        if exchange == "NSE":
            info = self._nse_by_token.get(token)
            return info["name"] if info else None
        # Brute force for other exchanges
        for key, tok in self._all_by_key.items():
            if tok == token and key.startswith(f"{exchange}:"):
                return key.split(":", 1)[1]
        return None

    def exists(self, ticker: str, exchange: str = "NSE") -> bool:
        """Check if a ticker exists on the given exchange."""
        self._ensure_loaded()
        if exchange == "NSE":
            return ticker in self._nse_by_name
        return f"{exchange}:{ticker}" in self._all_by_key

    # ──────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────
    def search(self, query: str, exchange: str = "NSE", limit: int = 20) -> List[dict]:
        """
        Search tickers by partial name match.

        Returns list of dicts with: name, token, symbol
        """
        self._ensure_loaded()
        query_upper = query.upper()
        results = []

        if exchange == "NSE":
            for name, info in self._nse_by_name.items():
                if query_upper in name:
                    results.append(info)
                    if len(results) >= limit:
                        break
        else:
            prefix = f"{exchange}:"
            for key, tok in self._all_by_key.items():
                if key.startswith(prefix) and query_upper in key.upper():
                    results.append({"key": key, "token": tok})
                    if len(results) >= limit:
                        break

        return results

    # ──────────────────────────────────────────────
    # NFO option helpers
    # ──────────────────────────────────────────────
    def get_nfo_options(
        self, underlying: str, strike: int, expiry_str: str
    ) -> Dict[str, Tuple[str, str]]:
        """
        Find CE and PE tokens for an index option.

        Args:
            underlying: "NIFTY" or "BANKNIFTY"
            strike: 23200
            expiry_str: "17MAR26" (DDMMMYY)

        Returns:
            {"CE": (symbol, token), "PE": (symbol, token)}
            Missing legs are omitted.
        """
        self._ensure_loaded()

        # Use the existing find_option_token logic from options.data_service
        # but backed by our in-memory index for speed
        import re
        m = re.match(r'^(\d{1,2})([A-Z]{3})(\d{2,4})$', expiry_str.strip().upper())
        if not m:
            return {}

        day, mon, year = m.group(1), m.group(2), m.group(3)
        expected_expiry = f"{day.zfill(2)}{mon}{year[-2:]}"
        strike_str = str(strike)

        result = {}
        for option_type in ["CE", "PE"]:
            for key, inst in self._nfo_by_key.items():
                sym = inst.get("symbol", "")
                if (
                    inst.get("name") == underlying
                    and inst.get("instrumenttype") == "OPTIDX"
                    and option_type in sym
                    and strike_str in sym
                    and sym.startswith(underlying)
                ):
                    after_name = sym[len(underlying):]
                    strike_pos = after_name.find(strike_str)
                    if strike_pos > 0:
                        sym_expiry = after_name[:strike_pos]
                        sym_option = after_name[strike_pos + len(strike_str):]
                        if sym_option == option_type and sym_expiry.upper() == expected_expiry:
                            result[option_type] = (sym, inst.get("token", ""))
                            break

        return result

    def get_option_chain(
        self, underlying: str, expiry_str: str, strike_range: Tuple[int, int] = None, step: int = 50,
    ) -> List[dict]:
        """
        Get full option chain for an underlying + expiry.

        Args:
            underlying: "NIFTY"
            expiry_str: "17MAR26"
            strike_range: (low, high) — e.g. (23000, 23400). None = all.
            step: strike interval to filter (50 for NIFTY)

        Returns:
            List of {"strike": 23200, "CE": (sym, tok), "PE": (sym, tok)}
        """
        self._ensure_loaded()
        import re

        m = re.match(r'^(\d{1,2})([A-Z]{3})(\d{2,4})$', expiry_str.strip().upper())
        if not m:
            return []

        day, mon, year = m.group(1), m.group(2), m.group(3)
        expected_expiry = f"{day.zfill(2)}{mon}{year[-2:]}"

        # Collect all strikes for this expiry
        strikes: Dict[int, dict] = {}

        for key, inst in self._nfo_by_key.items():
            sym = inst.get("symbol", "")
            if (
                inst.get("name") != underlying
                or inst.get("instrumenttype") != "OPTIDX"
                or not sym.startswith(underlying)
            ):
                continue

            after_name = sym[len(underlying):]

            # Try to extract expiry and strike from symbol
            for ot in ["CE", "PE"]:
                if not sym.endswith(ot):
                    continue
                body = after_name[:-len(ot)]  # e.g. "17MAR2623200"
                # Match expiry prefix
                if not body.upper().startswith(expected_expiry):
                    continue
                strike_part = body[len(expected_expiry):]
                try:
                    strike_val = int(strike_part)
                except ValueError:
                    continue

                if strike_range and not (strike_range[0] <= strike_val <= strike_range[1]):
                    continue
                if strike_val % step != 0:
                    continue

                if strike_val not in strikes:
                    strikes[strike_val] = {"strike": strike_val}
                strikes[strike_val][ot] = (sym, inst.get("token", ""))
                break

        return sorted(strikes.values(), key=lambda x: x["strike"])

    # ──────────────────────────────────────────────
    # NIFTY 50 validation
    # ──────────────────────────────────────────────
    def validate_universe(self, tickers: List[str]) -> Dict[str, bool]:
        """
        Check which tickers in a list are valid NSE-EQ instruments.
        Useful for detecting stale/delisted tickers in the stock universe.

        Returns:
            {"RELIANCE": True, "TATAMOTORS": False, ...}
        """
        self._ensure_loaded()
        return {t: t in self._nse_by_name for t in tickers}

    def get_stats(self) -> dict:
        """Service health/stats."""
        self._ensure_loaded()
        return {
            "loaded": self._loaded,
            "load_date": self._load_date,
            "total_instruments": len(self._instruments or []),
            "nse_equities": len(self._nse_by_name),
            "nfo_instruments": len(self._nfo_by_key),
        }


# ── Module-level singleton ──
ticker_service = TickerService()
