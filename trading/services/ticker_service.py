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

            # NFO + BFO instruments: index by full key for option lookups
            # BFO = BSE F&O segment (SENSEX, BANKEX options)
            elif seg in ("NFO", "BFO"):
                self._nfo_by_key[f"{seg}:{sym}"] = inst

    # ──────────────────────────────────────────────
    # Token lookups
    # ──────────────────────────────────────────────
    # Hardcoded index-spot tokens — Angel One doesn't list these as -EQ
    # so the equity scrip-master lookup misses them. Centralising here
    # so any caller (cockpit service, screener, planner) that does
    # `get_token("NIFTY")` returns the correct token instead of None.
    INDEX_SPOT_TOKENS: dict[str, tuple[str, str]] = {
        # underlying → (token, exchange)
        "NIFTY":     ("99926000", "NSE"),
        "BANKNIFTY": ("99926009", "NSE"),
        "FINNIFTY":  ("99926037", "NSE"),
        "MIDCPNIFTY":("99926074", "NSE"),
        "INDIAVIX":  ("99926017", "NSE"),
        "SENSEX":    ("99919000", "BSE"),
        "BANKEX":    ("99919012", "BSE"),
    }

    # Every user-facing alias the UI / planner / screener might send →
    # the canonical name Angel One's scrip-master uses (`inst.name`).
    # Match is case-insensitive after a strip+collapse-whitespace pass.
    UNDERLYING_ALIASES: dict[str, str] = {
        "NIFTY":            "NIFTY",
        "NIFTY50":          "NIFTY",
        "NIFTY 50":         "NIFTY",
        "NSEI":             "NIFTY",
        "BANKNIFTY":        "BANKNIFTY",
        "BANK NIFTY":       "BANKNIFTY",
        "NIFTY BANK":       "BANKNIFTY",
        "NSEBANK":          "BANKNIFTY",
        "FINNIFTY":         "FINNIFTY",
        "FIN NIFTY":        "FINNIFTY",
        "NIFTY FIN SERVICE":"FINNIFTY",
        "MIDCPNIFTY":       "MIDCPNIFTY",
        "NIFTY MIDCAP":     "MIDCPNIFTY",
        "MIDCAP NIFTY":     "MIDCPNIFTY",
        "INDIAVIX":         "INDIAVIX",
        "INDIA VIX":        "INDIAVIX",
        "VIX":              "INDIAVIX",
        "SENSEX":           "SENSEX",
        "BSE SENSEX":       "SENSEX",
        "BANKEX":           "BANKEX",
        "BSE BANKEX":       "BANKEX",
    }

    @classmethod
    def normalize_underlying(cls, name: str) -> str:
        """User input → canonical underlying name used by Angel One.

        Strips, uppercases, collapses whitespace, and falls back to the
        cleaned input when no alias matches (so stock tickers like
        "HDFCBANK" still pass through unchanged).
        """
        if not name:
            return ""
        s = " ".join(str(name).upper().split())
        return cls.UNDERLYING_ALIASES.get(s, s.replace(" ", ""))

    @classmethod
    def resolve_exchange(cls, symbol: str) -> str:
        """Best-effort exchange picker. Caller should still pass exchange
        explicitly when known.

          SENSEX / BANKEX (spot or options) → BSE / BFO
          NIFTY / BANKNIFTY / FINNIFTY  (spot) → NSE
          *_FUT or NIFTY/BANKNIFTY weekly options → NFO
          everything else → NSE
        """
        s = (symbol or "").upper().strip()
        if not s:
            return "NSE"
        # Index spots — exact match
        if s in cls.INDEX_SPOT_TOKENS:
            return cls.INDEX_SPOT_TOKENS[s][1]
        # BSE family — options on BFO, spot on BSE
        if s.startswith(("SENSEX", "BANKEX")):
            return "BFO" if any(c.isdigit() for c in s) else "BSE"
        # Index options / futures — NFO
        if s.startswith(("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")) and any(c.isdigit() for c in s):
            return "NFO"
        # Stock futures / options — name + numeric strike → NFO
        if s.endswith(("FUT", "CE", "PE")):
            return "NFO"
        return "NSE"

    def get_token(self, ticker: str, exchange: str = "NSE") -> Optional[str]:
        """
        Get Angel One token for a ticker.

        Args:
            ticker: "RELIANCE", "HDFCBANK", "NIFTY", "SENSEX", …
            exchange: "NSE" (default), "BSE", "NFO", "BFO"

        Returns:
            Token string or None if not found.

        Falls back through:
          1. Hardcoded INDEX_SPOT_TOKENS for index symbols (any exchange)
          2. NSE name-index for plain equities
          3. `-EQ` suffix lookup
          4. Generic key lookup
        """
        self._ensure_loaded()
        t = (ticker or "").upper().strip()

        # 1) Index spot fallback — index names don't live in the -EQ
        # name-index, but every caller (planner / cockpit / screener)
        # expects get_token("NIFTY") to "just work".
        if t in self.INDEX_SPOT_TOKENS:
            return self.INDEX_SPOT_TOKENS[t][0]

        if exchange == "NSE":
            info = self._nse_by_name.get(t)
            if info:
                return info["token"]
            # Fallback: try with -EQ suffix
            return self._all_by_key.get(f"NSE:{t}-EQ")

        return self._all_by_key.get(f"{exchange}:{t}")

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

        Searches both NFO (NIFTY, BANKNIFTY) and BFO (SENSEX, BANKEX)
        instruments.

        Args:
            underlying: "NIFTY", "BANKNIFTY", or "SENSEX"
            strike: 23200
            expiry_str: "17MAR26" (DDMMMYY)

        Returns:
            {"CE": (symbol, token), "PE": (symbol, token)}
            Missing legs are omitted.
        """
        self._ensure_loaded()

        from datetime import datetime
        import re

        # Normalise the caller-supplied underlying so "NIFTY50" / "NIFTY 50"
        # / "NSEI" all collapse to "NIFTY" (the name in Angel One's NFO
        # scrip master). Was the root cause of "No option token found for
        # NIFTY50 …" — caller passed an alias, lookup filter saw no match.
        underlying = self.normalize_underlying(underlying)

        # Parse expiry string (DDMMMYY or DDMMMYYYY) to a date for matching
        m = re.match(r'^(\d{1,2})([A-Z]{3})(\d{2,4})$', expiry_str.strip().upper())
        if not m:
            return {}

        day, mon, year = m.group(1), m.group(2), m.group(3)
        year_full = f"20{year}" if len(year) == 2 else year
        try:
            target_date = datetime.strptime(f"{day.zfill(2)}{mon}{year_full}", "%d%b%Y").date()
        except ValueError:
            return {}

        # Angel One stores strike in paisa (78000 → 7800000.000000)
        strike_paisa = strike * 100.0

        result = {}
        for option_type in ["CE", "PE"]:
            for key, inst in self._nfo_by_key.items():
                sym = inst.get("symbol", "")
                # OPTIDX = index options (NIFTY/BANKNIFTY/SENSEX)
                # OPTSTK = stock options (HDFCBANK/RELIANCE/etc.)
                if (
                    inst.get("name") == underlying
                    and inst.get("instrumenttype") in ("OPTIDX", "OPTSTK")
                    and sym.endswith(option_type)
                ):
                    # Match strike from metadata (exact, no substring ambiguity)
                    try:
                        inst_strike = float(inst.get("strike", 0))
                    except (ValueError, TypeError):
                        continue
                    if abs(inst_strike - strike_paisa) > 1:
                        continue

                    # Match expiry date from metadata (works for both NSE and BSE symbol formats)
                    inst_expiry = inst.get("expiry", "")
                    try:
                        inst_date = datetime.strptime(inst_expiry, "%d%b%Y").date()
                    except (ValueError, TypeError):
                        continue
                    if inst_date == target_date:
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

        underlying = self.normalize_underlying(underlying)

        m = re.match(r'^(\d{1,2})([A-Z]{3})(\d{2,4})$', expiry_str.strip().upper())
        if not m:
            return []

        day, mon, year = m.group(1), m.group(2), m.group(3)
        expected_expiry = f"{day.zfill(2)}{mon}{year[-2:]}"

        # Collect all strikes for this expiry
        strikes: Dict[int, dict] = {}

        for key, inst in self._nfo_by_key.items():
            sym = inst.get("symbol", "")
            # Accept both OPTIDX (NIFTY/BANKNIFTY/SENSEX) and OPTSTK
            # (HDFCBANK/RELIANCE/…) — was OPTIDX-only and silently hid
            # every stock option from the chain view.
            if (
                inst.get("name") != underlying
                or inst.get("instrumenttype") not in ("OPTIDX", "OPTSTK")
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
    # Futures (FUTIDX + FUTSTK) — first-class lookup helpers
    # ──────────────────────────────────────────────
    def list_futures(self, underlying: str) -> List[dict]:
        """Return every future contract for `underlying`, sorted by expiry.

        Works for index futures (NIFTY, BANKNIFTY, FINNIFTY → FUTIDX) AND
        stock futures (HDFCBANK, RELIANCE → FUTSTK). Each row carries
        token + symbol + expiry (DDMMYYYY string) so callers can
        immediately invoke `broker.ltp("NFO", symbol, token)`.

        Empty list when scrip master has no contracts (off-cycle, off
        market).
        """
        self._ensure_loaded()
        from datetime import datetime
        u = self.normalize_underlying(underlying)
        out: list[dict] = []
        for key, inst in self._nfo_by_key.items():
            if inst.get("name") != u:
                continue
            if inst.get("instrumenttype") not in ("FUTIDX", "FUTSTK"):
                continue
            exp_str = inst.get("expiry", "")
            try:
                exp_date = datetime.strptime(exp_str, "%d%b%Y").date()
            except (ValueError, TypeError):
                exp_date = None
            out.append({
                "symbol": inst.get("symbol", ""),
                "token": inst.get("token", ""),
                "expiry": exp_str,
                "expiry_date": exp_date.isoformat() if exp_date else None,
                "lot_size": inst.get("lotsize"),
                "instrument_type": inst.get("instrumenttype"),
                "exch_seg": inst.get("exch_seg", "NFO"),
            })
        out.sort(key=lambda r: r["expiry_date"] or "")
        return out

    def get_future_token(self, underlying: str, *, month_offset: int = 0) -> Optional[dict]:
        """Resolve the {symbol, token, expiry, exchange} of a future contract.

        Args:
            underlying:    "NIFTY", "BANKNIFTY", "HDFCBANK", …
            month_offset:  0 = nearest live (default), 1 = next-month,
                           2 = far-month, etc.

        Returns None when no contract exists at the requested offset.
        """
        from datetime import date as dt_date
        contracts = [
            c for c in self.list_futures(underlying)
            if c["expiry_date"] and dt_date.fromisoformat(c["expiry_date"]) >= dt_date.today()
        ]
        if not contracts or month_offset < 0 or month_offset >= len(contracts):
            return None
        c = contracts[month_offset]
        return {
            "symbol": c["symbol"],
            "token": c["token"],
            "expiry": c["expiry"],
            "expiry_date": c["expiry_date"],
            "lot_size": c["lot_size"],
            "exchange": "NFO",   # futures live on NFO regardless of underlying
        }

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
