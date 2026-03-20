"""
Data Service — fetches and enriches market data from Angel One SmartAPI.

Reuses your existing BrokerClient and add_new_high_low_indicators logic.
"""
import os
import time
import threading
import pandas as pd
import pyotp
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from logzero import logger
from SmartApi import SmartConnect
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# Symbol master
# ──────────────────────────────────────────────
def load_symbol_master(file_path: str) -> list:
    """
    Load Angel One OpenAPI scrip master JSON.
    Returns raw list of instrument dicts.
    Format: [{"token": "2142", "symbol": "MFSL-EQ", "name": "MFSL",
              "exch_seg": "NSE", ...}, ...]
    """
    import json
    with open(file_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Legacy dict-of-dicts format — flatten
        return list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []
    return []


class TokenFetcher:
    """Map NSE ticker symbols to exchange tokens using Angel One scrip master."""

    def __init__(self, instruments: list):
        """
        Args:
            instruments: list of dicts from Angel One OpenAPI scrip master.
                         Each dict has keys: token, symbol, name, exch_seg, etc.
        """
        # Build lookup: "NSE:MFSL-EQ" → "2142"
        self._lookup = {}
        for inst in instruments:
            seg = inst.get("exch_seg", "")
            sym = inst.get("symbol", "")
            tok = inst.get("token", "")
            if seg and sym and tok:
                self._lookup[f"{seg}:{sym}"] = tok

        logger.info(f"TokenFetcher loaded {len(self._lookup)} instruments")

    def get_token(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        """
        Get token for a ticker.
        Accepts: 'MFSL' or 'NSE:MFSL-EQ'
        """
        # Normalize to 'NSE:SYMBOL-EQ' format
        if ":" not in symbol:
            sym_key = f"{exchange}:{symbol}-EQ"
        else:
            sym_key = symbol

        token = self._lookup.get(sym_key)
        if not token:
            logger.error(f"Token not found for {sym_key} (have {len(self._lookup)} instruments)")
        return token


# ──────────────────────────────────────────────
# Broker client
# ──────────────────────────────────────────────
class BrokerClient:
    """
    Centralized Angel One SmartAPI gateway.

    ALL broker API calls go through this class. It provides:
      - Single authenticated session (thread-safe login)
      - Global rate limiter (0.4s min gap between ANY API call)
      - TTL cache for ltpData (avoids redundant spot/LTP fetches)
      - Retry with backoff on rate-limit errors
      - Request counting for monitoring

    Use the module-level singleton: `from trading.services.data_service import broker`
    """

    # ── Class-level singleton ──
    _instance: Optional["BrokerClient"] = None
    _init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "BrokerClient":
        """Get or create the process-wide singleton."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.api_key = os.getenv("SMARTAPI_KEY")
        self.username = os.getenv("SMARTAPI_USERNAME")
        self.password = os.getenv("SMARTAPI_PASSWORD")
        self.totp_secret = os.getenv("SMARTAPI_TOTP_SECRET")
        self.smart_api = SmartConnect(self.api_key)
        self._logged_in = False

        # Rate limiting: min 0.4s between any API call
        self._last_call_time = 0.0
        self._rate_lock = threading.Lock()
        self._min_interval = 0.4  # seconds

        # LTP cache: {cache_key: (timestamp, result)}
        self._ltp_cache: Dict[str, tuple] = {}
        self._ltp_cache_ttl = 5  # seconds

        # Stats
        self._call_count = 0
        self._cache_hits = 0

        # Login lock
        self._login_lock = threading.Lock()

    def _throttle(self):
        """Enforce minimum interval between API calls (thread-safe)."""
        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_time = time.monotonic()
            self._call_count += 1

    def login(self) -> bool:
        """Authenticate with Angel One (thread-safe, idempotent)."""
        with self._login_lock:
            if self._logged_in:
                return True
            try:
                totp = pyotp.TOTP(self.totp_secret).now()
                self.smart_api.generateSession(self.username, self.password, totp)
                self._logged_in = True
                logger.info("Broker login successful.")
                return True
            except Exception as e:
                logger.error(f"Broker login failed: {e}")
                self._logged_in = False
                return False

    def ensure_login(self):
        """Login if not already logged in."""
        if not self._logged_in:
            self.login()

    @property
    def is_logged_in(self) -> bool:
        return self._logged_in

    # ──────────────────────────────────────────────
    # LTP Data (with built-in cache)
    # ──────────────────────────────────────────────
    def ltp(self, exchange: str, symbol: str, token: str) -> dict:
        """
        Fetch LTP data with automatic caching.

        Returns parsed dict: {ltp, open, high, low, prev_close} or {} on error.
        Cached for 5 seconds — safe to call frequently.
        """
        cache_key = f"ltp:{exchange}:{token}"
        now = time.monotonic()

        # Check cache
        entry = self._ltp_cache.get(cache_key)
        if entry is not None:
            ts, result = entry
            if now - ts < self._ltp_cache_ttl:
                self._cache_hits += 1
                return result

        # Fetch fresh
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.ltpData(exchange, symbol, token)
            result = self._parse_ltp(r)
            self._ltp_cache[cache_key] = (time.monotonic(), result)
            return result
        except Exception as e:
            logger.error(f"ltpData failed ({exchange}:{symbol}): {e}")
            return {}

    @staticmethod
    def _parse_ltp(r) -> dict:
        """Parse ltpData response. Guards against string error responses."""
        if not isinstance(r, dict):
            return {}
        d = r.get("data")
        if not isinstance(d, dict):
            return {}
        return {
            "ltp":        float(d.get("ltp", 0)),
            "open":       float(d.get("open", 0)),
            "high":       float(d.get("high", 0)),
            "low":        float(d.get("low", 0)),
            "prev_close": float(d.get("close", 0)),
        }

    # ──────────────────────────────────────────────
    # Candle Data
    # ──────────────────────────────────────────────
    def fetch_candles(
        self,
        symbol_token: str,
        start: str,
        end: str,
        interval: str = "FIVE_MINUTE",
        exchange: str = "NSE",
    ) -> List:
        """
        Fetch OHLCV candle data with retry on rate-limit errors.
        Dates in '%Y-%m-%d %H:%M' format.
        """
        self.ensure_login()
        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": start,
            "todate": end,
        }
        max_retries = 2
        for attempt in range(max_retries + 1):
            self._throttle()
            try:
                response = self.smart_api.getCandleData(params)
                if response is None:
                    if attempt < max_retries:
                        logger.warning(f"Rate limited on candles, retry {attempt+1}/{max_retries} in 2s...")
                        time.sleep(2)
                        continue
                    logger.warning("Candle fetch returned None after retries")
                    return []
                return response.get("data", []) or []
            except Exception as e:
                logger.exception(f"Candle fetch failed: {e}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return []
        return []

    # ──────────────────────────────────────────────
    # Portfolio / Orders
    # ──────────────────────────────────────────────
    def fetch_holdings(self) -> List[Dict]:
        """Fetch current portfolio holdings."""
        self.ensure_login()
        self._throttle()
        try:
            return self.smart_api.holding() or []
        except Exception as e:
            logger.error(f"Holdings fetch failed: {e}")
            return []

    def fetch_positions(self) -> Dict:
        """Fetch open positions."""
        self.ensure_login()
        self._throttle()
        try:
            return self.smart_api.position() or {}
        except Exception as e:
            logger.error(f"Positions fetch failed: {e}")
            return {}

    def fetch_order_book(self) -> list:
        """Fetch today's order book."""
        self.ensure_login()
        self._throttle()
        try:
            book = self.smart_api.orderBook()
            if isinstance(book, dict) and book.get("data"):
                return book["data"]
            return []
        except Exception as e:
            logger.error(f"Order book fetch failed: {e}")
            return []

    # ──────────────────────────────────────────────
    # Batch Market Data (THE key optimization)
    # ──────────────────────────────────────────────
    def market_data_batch(
        self,
        tokens: Dict[str, List[str]],
        mode: str = "OHLC",
    ) -> List[dict]:
        """
        Fetch market data for up to 50 instruments in ONE API call.

        This is 50x more efficient than calling ltpData() per stock.

        Args:
            tokens: {"NSE": ["2885", "1333"], "NFO": ["57710"]}
            mode: "LTP" (just price), "OHLC" (price + OHLC), "FULL" (everything)
                  FULL includes: volume, OI, 52wk hi/lo, circuit limits, bid/ask depth

        Returns:
            List of dicts with: exchange, tradingSymbol, symbolToken, ltp, open,
            high, low, close, percentChange, tradeVolume, opnInterest, etc.
        """
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.getMarketData(mode, tokens)
            if isinstance(r, dict) and r.get("status"):
                data = r.get("data", {})
                return data.get("fetched", [])
            return []
        except Exception as e:
            logger.error(f"Batch market data failed: {e}")
            return []

    # ──────────────────────────────────────────────
    # Option Greeks (real IV, delta, gamma from exchange)
    # ──────────────────────────────────────────────
    def option_greeks(self, underlying: str, expiry: str) -> List[dict]:
        """
        Fetch real option greeks from Angel One (only works during market hours).

        Args:
            underlying: "NIFTY" or "BANKNIFTY"
            expiry: "17MAR2026"

        Returns:
            List of dicts with: strikePrice, CE_delta, CE_gamma, CE_theta,
            CE_vega, CE_impliedVolatility, PE_* equivalents, etc.
        """
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.optionGreek({"name": underlying, "expirydate": expiry})
            if isinstance(r, dict) and r.get("data"):
                return r["data"] if isinstance(r["data"], list) else []
            return []
        except Exception as e:
            logger.error(f"Option greeks fetch failed: {e}")
            return []

    # ──────────────────────────────────────────────
    # OI Data (historical open interest)
    # ──────────────────────────────────────────────
    def oi_data(self, exchange: str, token: str, from_date: str, to_date: str) -> List[dict]:
        """
        Fetch historical open interest data.

        Args:
            exchange: "NFO"
            token: instrument token
            from_date: "2026-03-10 09:15"
            to_date: "2026-03-17 15:30"
        """
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.getOIData({
                "exchange": exchange,
                "symboltoken": token,
                "fromdate": from_date,
                "todate": to_date,
            })
            if isinstance(r, dict) and r.get("data"):
                return r["data"] if isinstance(r["data"], list) else []
            return []
        except Exception as e:
            logger.error(f"OI data fetch failed: {e}")
            return []

    # ──────────────────────────────────────────────
    # Margin / RMS Limits (real capital, not hardcoded)
    # ──────────────────────────────────────────────
    def margin_available(self) -> dict:
        """
        Fetch real-time margin/capital from broker.

        Returns:
            {net, availablecash, collateral, utilisedexposure, ...}
        """
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.rmsLimit()
            if isinstance(r, dict) and isinstance(r.get("data"), dict):
                return r["data"]
            return {}
        except Exception as e:
            logger.error(f"RMS limit fetch failed: {e}")
            return {}

    # ──────────────────────────────────────────────
    # Trade Book (today's executed trades)
    # ──────────────────────────────────────────────
    def fetch_trade_book(self) -> list:
        """Fetch all executed trades for today."""
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.tradeBook()
            if isinstance(r, dict) and r.get("data"):
                return r["data"] if isinstance(r["data"], list) else []
            return []
        except Exception as e:
            logger.error(f"Trade book fetch failed: {e}")
            return []

    # ──────────────────────────────────────────────
    # Search Scrip (live symbol search)
    # ──────────────────────────────────────────────
    def search_scrip(self, exchange: str, query: str) -> list:
        """
        Live symbol search by name.

        Args:
            exchange: "NSE", "NFO", "BSE"
            query: partial name like "RELIANCE"
        """
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.searchScrip(exchange, query)
            if isinstance(r, dict) and r.get("data"):
                return r["data"] if isinstance(r["data"], list) else []
            return []
        except Exception as e:
            logger.error(f"Search scrip failed: {e}")
            return []

    # ──────────────────────────────────────────────
    # Estimate Charges (brokerage + STT + stamp duty)
    # ──────────────────────────────────────────────
    def estimate_charges(self, orders: list) -> dict:
        """
        Estimate brokerage and charges for a list of orders.

        Args:
            orders: list of order param dicts
        """
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.estimateCharges({"orders": orders})
            if isinstance(r, dict):
                return r.get("data", {})
            return {}
        except Exception as e:
            logger.error(f"Estimate charges failed: {e}")
            return {}

    # ──────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────
    def get_stats(self) -> dict:
        return {
            "logged_in": self._logged_in,
            "api_calls": self._call_count,
            "cache_hits": self._cache_hits,
            "ltp_cache_size": len(self._ltp_cache),
        }


# ── Module-level singleton ──
broker = BrokerClient.get_instance()


# ──────────────────────────────────────────────
# Data enrichment (from your existing logic)
# ──────────────────────────────────────────────
def enrich_ohlcv(data: List) -> pd.DataFrame:
    """
    Enrich raw OHLCV candles with indicators.
    Input: list of [timestamp, open, high, low, close, volume]
    """
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame([dict(zip(cols, row)) for row in data])

    if df.empty:
        return df

    df["cumulative_high"] = df["high"].cummax()
    df["cumulative_low"] = df["low"].cummin()

    df["new_high"] = df["high"] == df["cumulative_high"]
    df["new_low"] = df["low"] == df["cumulative_low"]
    df.at[0, "new_high"] = False
    df.at[0, "new_low"] = False

    first_close = df["close"].iloc[0]
    first_open = df["open"].iloc[0]

    df["range"] = df["cumulative_high"] - df["cumulative_low"]
    df["range_percent"] = (df["range"] / first_close) * 100 if first_close else 0

    df["high_drawdown"] = 100 * (df["cumulative_high"] - df["close"]).abs() / first_open if first_open else 0
    df["low_drawdown"] = 100 * (df["cumulative_low"] - df["close"]).abs() / first_open if first_open else 0

    df["OH"] = df["open"] == df["high"]
    df["OL"] = df["open"] == df["low"]
    df["doji"] = (df["open"] - df["close"]).abs() < 0.10
    df["pivot"] = (df["low"] + df["high"]) / 2

    return df


# ──────────────────────────────────────────────
# High-level data service
# ──────────────────────────────────────────────
class DataService:
    """
    Top-level data service used by graph nodes.
    Uses the singleton BrokerClient for all API calls and
    ticker_service for symbol resolution.
    """

    def __init__(self):
        self._broker: Optional[BrokerClient] = None

    def _ensure_broker(self):
        if self._broker is None:
            self._broker = BrokerClient.get_instance()
        self._broker.ensure_login()

    def fetch_intraday(
        self,
        symbol: str,
        date: str,
        interval: str = "FIVE_MINUTE",
    ) -> Dict[str, Any]:
        """
        Fetch enriched intraday data for a symbol on a given date.

        Args:
            symbol: NSE symbol, e.g. 'MFSL'
            date: Date string '%Y-%m-%d'
            interval: FIVE_MINUTE, ONE_HOUR, etc.

        Returns:
            dict with keys: symbol, date, candle_count, last_close,
                            day_high, day_low, range_pct, summary
        """
        self._ensure_broker()

        from trading.services.ticker_service import ticker_service
        token = ticker_service.get_token(symbol)
        if not token:
            return {"error": f"Token not found for {symbol}", "symbol": symbol}

        start = f"{date} 09:15"
        end = f"{date} 15:30"

        raw = self._broker.fetch_candles(token, start, end, interval)
        if not raw:
            return {"error": "No candle data returned", "symbol": symbol}

        df = enrich_ohlcv(raw)

        # Build summary dict for the graph state
        last_row = df.iloc[-1]
        return {
            "symbol": symbol,
            "date": date,
            "candle_count": len(df),
            "open": float(df.iloc[0]["open"]),
            "last_close": float(last_row["close"]),
            "day_high": float(df["high"].max()),
            "day_low": float(df["low"].min()),
            "range_pct": float(last_row.get("range_percent", 0)),
            "new_highs": int(df["new_high"].sum()),
            "new_lows": int(df["new_low"].sum()),
            "doji_count": int(df["doji"].sum()),
            "pivot": float(last_row.get("pivot", 0)),
            "summary": self._build_text_summary(df, symbol, date),
        }

    def _build_text_summary(self, df: pd.DataFrame, symbol: str, date: str) -> str:
        """Build a text summary of market data for LLM context."""
        if df.empty:
            return f"No data available for {symbol} on {date}"

        last = df.iloc[-1]
        first = df.iloc[0]
        return (
            f"Market Data for {symbol} on {date}:\n"
            f"  Open: {first['open']:.2f} | Last Close: {last['close']:.2f}\n"
            f"  Day High: {df['high'].max():.2f} | Day Low: {df['low'].min():.2f}\n"
            f"  Range: {last.get('range', 0):.2f} ({last.get('range_percent', 0):.2f}%)\n"
            f"  New Highs: {df['new_high'].sum()} | New Lows: {df['new_low'].sum()}\n"
            f"  Candles: {len(df)} | Doji: {df['doji'].sum()}\n"
            f"  Pivot: {last.get('pivot', 0):.2f}\n"
            f"  Last 5 closes: {list(df['close'].tail(5).round(2))}\n"
            f"  Trend: {'Bullish' if last['close'] > first['open'] else 'Bearish'} "
            f"({((last['close'] - first['open']) / first['open'] * 100):.2f}%)"
        )

    def fetch_historical(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        interval: str = "FIVE_MINUTE",
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical candles across multiple days from Angel One broker.

        Angel One limits each API call to a single day for intraday intervals,
        so we loop day-by-day (skipping weekends) and merge results.

        Args:
            symbol: NSE symbol, e.g. 'MFSL'
            from_date: Start date '%Y-%m-%d'
            to_date: End date '%Y-%m-%d'
            interval: FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, ONE_DAY

        Returns:
            List of candle dicts: [{date, open, high, low, close, volume}, ...]
            For daily interval: one candle per day (OHLCV aggregated).
            For intraday intervals: individual candles per trading session.
        """
        self._ensure_broker()

        from trading.services.ticker_service import ticker_service
        token = ticker_service.get_token(symbol)
        if not token:
            logger.error(f"Token not found for {symbol}")
            return []

        start_dt = datetime.strptime(from_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(to_date, "%Y-%m-%d").date()

        all_candles: List[Dict[str, Any]] = []
        current = start_dt

        logger.info(
            f"Fetching historical data: {symbol} | {from_date} → {to_date} | interval={interval}"
        )

        # For ONE_DAY interval, we can fetch the whole range in one call
        if interval == "ONE_DAY":
            raw = self._broker.fetch_candles(
                token,
                f"{from_date} 09:15",
                f"{to_date} 15:30",
                interval,
            )
            for row in raw:
                ts, o, h, l, c, v = row[0], row[1], row[2], row[3], row[4], row[5]
                # Parse date from timestamp like "2026-02-20T00:00:00+05:30"
                candle_date = ts[:10] if isinstance(ts, str) else str(ts)
                all_candles.append({
                    "date": candle_date,
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": int(v),
                })
            logger.info(f"  ONE_DAY: fetched {len(all_candles)} daily candles")
            return all_candles

        # For intraday intervals, fetch day-by-day
        while current <= end_dt:
            # Skip weekends (Saturday=5, Sunday=6)
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            day_str = current.strftime("%Y-%m-%d")
            start_time = f"{day_str} 09:15"
            end_time = f"{day_str} 15:30"

            raw = self._broker.fetch_candles(token, start_time, end_time, interval)

            if raw:
                # Aggregate into a single daily candle (OHLCV summary for backtest)
                opens = [r[1] for r in raw]
                highs = [r[2] for r in raw]
                lows = [r[3] for r in raw]
                closes = [r[4] for r in raw]
                volumes = [r[5] for r in raw]

                all_candles.append({
                    "date": day_str,
                    "open": float(opens[0]),
                    "high": float(max(highs)),
                    "low": float(min(lows)),
                    "close": float(closes[-1]),
                    "volume": int(sum(volumes)),
                })
                logger.info(f"  {day_str}: {len(raw)} intraday candles → 1 daily candle")
            else:
                logger.warning(f"  {day_str}: no data (holiday or no trading)")

            current += timedelta(days=1)

        logger.info(f"Historical fetch complete: {len(all_candles)} trading days for {symbol}")
        return all_candles

    def fetch_multi_timeframe(
        self,
        symbol: str,
        date_str: str,
        intervals: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch candles at multiple timeframes for the same symbol/date.
        Used by scanning agents for multi-TF structure confirmation.

        Args:
            symbol: NSE ticker e.g. 'RELIANCE'
            date_str: Date string '%Y-%m-%d'
            intervals: List of intervals. Default: ['FIVE_MINUTE', 'FIFTEEN_MINUTE', 'ONE_HOUR']

        Returns:
            {"5m": [candles], "15m": [candles], "1h": [candles]}
        """
        if intervals is None:
            intervals = ["FIVE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR"]

        self._ensure_broker()
        from trading.services.ticker_service import ticker_service
        token = ticker_service.get_token(symbol)
        if not token:
            return {"error": f"Token not found for {symbol}"}

        # Cap end time for today
        from datetime import datetime as _dt
        now = _dt.now()
        if date_str == now.strftime("%Y-%m-%d"):
            if now.hour < 9 or (now.hour == 9 and now.minute < 16):
                return {"error": "Market not open yet"}
            end = f"{date_str} {min(now, now.replace(hour=15, minute=30)).strftime('%H:%M')}"
        else:
            end = f"{date_str} 15:30"

        interval_labels = {
            "ONE_MINUTE": "1m", "THREE_MINUTE": "3m", "FIVE_MINUTE": "5m",
            "TEN_MINUTE": "10m", "FIFTEEN_MINUTE": "15m", "THIRTY_MINUTE": "30m",
            "ONE_HOUR": "1h", "ONE_DAY": "1d",
        }

        result = {}
        for interval in intervals:
            label = interval_labels.get(interval, interval)
            raw = self._broker.fetch_candles(token, f"{date_str} 09:15", end, interval)
            result[label] = raw or []

        return result

    def fetch_batch_ltp(self, symbols: List[str]) -> List[dict]:
        """
        Fetch LTP for multiple symbols in ONE API call (up to 50).
        Returns list of dicts with: symbol, ltp, open, high, low, prev_close, volume, percentChange.
        """
        self._ensure_broker()
        from trading.services.ticker_service import ticker_service

        tokens = []
        token_map = {}
        for sym in symbols[:50]:  # API limit is 50
            tok = ticker_service.get_token(sym)
            if tok:
                tokens.append(tok)
                token_map[tok] = sym

        if not tokens:
            return []

        fetched = self._broker.market_data_batch({"NSE": tokens}, mode="FULL")

        results = []
        for item in fetched:
            tok = str(item.get("symbolToken", ""))
            sym = token_map.get(tok)
            if not sym:
                continue
            results.append({
                "symbol": sym,
                "ltp": float(item.get("ltp", 0)),
                "open": float(item.get("open", 0)),
                "high": float(item.get("high", 0)),
                "low": float(item.get("low", 0)),
                "prev_close": float(item.get("close", 0)),
                "volume": int(item.get("tradeVolume", 0)),
                "oi": int(item.get("opnInterest", 0)),
                "pct_change": float(item.get("percentChange", 0)),
                "low_52w": float(item.get("52WeekLow", 0)),
                "high_52w": float(item.get("52WeekHigh", 0)),
                "upper_circuit": float(item.get("upperCircuit", 0)),
                "lower_circuit": float(item.get("lowerCircuit", 0)),
            })

        return results

    def fetch_holdings(self) -> List[Dict]:
        """Fetch current broker holdings."""
        self._ensure_broker()
        return self._broker.fetch_holdings()

    def fetch_positions(self) -> Dict:
        """Fetch open positions."""
        self._ensure_broker()
        return self._broker.fetch_positions()
