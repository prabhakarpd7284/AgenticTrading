"""
Data Service — fetches and enriches market data from Angel One SmartAPI.

Reuses your existing BrokerClient and add_new_high_low_indicators logic.
"""
import os
import time
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
    """Wraps SmartConnect: authentication + data retrieval."""

    def __init__(self):
        self.api_key = os.getenv("SMARTAPI_KEY")
        self.username = os.getenv("SMARTAPI_USERNAME")
        self.password = os.getenv("SMARTAPI_PASSWORD")
        self.totp_secret = os.getenv("SMARTAPI_TOTP_SECRET")
        self.smart_api = SmartConnect(self.api_key)
        self._logged_in = False

    def login(self) -> bool:
        """Authenticate with Angel One."""
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

    @property
    def is_logged_in(self) -> bool:
        return self._logged_in

    def fetch_candles(
        self,
        symbol_token: str,
        start: str,
        end: str,
        interval: str = "FIVE_MINUTE",
        exchange: str = "NSE",
    ) -> List:
        """
        Fetch OHLCV candle data.
        Dates in '%Y-%m-%d %H:%M' format.
        """
        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": start,
            "todate": end,
        }
        try:
            response = self.smart_api.getCandleData(params)
            time.sleep(0.2)  # rate-limit guard
            return response.get("data", [])
        except Exception as e:
            logger.exception(f"Candle fetch failed: {e}")
            return []

    def fetch_holdings(self) -> List[Dict]:
        """Fetch current portfolio holdings."""
        try:
            return self.smart_api.holding() or []
        except Exception as e:
            logger.error(f"Holdings fetch failed: {e}")
            return []

    def fetch_positions(self) -> Dict:
        """Fetch open positions."""
        try:
            return self.smart_api.position() or {}
        except Exception as e:
            logger.error(f"Positions fetch failed: {e}")
            return {}


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
    Handles broker auth, token lookup, data fetch + enrichment.
    """

    def __init__(self):
        self._broker: Optional[BrokerClient] = None
        self._token_fetcher: Optional[TokenFetcher] = None

    def _ensure_broker(self):
        if self._broker is None:
            self._broker = BrokerClient()
        if not self._broker.is_logged_in:
            self._broker.login()

    def _ensure_tokens(self):
        if self._token_fetcher is not None:
            return

        master_path = os.getenv("SYMBOL_MASTER_JSON", "")

        # Fallback: look for local file in project dir
        if not master_path or not os.path.exists(master_path):
            local_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "NSE_CM_sym_master.json",
            )
            if os.path.exists(local_path):
                master_path = local_path
                logger.info(f"Using local symbol master: {local_path}")

        # Fallback: auto-download from Angel One
        if not master_path or not os.path.exists(master_path):
            try:
                import urllib.request, json
                url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
                logger.info(f"Downloading symbol master from Angel One...")
                resp = urllib.request.urlopen(url, timeout=30)
                data = json.loads(resp.read().decode("utf-8"))
                download_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "NSE_CM_sym_master.json",
                )
                with open(download_path, "w") as f:
                    json.dump(data, f)
                master_path = download_path
                logger.info(f"Symbol master downloaded: {len(data)} instruments → {download_path}")
            except Exception as e:
                logger.error(f"Failed to download symbol master: {e}")
                return

        if master_path and os.path.exists(master_path):
            instruments = load_symbol_master(master_path)
            self._token_fetcher = TokenFetcher(instruments)
        else:
            logger.warning("Symbol master not available")

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
        self._ensure_tokens()

        if self._token_fetcher is None:
            return {"error": "Symbol master not loaded", "symbol": symbol}

        token = self._token_fetcher.get_token(symbol)
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
        self._ensure_tokens()

        if self._token_fetcher is None:
            logger.error("Symbol master not loaded — cannot fetch historical data")
            return []

        token = self._token_fetcher.get_token(symbol)
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

    def fetch_holdings(self) -> List[Dict]:
        """Fetch current broker holdings."""
        self._ensure_broker()
        return self._broker.fetch_holdings()

    def fetch_positions(self) -> Dict:
        """Fetch open positions."""
        self._ensure_broker()
        return self._broker.fetch_positions()
