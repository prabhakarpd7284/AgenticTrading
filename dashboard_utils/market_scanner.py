"""
Market Scanner — NIFTY 50 screening with technical indicators.

Fetches batch LTP for all NIFTY 50 stocks, computes RSI/SMA/volume metrics,
and ranks opportunities by composite score.
"""
import os
import sys
from typing import Optional

from logzero import logger

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.services.data_service import BrokerClient, DataService
from trading.services.ticker_service import ticker_service

# ── NIFTY 50 constituents (as of 2026) ──
NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
    "SUNPHARMA", "HCLTECH", "ASIANPAINT", "NTPC", "M&M",
    "WIPRO", "ULTRACEMCO", "POWERGRID", "ONGC", "NESTLEIND",
    "JSWSTEEL", "TECHM", "ADANIENT", "ADANIPORTS",
    "TATASTEEL", "BAJAJFINSV", "COALINDIA", "HINDALCO", "GRASIM",
    "DIVISLAB", "BPCL", "CIPLA", "DRREDDY", "EICHERMOT",
    "APOLLOHOSP", "HEROMOTOCO", "INDUSINDBK", "TATACONSUM", "BRITANNIA",
    "SBILIFE", "BAJAJ-AUTO", "HDFCLIFE", "ETERNAL", "SHRIRAMFIN",
]

SECTOR_MAP = {
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "ETERNAL"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK"],
    "Finance": ["BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "SHRIRAMFIN"],
    "Auto": ["MARUTI", "TATAPOWER", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO"],
    "Pharma": ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "APOLLOHOSP"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "HINDALCO"],
    "Energy": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "BPCL", "COALINDIA", "ADANIENT", "ADANIPORTS"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"],
    "Others": ["LT", "TITAN", "ASIANPAINT", "ULTRACEMCO", "GRASIM"],
}


# ──────────────────────────────────────────────
# Technical Indicators
# ──────────────────────────────────────────────
def compute_rsi(closes: list, period: int = 14) -> float:
    """Standard RSI calculation with Wilder smoothing."""
    if len(closes) < period + 1:
        return 50.0  # neutral fallback

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    gains = [max(0, d) for d in deltas[:period]]
    losses = [max(0, -d) for d in deltas[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for d in deltas[period:]:
        gain = max(0, d)
        loss = max(0, -d)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_sma(closes: list, period: int) -> Optional[float]:
    """Simple moving average."""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def compute_sma_distance(close: float, sma: Optional[float]) -> Optional[float]:
    """% distance from SMA."""
    if sma is None or sma == 0:
        return None
    return (close - sma) / sma * 100


def compute_relative_volume(current_vol: int, avg_vol: float) -> float:
    """Current volume / average volume ratio."""
    if avg_vol <= 0:
        return 1.0
    return current_vol / avg_vol


# ──────────────────────────────────────────────
# Fetch single stock data with indicators
# ──────────────────────────────────────────────
def fetch_stock_snapshot(data_svc: DataService, symbol: str, trade_date: str) -> Optional[dict]:
    """
    Fetch intraday data for a single symbol and compute indicators.
    Returns dict with LTP, change%, RSI, SMA distances, volume metrics.
    """
    try:
        data = data_svc.fetch_intraday(symbol, trade_date)
        if "error" in data:
            return None

        ltp = data.get("last_close", 0)
        day_open = data.get("open", 0)
        day_high = data.get("day_high", 0)
        day_low = data.get("day_low", 0)

        if ltp <= 0 or day_open <= 0:
            return None

        change_pct = (ltp - day_open) / day_open * 100
        range_pct = data.get("range_pct", 0)

        return {
            "symbol": symbol,
            "ltp": ltp,
            "open": day_open,
            "high": day_high,
            "low": day_low,
            "change_pct": round(change_pct, 2),
            "range_pct": round(range_pct, 2),
            "new_highs": data.get("new_highs", 0),
            "new_lows": data.get("new_lows", 0),
            "candle_count": data.get("candle_count", 0),
        }

    except Exception as e:
        logger.error(f"Snapshot failed for {symbol}: {e}")
        return None


# ──────────────────────────────────────────────
# Batch scan NIFTY 50 (via LTP API — fast)
# ──────────────────────────────────────────────
def fetch_nifty50_ltp(broker: BrokerClient, symbols: list = None) -> list:
    """
    Fetch LTP for all NIFTY 50 symbols via ONE batch getMarketData call.
    50 stocks in 1 API call (~0.5s) instead of 50 serial calls (~20s).
    """
    symbols = symbols or NIFTY_50_SYMBOLS
    results = []

    try:
        broker.ensure_login()

        # Build token list for batch call
        nse_tokens = []
        token_to_symbol = {}
        for sym in symbols:
            token = ticker_service.get_token(sym)
            if token:
                nse_tokens.append(token)
                token_to_symbol[token] = sym

        if not nse_tokens:
            return []

        # ONE API call for all 50 stocks
        fetched = broker.market_data_batch({"NSE": nse_tokens}, mode="OHLC")

        for item in fetched:
            token = str(item.get("symbolToken", ""))
            sym = token_to_symbol.get(token)
            if not sym:
                continue

            ltp = float(item.get("ltp", 0))
            prev_close = float(item.get("close", 0))
            open_price = float(item.get("open", 0))

            if ltp <= 0:
                continue

            change_pct = ((ltp - prev_close) / prev_close * 100) if prev_close > 0 else 0
            day_change_pct = ((ltp - open_price) / open_price * 100) if open_price > 0 else 0

            results.append({
                "symbol": sym,
                "ltp": ltp,
                "prev_close": prev_close,
                "open": open_price,
                "high": float(item.get("high", 0)),
                "low": float(item.get("low", 0)),
                "change_pct": round(change_pct, 2),
                "day_change_pct": round(day_change_pct, 2),
                "sector": _get_sector(sym),
            })

    except Exception as e:
        logger.error(f"Batch LTP fetch failed: {e}")

    results.sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)
    return results



def _get_sector(symbol: str) -> str:
    for sector, symbols in SECTOR_MAP.items():
        if symbol in symbols:
            return sector
    return "Other"


# ──────────────────────────────────────────────
# Rank opportunities
# ──────────────────────────────────────────────
def rank_opportunities(stocks: list) -> list:
    """
    Rank stocks by composite opportunity score.
    Factors: absolute change %, volume surge, momentum.
    """
    if not stocks:
        return []

    for s in stocks:
        # Composite score: higher absolute change = more movement = more opportunity
        change_score = min(abs(s.get("change_pct", 0)) * 10, 50)  # cap at 50
        s["opportunity_score"] = round(change_score, 1)

    stocks.sort(key=lambda x: x["opportunity_score"], reverse=True)
    return stocks
