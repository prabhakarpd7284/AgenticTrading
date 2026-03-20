"""
Technical Indicators — pure math, no broker dependency.

Used by scanners, structure detectors, and analysis across all timeframes.
All functions accept lists of floats/dicts and return computed values.
"""
from typing import Dict, List, Optional, Tuple
import math


# ──────────────────────────────────────────────
# Camarilla Pivot Points
# ──────────────────────────────────────────────
def camarilla_pivots(high: float, low: float, close: float) -> Dict[str, float]:
    """
    Camarilla pivot points from previous day OHLC.

    Trading rules:
      - BUY near S3, stop at S4, target R1-R2
      - SELL near R3, stop at R4, target S1-S2
      - S4/R4 breakout = strong trend day, trade in direction

    Returns:
        {P, S1, S2, S3, S4, R1, R2, R3, R4}
    """
    diff = high - low
    p = (high + low + close) / 3

    return {
        "P":  round(p, 2),
        "S1": round(close - diff * 1.1 / 12, 2),
        "S2": round(close - diff * 1.1 / 6, 2),
        "S3": round(close - diff * 1.1 / 4, 2),
        "S4": round(close - diff * 1.1 / 2, 2),
        "R1": round(close + diff * 1.1 / 12, 2),
        "R2": round(close + diff * 1.1 / 6, 2),
        "R3": round(close + diff * 1.1 / 4, 2),
        "R4": round(close + diff * 1.1 / 2, 2),
    }


# ──────────────────────────────────────────────
# Bollinger Bands
# ──────────────────────────────────────────────
def bollinger_bands(
    closes: List[float], period: int = 20, num_std: float = 2.0
) -> Dict[str, float]:
    """
    Bollinger Bands with squeeze detection.

    Returns:
        {upper, middle, lower, bandwidth, squeeze}
        squeeze: True if bandwidth < 20-period average bandwidth * 0.75
    """
    if len(closes) < period:
        mid = sum(closes) / len(closes) if closes else 0
        return {"upper": mid, "middle": mid, "lower": mid, "bandwidth": 0, "squeeze": False}

    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(variance)

    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = ((upper - lower) / mid * 100) if mid > 0 else 0

    # Squeeze detection: compare current bandwidth to recent average
    squeeze = False
    if len(closes) >= period * 2:
        recent_bws = []
        for i in range(period):
            w = closes[-(period + i): len(closes) - i]
            if len(w) >= period:
                m = sum(w) / period
                v = sum((x - m) ** 2 for x in w) / period
                s = math.sqrt(v)
                bw = ((m + num_std * s) - (m - num_std * s)) / m * 100 if m > 0 else 0
                recent_bws.append(bw)
        if recent_bws:
            avg_bw = sum(recent_bws) / len(recent_bws)
            squeeze = bandwidth < avg_bw * 0.75

    return {
        "upper": round(upper, 2),
        "middle": round(mid, 2),
        "lower": round(lower, 2),
        "bandwidth": round(bandwidth, 2),
        "squeeze": squeeze,
    }


# ──────────────────────────────────────────────
# MACD
# ──────────────────────────────────────────────
def macd(
    closes: List[float],
    fast: int = 12, slow: int = 26, signal_period: int = 9,
) -> Dict[str, float]:
    """
    MACD (Moving Average Convergence Divergence).

    Returns:
        {macd, signal, histogram, crossover}
        crossover: "BULLISH" (MACD crosses above signal),
                   "BEARISH" (MACD crosses below signal),
                   "NONE"
    """
    if len(closes) < slow + signal_period:
        return {"macd": 0, "signal": 0, "histogram": 0, "crossover": "NONE"}

    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)

    macd_line = [f - s for f, s in zip(fast_ema[-len(slow_ema):], slow_ema)]
    signal_line = _ema(macd_line, signal_period)

    if not signal_line or not macd_line:
        return {"macd": 0, "signal": 0, "histogram": 0, "crossover": "NONE"}

    current_macd = macd_line[-1]
    current_signal = signal_line[-1]
    histogram = current_macd - current_signal

    # Crossover detection
    crossover = "NONE"
    if len(macd_line) >= 2 and len(signal_line) >= 2:
        prev_macd = macd_line[-2]
        prev_signal = signal_line[-2]
        if prev_macd <= prev_signal and current_macd > current_signal:
            crossover = "BULLISH"
        elif prev_macd >= prev_signal and current_macd < current_signal:
            crossover = "BEARISH"

    return {
        "macd": round(current_macd, 2),
        "signal": round(current_signal, 2),
        "histogram": round(histogram, 2),
        "crossover": crossover,
    }


# ──────────────────────────────────────────────
# RSI
# ──────────────────────────────────────────────
def rsi(closes: List[float], period: int = 14) -> float:
    """Standard RSI with Wilder smoothing. Returns 0-100."""
    if len(closes) < period + 1:
        return 50.0

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    gains = [max(0, d) for d in deltas[:period]]
    losses = [max(0, -d) for d in deltas[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(0, d)) / period
        avg_loss = (avg_loss * (period - 1) + max(0, -d)) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


# ──────────────────────────────────────────────
# VWAP
# ──────────────────────────────────────────────
def vwap(candles: List[dict]) -> float:
    """
    Volume Weighted Average Price from intraday candles.
    Candles: [{"high", "low", "close", "volume"}, ...]
    """
    cum_vol = 0
    cum_tp_vol = 0
    for c in candles:
        tp = (c["high"] + c["low"] + c["close"]) / 3
        vol = c.get("volume", 0)
        cum_tp_vol += tp * vol
        cum_vol += vol
    return round(cum_tp_vol / cum_vol, 2) if cum_vol > 0 else 0


# ──────────────────────────────────────────────
# ATR (Average True Range)
# ──────────────────────────────────────────────
def atr(candles: List[dict], period: int = 14) -> float:
    """
    ATR from candle dicts with {high, low, close}.
    Uses Wilder smoothing.
    """
    if len(candles) < 2:
        return candles[0]["high"] - candles[0]["low"] if candles else 0

    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))

    if not trs:
        return 0

    # Wilder smoothing
    atr_val = sum(trs[:period]) / period if len(trs) >= period else sum(trs) / len(trs)
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period

    return round(atr_val, 2)


# ──────────────────────────────────────────────
# EMA helper
# ──────────────────────────────────────────────
def _ema(data: List[float], period: int) -> List[float]:
    """Exponential Moving Average."""
    if len(data) < period:
        return data
    multiplier = 2 / (period + 1)
    ema_values = [sum(data[:period]) / period]
    for price in data[period:]:
        ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values


def ema(closes: List[float], period: int) -> Optional[float]:
    """Return latest EMA value."""
    result = _ema(closes, period)
    return round(result[-1], 2) if result else None


def sma(closes: List[float], period: int) -> Optional[float]:
    """Simple moving average — latest value."""
    if len(closes) < period:
        return None
    return round(sum(closes[-period:]) / period, 2)


# ──────────────────────────────────────────────
# Composite signal scorer
# ──────────────────────────────────────────────
def compute_indicator_confluence(
    closes: List[float],
    candles: List[dict] = None,
    prev_high: float = 0,
    prev_low: float = 0,
    prev_close: float = 0,
) -> Dict[str, any]:
    """
    Compute all indicators and return a confluence score.

    This is the single function scanners/monitors should call for
    a complete technical picture.

    Returns dict with all indicators + confluence score (0-100).
    """
    result = {}

    # Camarilla pivots (need previous day OHLC)
    if prev_high > 0 and prev_low > 0 and prev_close > 0:
        result["pivots"] = camarilla_pivots(prev_high, prev_low, prev_close)

    # Bollinger Bands
    if len(closes) >= 20:
        result["bb"] = bollinger_bands(closes)

    # MACD
    if len(closes) >= 35:
        result["macd"] = macd(closes)

    # RSI
    if len(closes) >= 15:
        result["rsi"] = rsi(closes)

    # VWAP
    if candles and len(candles) >= 3:
        result["vwap"] = vwap(candles)

    # ATR
    if candles and len(candles) >= 5:
        result["atr"] = atr(candles)

    # EMAs
    if len(closes) >= 20:
        result["ema_9"] = ema(closes, 9)
        result["ema_20"] = ema(closes, 20)

    # Confluence scoring
    score = 0
    bias_votes = {"LONG": 0, "SHORT": 0}
    current = closes[-1] if closes else 0

    # MACD signal
    if result.get("macd"):
        m = result["macd"]
        if m["crossover"] == "BULLISH":
            score += 15
            bias_votes["LONG"] += 2
        elif m["crossover"] == "BEARISH":
            score += 15
            bias_votes["SHORT"] += 2
        elif m["histogram"] > 0:
            bias_votes["LONG"] += 1
        elif m["histogram"] < 0:
            bias_votes["SHORT"] += 1

    # BB squeeze = high opportunity
    if result.get("bb", {}).get("squeeze"):
        score += 20

    # RSI extremes
    rsi_val = result.get("rsi", 50)
    if rsi_val > 70:
        bias_votes["SHORT"] += 1
    elif rsi_val < 30:
        bias_votes["LONG"] += 1

    # Price vs pivots
    pivots = result.get("pivots", {})
    if pivots and current > 0:
        if current <= pivots.get("S3", 0) * 1.005:
            score += 15
            bias_votes["LONG"] += 2  # Near S3 = buy zone
        elif current >= pivots.get("R3", float("inf")) * 0.995:
            score += 15
            bias_votes["SHORT"] += 2  # Near R3 = sell zone
        if current < pivots.get("S4", 0):
            score += 10  # S4 breakout = strong trend day
            bias_votes["SHORT"] += 1
        elif current > pivots.get("R4", float("inf")):
            score += 10
            bias_votes["LONG"] += 1

    # EMA alignment
    ema9 = result.get("ema_9")
    ema20 = result.get("ema_20")
    if ema9 and ema20:
        if ema9 > ema20 and current > ema9:
            score += 10
            bias_votes["LONG"] += 1
        elif ema9 < ema20 and current < ema9:
            score += 10
            bias_votes["SHORT"] += 1

    # Price vs VWAP
    vwap_val = result.get("vwap", 0)
    if vwap_val > 0 and current > 0:
        if current > vwap_val:
            bias_votes["LONG"] += 1
        else:
            bias_votes["SHORT"] += 1

    result["confluence_score"] = min(score, 100)
    result["confluence_bias"] = "LONG" if bias_votes["LONG"] > bias_votes["SHORT"] else \
                                 "SHORT" if bias_votes["SHORT"] > bias_votes["LONG"] else "NEUTRAL"
    result["bias_votes"] = bias_votes

    return result
