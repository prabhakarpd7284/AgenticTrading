"""
VWAP Fade Detector — mean reversion toward VWAP.

When price deviates significantly from VWAP and shows divergence
(RSI not confirming new extremes), fade back toward VWAP.

Best in RANGING markets. The regime filter should block this on trending days.

Entry: Price > 1% above VWAP + RSI divergence → SHORT toward VWAP
       Price > 1% below VWAP + RSI divergence → LONG toward VWAP
SL: Beyond the extreme
Target: VWAP
"""
from typing import List
from trading.intraday.levels import LevelMap
from trading.intraday.state import IntradaySignal, SetupType, TradeBias
from trading.utils.indicators import rsi as calc_rsi


# ── Parameters ──
MIN_DEVIATION_PCT = 0.8     # Price must be > 0.8% from VWAP
RSI_OVERBOUGHT = 65         # RSI above this + price far above VWAP = fade signal
RSI_OVERSOLD = 35           # RSI below this + price far below VWAP = fade signal
SL_BEYOND_EXTREME_ATR = 0.2 # SL beyond recent extreme
MIN_RR = 1.5
MAX_SIGNALS = 1


def detect_vwap_fade(
    level_map: LevelMap,
    candles: list,
    atr: float,
    avg_volume: float = 0,
    current_price: float = 0,
) -> List[IntradaySignal]:
    """
    Detect VWAP fade setups.

    Requires:
      - Price deviated > 0.8% from VWAP
      - RSI shows overextension (>65 for short, <35 for long)
      - VWAP is available (intraday only)
    """
    if not candles or len(candles) < 15 or atr <= 0:
        return []

    vwap = level_map.vwap
    if vwap <= 0:
        return []

    signals = []
    current = _f(candles[-1], "close")
    deviation_pct = (current - vwap) / vwap * 100

    # Compute RSI from closes
    closes = [_f(c, "close") for c in candles]
    current_rsi = calc_rsi(closes, 14)

    # Recent extreme (last 6 candles)
    recent = candles[-6:]
    recent_high = max(_f(c, "high") for c in recent)
    recent_low = min(_f(c, "low") for c in recent)

    # ── SHORT fade: price far above VWAP + RSI overbought ──
    if deviation_pct > MIN_DEVIATION_PCT and current_rsi > RSI_OVERBOUGHT:
        entry = current
        sl = recent_high + atr * SL_BEYOND_EXTREME_ATR
        target = vwap

        risk = sl - entry
        reward = entry - target
        if risk > 0 and reward > 0:
            rr = reward / risk
            if rr >= MIN_RR:
                confidence = 0.60 + min(abs(deviation_pct), 2.0) / 10  # +0 to +0.20
                signals.append(IntradaySignal(
                    symbol="", token="",
                    setup_type=SetupType.VWAP_REJECT,
                    bias=TradeBias.SHORT, side="SELL",
                    entry_price=round(entry, 2),
                    stop_loss=round(sl, 2),
                    target=round(target, 2),
                    risk_reward=round(rr, 2),
                    confidence=round(min(confidence, 0.90), 2),
                    reason=f"VWAP Fade SHORT: price {deviation_pct:+.1f}% above VWAP {vwap:.0f}, "
                           f"RSI={current_rsi:.0f} overbought. "
                           f"SELL {entry:.0f} SL {sl:.0f} TGT {target:.0f}",
                    near_pivot="VWAP",
                ))

    # ── LONG fade: price far below VWAP + RSI oversold ──
    elif deviation_pct < -MIN_DEVIATION_PCT and current_rsi < RSI_OVERSOLD:
        entry = current
        sl = recent_low - atr * SL_BEYOND_EXTREME_ATR
        target = vwap

        risk = entry - sl
        reward = target - entry
        if risk > 0 and reward > 0:
            rr = reward / risk
            if rr >= MIN_RR:
                confidence = 0.60 + min(abs(deviation_pct), 2.0) / 10
                signals.append(IntradaySignal(
                    symbol="", token="",
                    setup_type=SetupType.VWAP_RECLAIM,
                    bias=TradeBias.LONG, side="BUY",
                    entry_price=round(entry, 2),
                    stop_loss=round(sl, 2),
                    target=round(target, 2),
                    risk_reward=round(rr, 2),
                    confidence=round(min(confidence, 0.90), 2),
                    reason=f"VWAP Fade LONG: price {deviation_pct:+.1f}% below VWAP {vwap:.0f}, "
                           f"RSI={current_rsi:.0f} oversold. "
                           f"BUY {entry:.0f} SL {sl:.0f} TGT {target:.0f}",
                    near_pivot="VWAP",
                ))

    return signals[:MAX_SIGNALS]


def _f(candle, key) -> float:
    if isinstance(candle, dict):
        return float(candle.get(key, 0))
    idx = {"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4}.get(key, -1)
    if isinstance(candle, (list, tuple)) and 0 <= idx < len(candle):
        return float(candle[idx])
    return 0
