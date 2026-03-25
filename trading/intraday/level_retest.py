"""
Level Break + Retest Detector — the second highest-edge intraday setup.

A break+retest occurs when:
  1. Price breaks through a significant level (closes beyond it)
  2. Then pulls back to retest the level (old resistance becomes new support, or vice versa)
  3. The retest holds (candle doesn't close back through the level)
  4. Enter in the breakout direction with SL at the level

This is the "polarity flip" — when resistance becomes support (or vice versa).
Higher probability than raw breakouts because the retest CONFIRMS the break was real.

Example: NIFTY breaks above PDH 23500, pulls back to 23510, holds → BUY
         SL just below 23500 (the flipped level), target = next resistance
"""
from typing import List, Optional
from trading.intraday.levels import LevelMap
from trading.intraday.state import IntradaySignal, SetupType, TradeBias


# ── Parameters from centralized config ──
from trading.config import config as _cfg

MIN_LEVEL_SCORE = _cfg.level_retest.min_level_score
BREAK_CONFIRMATION = _cfg.level_retest.break_confirmation
RETEST_PROXIMITY_ATR = _cfg.level_retest.retest_proximity_atr
RETEST_HOLD_CANDLES = _cfg.level_retest.retest_hold_candles
SL_BEYOND_LEVEL_ATR = _cfg.level_retest.sl_beyond_level_atr
MIN_RR = _cfg.level_retest.min_rr
MAX_SIGNALS = _cfg.level_retest.max_signals


def detect_level_retest(
    level_map: LevelMap,
    candles: list,
    atr: float,
    avg_volume: float = 0,
    current_price: float = 0,
) -> List[IntradaySignal]:
    """
    Detect level break + retest setups.

    Scans the last 20 candles (100 min) for:
      1. A level that was broken (price closed beyond it)
      2. A pullback to that level (price returned near it)
      3. A hold at the level (price didn't close back through)
    """
    if not candles or len(candles) < 10 or atr <= 0:
        return []

    signals = []
    n = len(candles)

    # Check each significant level for break + retest pattern
    for lvl in level_map.levels:
        if lvl.strength < MIN_LEVEL_SCORE:
            continue

        price = lvl.price

        # Scan last 20 candles for break + retest sequence
        scan_start = max(0, n - 20)

        # Phase 1: Find where price broke through this level
        break_idx = None
        break_direction = None  # "UP" or "DOWN"

        for i in range(scan_start, n - BREAK_CONFIRMATION - RETEST_HOLD_CANDLES):
            c_before = candles[max(0, i - 1)]
            c_break = candles[i]

            # Breakout UP: was below, closed above
            if _f(c_before, "close") < price and _f(c_break, "close") > price:
                # Confirm: next N candles also close above
                confirmed = all(
                    _f(candles[i + k], "close") > price
                    for k in range(1, BREAK_CONFIRMATION + 1)
                    if i + k < n
                )
                if confirmed:
                    break_idx = i
                    break_direction = "UP"

            # Breakout DOWN: was above, closed below
            elif _f(c_before, "close") > price and _f(c_break, "close") < price:
                confirmed = all(
                    _f(candles[i + k], "close") < price
                    for k in range(1, BREAK_CONFIRMATION + 1)
                    if i + k < n
                )
                if confirmed:
                    break_idx = i
                    break_direction = "DOWN"

        if break_idx is None:
            continue

        # Phase 2: Find pullback to the level (retest)
        retest_idx = None
        retest_zone = atr * RETEST_PROXIMITY_ATR

        for i in range(break_idx + BREAK_CONFIRMATION, n - RETEST_HOLD_CANDLES):
            c = candles[i]

            if break_direction == "UP":
                # After breaking UP, price pulls back DOWN toward the level
                if abs(_f(c, "low") - price) <= retest_zone:
                    # Does it hold? Next candles must close above the level
                    holds = all(
                        _f(candles[i + k], "close") > price
                        for k in range(1, RETEST_HOLD_CANDLES + 1)
                        if i + k < n
                    )
                    if holds:
                        retest_idx = i
                        break

            elif break_direction == "DOWN":
                # After breaking DOWN, price pulls back UP toward the level
                if abs(_f(c, "high") - price) <= retest_zone:
                    holds = all(
                        _f(candles[i + k], "close") < price
                        for k in range(1, RETEST_HOLD_CANDLES + 1)
                        if i + k < n
                    )
                    if holds:
                        retest_idx = i
                        break

        if retest_idx is None:
            continue

        # Only signal if the retest is recent (within last 6 candles)
        if n - retest_idx > 8:
            continue

        # Phase 3: Build the signal
        current = _f(candles[-1], "close")

        if break_direction == "UP":
            # Broke resistance, retested as support → BUY
            entry = current
            sl = price - atr * SL_BEYOND_LEVEL_ATR  # SL just below the flipped level
            target_lvl = level_map.find_resistance_above(entry)
            target = target_lvl.price if target_lvl else entry + (entry - sl) * 2.5

            risk = entry - sl
            if risk <= 0:
                continue
            rr = (target - entry) / risk
            if rr < MIN_RR:
                continue

            confidence = 0.65 + min(lvl.strength, 50) / 250
            signals.append(IntradaySignal(
                symbol="", token="",
                setup_type=SetupType.PDH_BREAK,  # reuse closest type
                bias=TradeBias.LONG, side="BUY",
                entry_price=round(entry, 2), stop_loss=round(sl, 2),
                target=round(target, 2), risk_reward=round(rr, 2),
                confidence=round(min(confidence, 0.95), 2),
                reason=f"Break+Retest: {lvl.source}={price:.0f} broken UP, retested and held. "
                       f"BUY {entry:.0f} SL {sl:.0f} TGT {target:.0f} R:R {rr:.1f}",
                near_pivot=lvl.source,
            ))

        else:
            # Broke support, retested as resistance → SELL
            entry = current
            sl = price + atr * SL_BEYOND_LEVEL_ATR
            target_lvl = level_map.find_support_below(entry)
            target = target_lvl.price if target_lvl else entry - (sl - entry) * 2.5

            risk = sl - entry
            if risk <= 0:
                continue
            rr = (entry - target) / risk
            if rr < MIN_RR:
                continue

            confidence = 0.65 + min(lvl.strength, 50) / 250
            signals.append(IntradaySignal(
                symbol="", token="",
                setup_type=SetupType.PDL_BREAK,  # reuse closest type
                bias=TradeBias.SHORT, side="SELL",
                entry_price=round(entry, 2), stop_loss=round(sl, 2),
                target=round(target, 2), risk_reward=round(rr, 2),
                confidence=round(min(confidence, 0.95), 2),
                reason=f"Break+Retest: {lvl.source}={price:.0f} broken DOWN, retested and held. "
                       f"SELL {entry:.0f} SL {sl:.0f} TGT {target:.0f} R:R {rr:.1f}",
                near_pivot=lvl.source,
            ))

    signals.sort(key=lambda s: -s.confidence)
    return signals[:MAX_SIGNALS]


def _f(candle, key) -> float:
    if isinstance(candle, dict):
        return float(candle.get(key, 0))
    idx = {"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4}.get(key, -1)
    if isinstance(candle, (list, tuple)) and 0 <= idx < len(candle):
        return float(candle[idx])
    return 0
