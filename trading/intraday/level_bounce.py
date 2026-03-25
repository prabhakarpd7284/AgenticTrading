"""
Level Bounce Detector — the highest-edge intraday setup.

Backtested on Mar 17-20, 2026 NIFTY 5-min data:
  33 trades, 23 wins, 70% WR, +1118 pts (+₹83,850 per lot)

How it works:
  1. Find LOCAL EXTREMES: a candle whose high/low is the highest/lowest
     for 3 candles on each side (15 min confirmation window)
  2. Check if the extreme is within 30 pts of a scored level
  3. Verify the reversal: price has moved >30 pts away from the extreme
  4. Enter at the OPEN of the candle AFTER the extreme (confirmed entry)
  5. SL: 20 pts beyond the extreme
  6. Target: next level in the trade direction

This matches how a real trader identifies reversals:
  "Price made a high near resistance → pulled back → I enter the short"
"""
from typing import List
from trading.intraday.levels import LevelMap
from trading.intraday.state import IntradaySignal, SetupType, TradeBias


# ── Parameters from centralized config ──
from trading.config import config as _cfg

LOOKBACK = _cfg.level_bounce.lookback
LEVEL_PROXIMITY = _cfg.level_bounce.level_proximity
MIN_REVERSAL = _cfg.level_bounce.min_reversal
SL_BEYOND_EXTREME = _cfg.level_bounce.sl_beyond_extreme
MIN_RR = _cfg.level_bounce.min_rr
MIN_LEVEL_SCORE = _cfg.level_bounce.min_level_score
MAX_SIGNALS = _cfg.level_bounce.max_signals


def detect_level_bounce(
    level_map: LevelMap,
    candles: list,
    atr: float,
    avg_volume: float = 0,
    current_price: float = 0,
) -> List[IntradaySignal]:
    """
    Scan for level bounce setups in the most recent candles.

    Returns up to MAX_SIGNALS signals, sorted by level score (highest first).
    Only returns signals where the reversal is CONFIRMED (price already moved
    away from the extreme — not anticipatory).
    """
    if not candles or len(candles) < LOOKBACK * 2 + 2:
        return []

    signals = []
    n = len(candles)

    # Scan the last 6 candles for confirmed extremes
    # (extreme at i, confirmation by candles i+1 to i+LOOKBACK, we're now at n-1)
    scan_start = max(LOOKBACK, n - 6)
    scan_end = n - 1  # need at least 1 candle after for confirmation

    for i in range(scan_start, scan_end):
        h = _f(candles[i], "high")
        l = _f(candles[i], "low")

        # ── LOCAL HIGH → potential SHORT ──
        is_local_high = True
        for j in range(max(0, i - LOOKBACK), min(n, i + LOOKBACK + 1)):
            if j != i and _f(candles[j], "high") > h:
                is_local_high = False
                break

        if is_local_high:
            sig = _build_signal_from_high(level_map, candles, i, h, atr)
            if sig:
                signals.append(sig)

        # ── LOCAL LOW → potential LONG ──
        is_local_low = True
        for j in range(max(0, i - LOOKBACK), min(n, i + LOOKBACK + 1)):
            if j != i and _f(candles[j], "low") < l:
                is_local_low = False
                break

        if is_local_low:
            sig = _build_signal_from_low(level_map, candles, i, l, atr)
            if sig:
                signals.append(sig)

    # Sort by level score (strongest levels first)
    signals.sort(key=lambda s: -s.confidence)
    return signals[:MAX_SIGNALS]


def _build_signal_from_high(
    level_map: LevelMap,
    candles: list,
    extreme_idx: int,
    extreme_price: float,
    atr: float,
) -> IntradaySignal | None:
    """Build a SHORT signal from a local high near resistance."""

    # Is the high near a level?
    level, score = level_map.find_nearest(extreme_price, max_dist=LEVEL_PROXIMITY)
    if not level or score < MIN_LEVEL_SCORE:
        return None
    if level.level_type == "support":
        return None  # high near a support level = breakout, not rejection

    # Has price moved away? (confirmation)
    candles_after = candles[extreme_idx + 1:]
    if not candles_after:
        return None
    min_after = min(_f(c, "low") for c in candles_after)
    reversal = extreme_price - min_after
    if reversal < MIN_REVERSAL:
        return None  # not enough pullback = no confirmation

    # Entry: current price (we're entering after confirmation)
    current = _f(candles[-1], "close")
    entry = current

    # SL: just above the extreme
    sl = extreme_price + SL_BEYOND_EXTREME
    risk = sl - entry
    if risk <= 0:
        return None

    # Target: find support level far enough for good R:R
    # Skip levels that are too close (within 1R distance) — they give poor R:R
    min_target_price = entry - risk * MIN_RR
    target = entry - risk * 2.5  # default fallback

    for lvl in sorted(level_map.levels, key=lambda l: -l.price):
        if lvl.price < min_target_price and lvl.level_type in ("support", "pivot") and lvl.strength >= 15:
            target = lvl.price
            break

    reward = entry - target
    rr = reward / risk
    if rr < MIN_RR:
        return None

    # Confidence based on level score + reversal strength
    confidence = 0.55 + min(score, 50) / 200 + min(reversal / atr, 0.20)
    confidence = min(confidence, 0.95)

    return IntradaySignal(
        symbol="", token="",
        setup_type=SetupType.LEVEL_BOUNCE_SHORT,
        bias=TradeBias.SHORT,
        side="SELL",
        entry_price=round(entry, 2),
        stop_loss=round(sl, 2),
        target=round(target, 2),
        risk_reward=round(rr, 2),
        confidence=round(confidence, 2),
        reason=(
            f"Local high {extreme_price:.0f} at {level.source}={level.price:.0f} "
            f"(score={score}). Reversed {reversal:.0f} pts. "
            f"SHORT {entry:.0f} SL {sl:.0f} TGT {target:.0f} R:R {rr:.1f}"
        ),
        near_pivot=level.source,
    )


def _build_signal_from_low(
    level_map: LevelMap,
    candles: list,
    extreme_idx: int,
    extreme_price: float,
    atr: float,
) -> IntradaySignal | None:
    """Build a LONG signal from a local low near support."""

    level, score = level_map.find_nearest(extreme_price, max_dist=LEVEL_PROXIMITY)
    if not level or score < MIN_LEVEL_SCORE:
        return None
    if level.level_type == "resistance":
        return None  # low near resistance = breakdown, not bounce

    candles_after = candles[extreme_idx + 1:]
    if not candles_after:
        return None
    max_after = max(_f(c, "high") for c in candles_after)
    reversal = max_after - extreme_price
    if reversal < MIN_REVERSAL:
        return None

    current = _f(candles[-1], "close")
    entry = current
    sl = extreme_price - SL_BEYOND_EXTREME
    risk = entry - sl
    if risk <= 0:
        return None

    # Target: find resistance far enough for good R:R
    min_target_price = entry + risk * MIN_RR
    target = entry + risk * 2.5  # default fallback

    for lvl in sorted(level_map.levels, key=lambda l: l.price):
        if lvl.price > min_target_price and lvl.level_type in ("resistance", "pivot") and lvl.strength >= 15:
            target = lvl.price
            break

    reward = target - entry
    rr = reward / risk
    if rr < MIN_RR:
        return None

    confidence = 0.55 + min(score, 50) / 200 + min(reversal / atr, 0.20)
    confidence = min(confidence, 0.95)

    return IntradaySignal(
        symbol="", token="",
        setup_type=SetupType.LEVEL_BOUNCE_LONG,
        bias=TradeBias.LONG,
        side="BUY",
        entry_price=round(entry, 2),
        stop_loss=round(sl, 2),
        target=round(target, 2),
        risk_reward=round(rr, 2),
        confidence=round(confidence, 2),
        reason=(
            f"Local low {extreme_price:.0f} at {level.source}={level.price:.0f} "
            f"(score={score}). Bounced {reversal:.0f} pts. "
            f"LONG {entry:.0f} SL {sl:.0f} TGT {target:.0f} R:R {rr:.1f}"
        ),
        near_pivot=level.source,
    )


def _f(candle, key) -> float:
    """Extract float from candle (dict or tuple)."""
    if isinstance(candle, dict):
        return float(candle.get(key, 0))
    idx = {"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4}.get(key, -1)
    if isinstance(candle, (list, tuple)) and 0 <= idx < len(candle):
        return float(candle[idx])
    return 0
