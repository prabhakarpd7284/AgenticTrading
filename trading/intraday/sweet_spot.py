"""
Sweet Spot Filter — enrich any signal with level awareness.

Takes a raw signal from any detector (ORB, PDH, PDL, GAP, VWAP, Level Bounce)
and verifies it's happening at a significant level. Rejects random noise,
boosts high-confluence setups, and improves SL/target placement using levels.

The spec says:
  "A PDH break at a random level is noise.
   A PDH break that aligns with a weekly resistance + Camarilla R3 = high probability."

Usage:
    from trading.intraday.sweet_spot import evaluate_signal
    enriched = evaluate_signal(signal, level_map, atr)
    if enriched is None:
        pass  # rejected — not at a significant level
"""
from typing import Optional, List
from trading.intraday.levels import LevelMap
from trading.intraday.state import IntradaySignal


# ── Minimum level score to accept a signal ──
MIN_LEVEL_SCORE_FOR_SIGNAL = 20   # reject if no level near entry with score >= 20


def evaluate_signal(
    signal: IntradaySignal,
    level_map: LevelMap,
    atr: float,
) -> Optional[IntradaySignal]:
    """
    Enrich a signal with level context. Returns None if signal is at no significant level.

    Modifications:
      - Rejects if entry not near any level scoring >= 20
      - Boosts confidence if at high-confluence level
      - Improves SL to nearest level behind entry (tighter, real-world S/R)
      - Improves target to next level in trade direction
      - Recalculates R:R with improved SL/target
    """
    # Find nearest level to entry price
    nearest, score = level_map.find_nearest(signal.entry_price, max_dist=atr * 0.5)

    if not nearest or score < MIN_LEVEL_SCORE_FOR_SIGNAL:
        return None  # Not at a significant level — reject

    # ── Boost confidence based on level score ──
    # Score 20 = +0%, Score 40 = +10%, Score 60 = +20%, Score 80+ = +25%
    level_boost = min(score, 80) / 400  # max +0.20
    signal.confidence = min(signal.confidence + level_boost, 0.95)

    # ── Improve SL using levels ──
    if signal.side == "BUY":
        # For longs: find support below entry for a better SL
        support = level_map.find_support_below(signal.entry_price)
        if support:
            level_sl = support.price - atr * 0.05  # tiny buffer below level
            # Only use if tighter than current SL
            if level_sl > signal.stop_loss and level_sl < signal.entry_price:
                signal.stop_loss = level_sl
    else:
        # For shorts: find resistance above entry for a better SL
        resistance = level_map.find_resistance_above(signal.entry_price)
        if resistance:
            level_sl = resistance.price + atr * 0.05
            if level_sl < signal.stop_loss and level_sl > signal.entry_price:
                signal.stop_loss = level_sl

    # ── Improve target using levels ──
    if signal.side == "BUY":
        next_resistance = level_map.find_resistance_above(signal.entry_price)
        if next_resistance and next_resistance.price > signal.entry_price:
            signal.target = next_resistance.price
    else:
        next_support = level_map.find_support_below(signal.entry_price)
        if next_support and next_support.price < signal.entry_price:
            signal.target = next_support.price

    # ── Recalculate R:R ──
    risk = abs(signal.entry_price - signal.stop_loss)
    reward = abs(signal.target - signal.entry_price)
    if risk > 0:
        signal.risk_reward = round(reward / risk, 2)

    # ── Add level info ──
    signal.near_pivot = nearest.source

    return signal


def apply_confirmation_checklist(
    signal: IntradaySignal,
    volume_ok: bool,
    candle_pattern_ok: bool,
    indicator_confluence: int,
    at_level: bool,
    regime_supports: bool,
) -> bool:
    """
    Spec V2 confirmation checklist: require at least 3 of 5.

    Returns True if the signal passes, False if rejected.
    """
    checks = [
        volume_ok,           # Volume > 1.2x avg
        candle_pattern_ok,   # Engulfing, pin bar, or strong close
        indicator_confluence >= 40,  # RSI + MACD + BB alignment
        at_level,            # Level score >= 20
        regime_supports,     # Regime matches setup direction
    ]

    passed = sum(checks)
    return passed >= 3
