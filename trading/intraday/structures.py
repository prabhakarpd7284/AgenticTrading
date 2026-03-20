"""
Price Structure Detectors — the edge.

Each detector is pure math. No opinions, no LLM. Just price, volume, and levels.
These structures have been proven over decades of intraday trading:

1. ORB (Opening Range Breakout) — First 15 min sets the battlefield
2. PDH/PDL Break — Yesterday's extremes are today's battleground
3. Gap Analysis — Gaps create imbalance; trade the resolution
4. VWAP — Institutional average price; the gravity line

Rules:
  - Every signal MUST have a defined stop loss (structure invalidation)
  - Every signal MUST have risk:reward >= 1.5
  - Volume must confirm the move (no low-volume breakouts)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from logzero import logger

from trading.intraday.state import (
    IntradaySignal,
    SetupType,
    StockSetup,
    TradeBias,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _round(v: float, decimals: int = 2) -> float:
    return round(v, decimals)


def _risk_reward(entry: float, sl: float, target: float, side: str) -> float:
    """Calculate risk:reward ratio."""
    if side == "BUY":
        risk = abs(entry - sl)
        reward = abs(target - entry)
    else:
        risk = abs(sl - entry)
        reward = abs(entry - target)
    return round(reward / risk, 2) if risk > 0 else 0.0


def _gap_pct(setup: "StockSetup") -> float:
    """
    Calculate today's opening gap from prev close.
    Positive = gap up, negative = gap down.
    """
    if setup.today_open > 0 and setup.prev_close > 0:
        return (setup.today_open - setup.prev_close) / setup.prev_close * 100
    return 0.0


# Gap-aware confidence adjustment
# Backtest showed PDH_BREAK longs on gap-up days have 50% lower win rate
GAP_THRESHOLD_PCT = 0.5  # gaps > 0.5% are significant


# ──────────────────────────────────────────────
# 1. Opening Range Breakout (ORB)
# ──────────────────────────────────────────────

def detect_orb(
    setup: StockSetup,
    candles_5min: List[Dict],
    volume_avg: float = 0.0,
) -> Optional[IntradaySignal]:
    """
    Opening Range Breakout — first 15 minutes define the range.

    LONG: price breaks above ORB high with volume
    SHORT: price breaks below ORB low with volume

    Entry: breakout candle close
    SL: opposite end of opening range
    Target: 1.5x the range from entry

    Args:
        setup: StockSetup with orb_high, orb_low populated
        candles_5min: All 5-min candles so far today
        volume_avg: Average 5-min volume for comparison
    """
    if not candles_5min or len(candles_5min) < 4:
        return None  # Need at least 4 candles (3 for ORB + 1 for breakout)

    # ORB = first 3 five-minute candles (9:15–9:30)
    orb_candles = candles_5min[:3]
    orb_high = max(c["high"] for c in orb_candles)
    orb_low = min(c["low"] for c in orb_candles)
    orb_range = orb_high - orb_low

    if orb_range <= 0:
        return None

    # Update setup with ORB levels
    setup.orb_high = orb_high
    setup.orb_low = orb_low

    # Check candles AFTER the opening range for breakout
    for candle in candles_5min[3:]:
        close = candle["close"]
        high = candle["high"]
        low = candle["low"]
        vol = candle.get("volume", 0)

        # Volume confirmation: current candle volume > 1.2x average
        vol_ok = vol > (volume_avg * 1.2) if volume_avg > 0 else True

        # LONG breakout: close above ORB high
        if close > orb_high and vol_ok:
            entry = _round(close)
            sl = _round(orb_low)
            target = _round(entry + (orb_range * 1.5))
            rr = _risk_reward(entry, sl, target, "BUY")

            if rr >= 1.5:
                return IntradaySignal(
                    symbol=setup.symbol,
                    token=setup.token,
                    setup_type=SetupType.ORB_LONG,
                    bias=TradeBias.LONG,
                    entry_price=entry,
                    stop_loss=sl,
                    target=target,
                    risk_reward=rr,
                    confidence=min(0.9, 0.6 + (rr - 1.5) * 0.1),
                    reason=(
                        f"ORB Long: price {close:.2f} broke above ORB high {orb_high:.2f} "
                        f"(range {orb_range:.2f}). SL at ORB low {orb_low:.2f}. "
                        f"Target {target:.2f} (1.5x range). R:R {rr}"
                    ),
                    timestamp=candle.get("timestamp", ""),
                )

        # SHORT breakout: close below ORB low
        if close < orb_low and vol_ok:
            entry = _round(close)
            sl = _round(orb_high)
            target = _round(entry - (orb_range * 1.5))
            rr = _risk_reward(entry, sl, target, "SELL")

            if rr >= 1.5:
                return IntradaySignal(
                    symbol=setup.symbol,
                    token=setup.token,
                    setup_type=SetupType.ORB_SHORT,
                    bias=TradeBias.SHORT,
                    entry_price=entry,
                    stop_loss=sl,
                    target=target,
                    risk_reward=rr,
                    confidence=min(0.9, 0.6 + (rr - 1.5) * 0.1),
                    reason=(
                        f"ORB Short: price {close:.2f} broke below ORB low {orb_low:.2f} "
                        f"(range {orb_range:.2f}). SL at ORB high {orb_high:.2f}. "
                        f"Target {target:.2f} (1.5x range). R:R {rr}"
                    ),
                    timestamp=candle.get("timestamp", ""),
                )

    return None


# ──────────────────────────────────────────────
# 2. Previous Day High / Low Break
# ──────────────────────────────────────────────

def detect_pdh_pdl(
    setup: StockSetup,
    candles_5min: List[Dict],
    volume_avg: float = 0.0,
) -> Optional[IntradaySignal]:
    """
    Previous Day High/Low Break — yesterday's extremes are key levels.

    PDH Break (Long): price closes above previous day high with volume
    PDL Break (Short): price closes below previous day low with volume

    Entry: breakout candle close
    SL: PDH/PDL level itself (if price re-enters, structure is invalid)
    Target: ATR-based or 1.5x risk from entry
    """
    if not candles_5min or len(candles_5min) < 4:
        return None  # Wait for ORB to complete first

    pdh = setup.prev_high
    pdl = setup.prev_low
    atr = setup.prev_atr if setup.prev_atr > 0 else (pdh - pdl)

    if pdh <= 0 or pdl <= 0 or pdh <= pdl:
        return None

    # Only check candles after ORB (first 15 min)
    for candle in candles_5min[3:]:
        close = candle["close"]
        vol = candle.get("volume", 0)
        vol_ok = vol > (volume_avg * 1.0) if volume_avg > 0 else True

        # PDH breakout → Long
        if close > pdh and vol_ok:
            entry = _round(close)

            # Smart SL: use Camarilla R3 as support (just broken through), or pivot P
            # Minimum SL distance = 0.15% of entry (avoid noise stops)
            min_sl_distance = entry * 0.0015
            atr_sl = _round(pdh - atr * 0.3)  # ATR-based fallback

            if setup.pivot_r3 > 0 and setup.pivot_r3 < entry and (entry - setup.pivot_r3) >= min_sl_distance:
                sl = _round(setup.pivot_r3)  # R3 becomes support after breakout
            elif setup.pivot_p > 0 and setup.pivot_p < entry:
                sl = _round(max(setup.pivot_p, pdh - atr * 0.3))
            else:
                sl = _round(pdh - atr * 0.3)

            # Smart target: next Camarilla resistance (R4) or ATR-based
            if setup.pivot_r4 > 0 and setup.pivot_r4 > entry:
                target = _round(setup.pivot_r4)
            else:
                target = _round(entry + atr * 1.0)

            rr = _risk_reward(entry, sl, target, "BUY")

            if rr >= 1.5:
                gap = _gap_pct(setup)
                base_conf = min(0.85, 0.55 + (rr - 1.5) * 0.1)

                # Boost confidence if entry is near Camarilla R3 (validated level)
                if setup.pivot_r3 > 0 and abs(entry - setup.pivot_r3) / entry * 100 < 0.5:
                    base_conf = min(0.90, base_conf + 0.05)

                if gap > GAP_THRESHOLD_PCT:
                    base_conf *= 0.7
                    gap_note = f" [gap {gap:+.1f}%]"
                else:
                    gap_note = ""

                pivot_note = f" SL@R3={setup.pivot_r3:.0f}" if setup.pivot_r3 > 0 and sl == _round(setup.pivot_r3) else ""

                return IntradaySignal(
                    symbol=setup.symbol, token=setup.token,
                    setup_type=SetupType.PDH_BREAK, bias=TradeBias.LONG,
                    entry_price=entry, stop_loss=sl, target=target,
                    risk_reward=rr, confidence=base_conf,
                    reason=(
                        f"PDH Break Long: {close:.2f} > PDH {pdh:.2f}. "
                        f"SL {sl:.2f}, TGT {target:.2f}. R:R {rr}{pivot_note}{gap_note}"
                    ),
                    timestamp=candle.get("timestamp", ""),
                )

        # PDL breakdown → Short
        if close < pdl and vol_ok:
            entry = _round(close)

            # Smart SL: use Camarilla S3 as resistance (just broken through)
            min_sl_distance = entry * 0.0015
            atr_sl = _round(pdl + atr * 0.3)  # ATR-based fallback

            if setup.pivot_s3 > 0 and setup.pivot_s3 > entry and (setup.pivot_s3 - entry) >= min_sl_distance:
                sl = _round(setup.pivot_s3)  # S3 becomes resistance after breakdown
            elif setup.pivot_p > 0 and setup.pivot_p > entry:
                sl = _round(min(setup.pivot_p, pdl + atr * 0.3))
            else:
                sl = _round(pdl + atr * 0.3)

            # Smart target: next Camarilla support (S4) or ATR-based
            if setup.pivot_s4 > 0 and setup.pivot_s4 < entry:
                target = _round(setup.pivot_s4)
            else:
                target = _round(entry - atr * 1.0)

            rr = _risk_reward(entry, sl, target, "SELL")

            if rr >= 1.5:
                gap = _gap_pct(setup)
                base_conf = min(0.85, 0.55 + (rr - 1.5) * 0.1)

                # Boost confidence if entry is near Camarilla S3
                if setup.pivot_s3 > 0 and abs(entry - setup.pivot_s3) / entry * 100 < 0.5:
                    base_conf = min(0.90, base_conf + 0.05)

                if gap < -GAP_THRESHOLD_PCT:
                    base_conf *= 0.7
                    gap_note = f" [gap {gap:+.1f}%]"
                else:
                    gap_note = ""

                pivot_note = f" SL@S3={setup.pivot_s3:.0f}" if setup.pivot_s3 > 0 and sl == _round(setup.pivot_s3) else ""

                return IntradaySignal(
                    symbol=setup.symbol, token=setup.token,
                    setup_type=SetupType.PDL_BREAK, bias=TradeBias.SHORT,
                    entry_price=entry, stop_loss=sl, target=target,
                    risk_reward=rr, confidence=base_conf,
                    reason=(
                        f"PDL Break Short: {close:.2f} < PDL {pdl:.2f}. "
                        f"SL {sl:.2f}, TGT {target:.2f}. R:R {rr}{pivot_note}{gap_note}"
                    ),
                    timestamp=candle.get("timestamp", ""),
                )

    return None


# ──────────────────────────────────────────────
# 3. Gap Analysis (Gap and Go / Gap Fill)
# ──────────────────────────────────────────────

def detect_gap(
    setup: StockSetup,
    candles_5min: List[Dict],
    volume_avg: float = 0.0,
) -> Optional[IntradaySignal]:
    """
    Gap Analysis — trade the imbalance.

    GAP AND GO: Gap holds after ORB → trade in gap direction
        Entry: ORB breakout in gap direction
        SL: ORB opposite end / gap fill level
        Target: gap size extension

    GAP FILL: Gap starts filling → trade the fill
        Entry: when price crosses prev close heading toward fill
        SL: today's open (gap extreme)
        Target: previous close (full gap fill)
    """
    if not candles_5min or len(candles_5min) < 4:
        return None

    today_open = candles_5min[0]["open"]
    prev_close = setup.prev_close
    gap = today_open - prev_close
    gap_pct = (gap / prev_close * 100) if prev_close > 0 else 0

    # Only trade significant gaps (> 0.3%)
    if abs(gap_pct) < 0.3:
        return None

    setup.gap_pct = round(gap_pct, 2)
    atr = setup.prev_atr if setup.prev_atr > 0 else abs(gap)

    # ORB levels
    orb_candles = candles_5min[:3]
    orb_high = max(c["high"] for c in orb_candles)
    orb_low = min(c["low"] for c in orb_candles)

    for candle in candles_5min[3:]:
        close = candle["close"]
        vol = candle.get("volume", 0)
        vol_ok = vol > (volume_avg * 1.0) if volume_avg > 0 else True

        if gap > 0:  # Gap UP
            # GAP AND GO: gap holds, ORB breaks up → ride momentum
            if close > orb_high and close > today_open and vol_ok:
                entry = _round(close)
                sl = _round(orb_low)
                target = _round(entry + abs(gap))
                rr = _risk_reward(entry, sl, target, "BUY")
                if rr >= 1.5:
                    return IntradaySignal(
                        symbol=setup.symbol, token=setup.token,
                        setup_type=SetupType.GAP_AND_GO, bias=TradeBias.LONG,
                        entry_price=entry, stop_loss=sl, target=target,
                        risk_reward=rr,
                        confidence=min(0.85, 0.6 + (rr - 1.5) * 0.1),
                        reason=f"Gap & Go Long: {gap_pct:+.1f}% gap held. "
                               f"ORB broke up at {close:.2f}. Target {target:.2f}",
                        timestamp=candle.get("timestamp", ""),
                    )

            # GAP FILL: price coming back down toward prev close
            if close < today_open and close > prev_close and vol_ok:
                entry = _round(close)
                sl = _round(today_open + atr * 0.2)
                target = _round(prev_close)
                rr = _risk_reward(entry, sl, target, "SELL")
                if rr >= 1.5:
                    return IntradaySignal(
                        symbol=setup.symbol, token=setup.token,
                        setup_type=SetupType.GAP_FILL, bias=TradeBias.SHORT,
                        entry_price=entry, stop_loss=sl, target=target,
                        risk_reward=rr,
                        confidence=min(0.8, 0.5 + (rr - 1.5) * 0.1),
                        reason=f"Gap Fill Short: {gap_pct:+.1f}% gap filling. "
                               f"Price {close:.2f} heading to prev close {prev_close:.2f}",
                        timestamp=candle.get("timestamp", ""),
                    )

        else:  # Gap DOWN
            # GAP AND GO: gap holds, ORB breaks down → ride weakness
            if close < orb_low and close < today_open and vol_ok:
                entry = _round(close)
                sl = _round(orb_high)
                target = _round(entry - abs(gap))
                rr = _risk_reward(entry, sl, target, "SELL")
                if rr >= 1.5:
                    return IntradaySignal(
                        symbol=setup.symbol, token=setup.token,
                        setup_type=SetupType.GAP_AND_GO, bias=TradeBias.SHORT,
                        entry_price=entry, stop_loss=sl, target=target,
                        risk_reward=rr,
                        confidence=min(0.85, 0.6 + (rr - 1.5) * 0.1),
                        reason=f"Gap & Go Short: {gap_pct:+.1f}% gap held. "
                               f"ORB broke down at {close:.2f}. Target {target:.2f}",
                        timestamp=candle.get("timestamp", ""),
                    )

            # GAP FILL: price recovering back toward prev close
            if close > today_open and close < prev_close and vol_ok:
                entry = _round(close)
                sl = _round(today_open - atr * 0.2)
                target = _round(prev_close)
                rr = _risk_reward(entry, sl, target, "BUY")
                if rr >= 1.5:
                    return IntradaySignal(
                        symbol=setup.symbol, token=setup.token,
                        setup_type=SetupType.GAP_FILL, bias=TradeBias.LONG,
                        entry_price=entry, stop_loss=sl, target=target,
                        risk_reward=rr,
                        confidence=min(0.8, 0.5 + (rr - 1.5) * 0.1),
                        reason=f"Gap Fill Long: {gap_pct:+.1f}% gap filling. "
                               f"Price {close:.2f} recovering to prev close {prev_close:.2f}",
                        timestamp=candle.get("timestamp", ""),
                    )

    return None


# ──────────────────────────────────────────────
# 4. VWAP Reclaim / Reject
# ──────────────────────────────────────────────

def calculate_vwap(candles_5min: List[Dict]) -> float:
    """Calculate Volume Weighted Average Price from intraday candles."""
    cum_volume = 0
    cum_pv = 0.0
    for c in candles_5min:
        typical_price = (c["high"] + c["low"] + c["close"]) / 3
        vol = c.get("volume", 0)
        cum_pv += typical_price * vol
        cum_volume += vol
    return round(cum_pv / cum_volume, 2) if cum_volume > 0 else 0.0


def detect_vwap(
    setup: StockSetup,
    candles_5min: List[Dict],
    volume_avg: float = 0.0,
) -> Optional[IntradaySignal]:
    """
    VWAP Reclaim / Reject — institutional gravity line.

    VWAP RECLAIM (Long): price was below VWAP, crosses above with volume
    VWAP REJECT (Short): price tests VWAP from below, gets rejected

    Entry: candle that crosses VWAP
    SL: recent swing low/high or VWAP ± ATR*0.3
    Target: ATR-based from entry
    """
    if not candles_5min or len(candles_5min) < 6:
        return None  # Need enough candles for meaningful VWAP

    vwap = calculate_vwap(candles_5min)
    if vwap <= 0:
        return None

    setup.vwap = vwap
    atr = setup.prev_atr if setup.prev_atr > 0 else (setup.prev_high - setup.prev_low)

    # Look at recent candles (skip first 15 min)
    recent = candles_5min[3:]
    if len(recent) < 2:
        return None

    prev_candle = recent[-2]
    curr_candle = recent[-1]
    vol = curr_candle.get("volume", 0)
    vol_ok = vol > (volume_avg * 1.0) if volume_avg > 0 else True

    prev_close = prev_candle["close"]
    curr_close = curr_candle["close"]

    # VWAP Reclaim: was below, now above
    if prev_close < vwap and curr_close > vwap and vol_ok:
        entry = _round(curr_close)
        sl = _round(vwap - atr * 0.3)
        target = _round(entry + atr * 0.8)
        rr = _risk_reward(entry, sl, target, "BUY")

        if rr >= 1.5:
            return IntradaySignal(
                symbol=setup.symbol, token=setup.token,
                setup_type=SetupType.VWAP_RECLAIM, bias=TradeBias.LONG,
                entry_price=entry, stop_loss=sl, target=target,
                risk_reward=rr,
                confidence=min(0.8, 0.55 + (rr - 1.5) * 0.1),
                reason=f"VWAP Reclaim Long: price crossed above VWAP {vwap:.2f}. "
                       f"Entry {entry:.2f}, SL {sl:.2f}, Target {target:.2f}",
                timestamp=curr_candle.get("timestamp", ""),
            )

    # VWAP Reject: approached from below, got rejected
    if prev_close < vwap and curr_candle["high"] >= vwap * 0.998 and curr_close < vwap and vol_ok:
        entry = _round(curr_close)
        sl = _round(vwap + atr * 0.3)
        target = _round(entry - atr * 0.8)
        rr = _risk_reward(entry, sl, target, "SELL")

        if rr >= 1.5:
            return IntradaySignal(
                symbol=setup.symbol, token=setup.token,
                setup_type=SetupType.VWAP_REJECT, bias=TradeBias.SHORT,
                entry_price=entry, stop_loss=sl, target=target,
                risk_reward=rr,
                confidence=min(0.75, 0.5 + (rr - 1.5) * 0.1),
                reason=f"VWAP Reject Short: price rejected at VWAP {vwap:.2f}. "
                       f"Entry {entry:.2f}, SL {sl:.2f}, Target {target:.2f}",
                timestamp=curr_candle.get("timestamp", ""),
            )

    return None


# ──────────────────────────────────────────────
# Master detector: run all structures
# ──────────────────────────────────────────────

def detect_all_structures(
    setup: StockSetup,
    candles_5min: List[Dict],
    volume_avg: float = 0.0,
) -> List[IntradaySignal]:
    """
    Run all price structure detectors on a stock's intraday candles.

    Returns list of triggered signals (usually 0 or 1).
    Multiple signals for the same stock → take the highest confidence one.
    """
    signals = []

    detectors = [
        ("ORB", detect_orb),
        ("PDH/PDL", detect_pdh_pdl),
        ("GAP", detect_gap),
        ("VWAP", detect_vwap),
    ]

    for name, detector in detectors:
        try:
            signal = detector(setup, candles_5min, volume_avg)
            if signal:
                signals.append(signal)
                logger.info(
                    f"  [{setup.symbol}] {name} triggered: {signal.setup_type.value} "
                    f"@ {signal.entry_price:.2f} | R:R {signal.risk_reward} | "
                    f"Conf {signal.confidence:.2f}"
                )
        except Exception as e:
            logger.error(f"  [{setup.symbol}] {name} detector error: {e}")

    # Adjust confidence based on indicator confluence (if available on setup)
    if setup.confluence_score > 0:
        for sig in signals:
            is_long = sig.bias == TradeBias.LONG
            bias_agrees = (
                (is_long and setup.confluence_bias == "LONG") or
                (not is_long and setup.confluence_bias == "SHORT")
            )
            if bias_agrees and setup.confluence_score >= 25:
                sig.confidence = min(0.95, sig.confidence + 0.05)  # +5% for strong confluence
            elif not bias_agrees and setup.confluence_score >= 25:
                sig.confidence = max(0.3, sig.confidence - 0.10)   # -10% for conflicting confluence

    # Sort by confidence descending
    signals.sort(key=lambda s: s.confidence, reverse=True)
    return signals
