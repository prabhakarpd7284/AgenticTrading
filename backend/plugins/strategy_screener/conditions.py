"""
Condition Engine — evaluates trading conditions against indicator snapshots.

Each Condition is a pure function: (snapshot, bars, prev_snapshot) → bool.
A Strategy is a list of Conditions that ALL must be true to fire a Signal.

No side effects, no I/O — easy to backtest.
"""
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from enum import Enum
from typing import Optional

from trading.screener.candle_store import CandleStore, CandleBar
from trading.screener.indicator_engine import IndicatorSnapshot


class ConditionType(Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PRICE_CROSSES_ABOVE = "crosses_above"
    PRICE_CROSSES_BELOW = "crosses_below"
    INDICATOR_COMPARE = "indicator_compare"
    CANDLE_PATTERN = "candle_pattern"
    TIME_WINDOW = "time_window"
    BARS_SINCE = "bars_since"
    BB_POSITION = "bb_position"
    MACD_CROSSOVER = "macd_crossover"
    INDICATOR_VS_INDICATOR = "indicator_vs_indicator"
    VOLATILITY_FILTER = "volatility_filter"
    N_BAR_HIGH_BREAK = "n_bar_high_break"
    N_BAR_LOW_BREAK = "n_bar_low_break"
    VOLUME_SURGE = "volume_surge"


@dataclass
class Condition:
    """A single evaluatable condition."""
    type: ConditionType
    timeframe: str               # "1m", "5m", "15m"
    params: dict = field(default_factory=dict)
    description: str = ""

    def evaluate(
        self,
        snap: IndicatorSnapshot,
        bars: list[CandleBar],
        prev_snap: Optional[IndicatorSnapshot],
        now: datetime = None,
    ) -> bool:
        """Evaluate this condition. Returns True if met."""
        if now is None:
            now = datetime.now()

        evaluator = _EVALUATORS.get(self.type)
        if evaluator is None:
            return False
        return evaluator(self.params, snap, bars, prev_snap, now)


# ──────────────────────────────────────────────
# Condition evaluator functions
# ──────────────────────────────────────────────

def _eval_price_above(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Check if current price is above an indicator level."""
    level_name = params.get("level", "")
    offset_pct = params.get("offset_pct", 0)

    level = _resolve_level(level_name, snap)
    if level is None or level == 0:
        return False

    level_adj = level * (1 + offset_pct / 100)
    return snap.last_close > level_adj


def _eval_price_below(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Check if current price is below an indicator level."""
    level_name = params.get("level", "")
    offset_pct = params.get("offset_pct", 0)

    level = _resolve_level(level_name, snap)
    if level is None or level == 0:
        return False

    level_adj = level * (1 - offset_pct / 100)
    return snap.last_close < level_adj


def _eval_crosses_above(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Price was below level on previous bar, now above."""
    if prev is None:
        return False

    level_name = params.get("level", "")
    current_level = _resolve_level(level_name, snap)
    prev_level = _resolve_level(level_name, prev)

    if current_level is None or prev_level is None:
        return False

    return prev.last_close <= prev_level and snap.last_close > current_level


def _eval_crosses_below(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Price was above level on previous bar, now below."""
    if prev is None:
        return False

    level_name = params.get("level", "")
    current_level = _resolve_level(level_name, snap)
    prev_level = _resolve_level(level_name, prev)

    if current_level is None or prev_level is None:
        return False

    return prev.last_close >= prev_level and snap.last_close < current_level


def _eval_indicator_compare(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Compare an indicator value against a threshold."""
    indicator = params.get("indicator", "")
    op = params.get("op", ">")
    value = params.get("value", 0)

    actual = getattr(snap, indicator, None)
    if actual is None:
        return False

    if op == ">":
        return actual > value
    elif op == "<":
        return actual < value
    elif op == ">=":
        return actual >= value
    elif op == "<=":
        return actual <= value
    elif op == "==":
        return actual == value
    return False


def _eval_candle_pattern(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Detect candle patterns."""
    pattern = params.get("pattern", "")

    if pattern == "big_candle":
        # Bar range > N × ATR
        threshold = params.get("range_vs_atr", 1.5)
        lookback = params.get("lookback_bars", 0)
        time_before = params.get("time_before")

        if lookback > 0:
            # Check if ANY of the last N bars was a big candle
            check_bars = bars[-lookback:] if len(bars) >= lookback else bars
            for bar in check_bars:
                if time_before:
                    cutoff_h, cutoff_m = map(int, time_before.split(":"))
                    if bar.timestamp.hour > cutoff_h or (bar.timestamp.hour == cutoff_h and bar.timestamp.minute > cutoff_m):
                        continue
                bar_range = bar.high - bar.low
                if snap.atr_14 > 0 and bar_range / snap.atr_14 >= threshold:
                    return True
            return False
        else:
            return snap.bar_range_vs_atr >= threshold

    elif pattern == "bullish_engulfing":
        if len(bars) < 2:
            return False
        prev_bar, curr_bar = bars[-2], bars[-1]
        return (prev_bar.close < prev_bar.open and  # prev bearish
                curr_bar.close > curr_bar.open and   # curr bullish
                curr_bar.close > prev_bar.open and   # engulfs prev open
                curr_bar.open < prev_bar.close)      # opens below prev close

    elif pattern == "bearish_engulfing":
        if len(bars) < 2:
            return False
        prev_bar, curr_bar = bars[-2], bars[-1]
        return (prev_bar.close > prev_bar.open and
                curr_bar.close < curr_bar.open and
                curr_bar.close < prev_bar.open and
                curr_bar.open > prev_bar.close)

    elif pattern == "doji":
        if not bars:
            return False
        bar = bars[-1]
        body = abs(bar.close - bar.open)
        range_ = bar.high - bar.low
        return range_ > 0 and body / range_ < 0.1

    return False


def _eval_time_window(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Check if current time is within a window."""
    after = params.get("after", "00:00")
    before = params.get("before", "23:59")

    after_h, after_m = map(int, after.split(":"))
    before_h, before_m = map(int, before.split(":"))

    after_time = dt_time(after_h, after_m)
    before_time = dt_time(before_h, before_m)

    return after_time <= now.time() <= before_time


def _eval_bars_since(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Check that at least N bars have passed since a condition was true."""
    min_bars = params.get("min_bars", 1)
    condition_type = params.get("condition", "")

    if condition_type == "big_candle":
        # Find the most recent big candle in history
        threshold = params.get("range_vs_atr", 1.5)
        if snap.atr_14 <= 0:
            return False
        for i in range(len(bars) - 1, -1, -1):
            bar_range = bars[i].high - bars[i].low
            if bar_range / snap.atr_14 >= threshold:
                bars_since = len(bars) - 1 - i
                return bars_since >= min_bars
        return False

    return True


def _eval_bb_position(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Check price position relative to Bollinger Bands."""
    position = params.get("position", "")
    if position == "above_upper":
        return snap.last_close > snap.bb_upper and snap.bb_upper > 0
    elif position == "below_lower":
        return snap.last_close < snap.bb_lower and snap.bb_lower > 0
    elif position == "inside":
        return snap.bb_lower <= snap.last_close <= snap.bb_upper and snap.bb_upper > 0
    elif position == "squeeze":
        return snap.bb_squeeze
    return False


def _eval_macd_crossover(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Check for MACD crossover."""
    direction = params.get("direction", "BULLISH")
    return snap.macd_crossover == direction


def _eval_indicator_vs_indicator(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Compare two indicator values against each other. e.g., EMA9 > EMA21."""
    ind_a = params.get("indicator_a", "")
    ind_b = params.get("indicator_b", "")
    op = params.get("op", ">")

    val_a = getattr(snap, ind_a, None)
    val_b = getattr(snap, ind_b, None)
    if val_a is None or val_b is None:
        return False

    if op == ">":
        return val_a > val_b
    elif op == "<":
        return val_a < val_b
    elif op == ">=":
        return val_a >= val_b
    return False


def _eval_volatility_filter(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Filter stocks by ATR% (ATR / price). Rejects stocks that are too volatile or too quiet."""
    max_atr_pct = params.get("max_atr_pct", 2.0)
    min_atr_pct = params.get("min_atr_pct", 0.3)

    if snap.atr_14 <= 0 or snap.last_close <= 0:
        return False

    atr_pct = snap.atr_14 / snap.last_close * 100
    return min_atr_pct <= atr_pct <= max_atr_pct


def _eval_n_bar_high_break(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Price breaks above the highest high of the last N completed bars.
    Detects breakouts above recent resistance."""
    lookback = params.get("lookback", 20)
    if len(bars) < lookback + 1:
        return False
    # Exclude current (last) bar — check if it broke above prior N bars' high
    prior_bars = bars[-(lookback + 1):-1]
    prior_high = max(b.high for b in prior_bars)
    current = bars[-1]
    return current.close > prior_high


def _eval_n_bar_low_break(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Price breaks below the lowest low of the last N completed bars.
    Detects breakdowns below recent support."""
    lookback = params.get("lookback", 20)
    if len(bars) < lookback + 1:
        return False
    prior_bars = bars[-(lookback + 1):-1]
    prior_low = min(b.low for b in prior_bars)
    current = bars[-1]
    return current.close < prior_low


def _eval_volume_surge(params: dict, snap: IndicatorSnapshot, bars: list, prev: Optional[IndicatorSnapshot], now: datetime) -> bool:
    """Current bar volume > multiplier × average volume of last N bars.
    Confirms breakout/breakdown with participation."""
    lookback = params.get("lookback", 20)
    multiplier = params.get("multiplier", 1.5)
    if len(bars) < lookback + 1:
        return False
    prior_bars = bars[-(lookback + 1):-1]
    avg_vol = sum(b.volume for b in prior_bars) / len(prior_bars) if prior_bars else 0
    if avg_vol <= 0:
        return False
    current_vol = bars[-1].volume
    return current_vol > avg_vol * multiplier


# ── Evaluator registry ──
_EVALUATORS = {
    ConditionType.PRICE_ABOVE: _eval_price_above,
    ConditionType.PRICE_BELOW: _eval_price_below,
    ConditionType.PRICE_CROSSES_ABOVE: _eval_crosses_above,
    ConditionType.PRICE_CROSSES_BELOW: _eval_crosses_below,
    ConditionType.INDICATOR_COMPARE: _eval_indicator_compare,
    ConditionType.CANDLE_PATTERN: _eval_candle_pattern,
    ConditionType.TIME_WINDOW: _eval_time_window,
    ConditionType.BARS_SINCE: _eval_bars_since,
    ConditionType.BB_POSITION: _eval_bb_position,
    ConditionType.MACD_CROSSOVER: _eval_macd_crossover,
    ConditionType.INDICATOR_VS_INDICATOR: _eval_indicator_vs_indicator,
    ConditionType.VOLATILITY_FILTER: _eval_volatility_filter,
    ConditionType.N_BAR_HIGH_BREAK: _eval_n_bar_high_break,
    ConditionType.N_BAR_LOW_BREAK: _eval_n_bar_low_break,
    ConditionType.VOLUME_SURGE: _eval_volume_surge,
}


# ── Level resolver ──

def _resolve_level(name: str, snap: IndicatorSnapshot) -> Optional[float]:
    """Resolve a level name to a numeric value from the snapshot."""
    level_map = {
        "sma_9": snap.sma_9,
        "sma_20": snap.sma_20,
        "ema_9": snap.ema_9,
        "ema_21": snap.ema_21,
        "bb_upper": snap.bb_upper,
        "bb_middle": snap.bb_middle,
        "bb_lower": snap.bb_lower,
        "vwap": snap.vwap,
        "pivot": snap.pivot,
        "classic_pivot": snap.classic_pivot,
        "r1": snap.r1, "r2": snap.r2, "r3": snap.r3, "r4": snap.r4,
        "s1": snap.s1, "s2": snap.s2, "s3": snap.s3, "s4": snap.s4,
        "classic_r1": snap.classic_r1, "classic_r2": snap.classic_r2, "classic_r3": snap.classic_r3,
        "classic_s1": snap.classic_s1, "classic_s2": snap.classic_s2, "classic_s3": snap.classic_s3,
    }
    return level_map.get(name)
