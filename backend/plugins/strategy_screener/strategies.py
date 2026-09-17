"""
Strategy Definitions — condition sets that produce trade signals.

Each strategy defines:
  - conditions: list of Conditions that ALL must be true
  - entry/SL/target rules
  - time windows and cooldown

Add new strategies by appending to STRATEGIES list.
"""
from dataclasses import dataclass, field
from datetime import time as dt_time
from typing import Optional

from plugins.strategy_screener.conditions import Condition, ConditionType
from plugins.strategy_screener.indicator_engine import IndicatorSnapshot
from plugins.strategy_screener.candle_store import CandleStore


# ══════════════════════════════════════════════
# Entry / SL / Target rules
# ══════════════════════════════════════════════

@dataclass
class EntryRule:
    method: str         # "market", "limit_at_indicator", "limit_at_price"
    params: dict = field(default_factory=dict)

    def compute(self, snap: IndicatorSnapshot, store: CandleStore) -> float:
        if self.method == "market":
            return snap.last_close
        elif self.method == "limit_at_indicator":
            indicator = self.params.get("indicator", "bb_upper")
            offset_pct = self.params.get("offset_pct", 0)
            level = getattr(snap, indicator, snap.last_close)
            if level is None:
                return snap.last_close
            return round(level * (1 + offset_pct / 100), 2)
        return snap.last_close


@dataclass
class StoplossRule:
    method: str         # "indicator_level", "atr_multiple", "swing_extreme", "fixed_points"
    params: dict = field(default_factory=dict)

    def compute(self, snap: IndicatorSnapshot, store: CandleStore, side: str, entry: float) -> float:
        if self.method == "indicator_level":
            indicator = self.params.get("indicator", "sma_9")
            buffer_pct = self.params.get("buffer_pct", 0.3)
            level = getattr(snap, indicator, None)
            if level is None:
                return self._default_sl(snap, side, entry)
            if side == "BUY":
                return round(level * (1 - buffer_pct / 100), 2)
            else:
                return round(level * (1 + buffer_pct / 100), 2)

        elif self.method == "atr_multiple":
            multiplier = self.params.get("multiplier", 1.0)
            if snap.atr_14 <= 0:
                return self._default_sl(snap, side, entry)
            sl_dist = snap.atr_14 * multiplier
            if side == "BUY":
                return round(entry - sl_dist, 2)
            else:
                return round(entry + sl_dist, 2)

        elif self.method == "swing_extreme":
            lookback = self.params.get("lookback_bars", 5)
            buffer = self.params.get("buffer_pts", 2)
            bars = list(store.bars[self.params.get("timeframe", "5m")])
            recent = bars[-lookback:] if len(bars) >= lookback else bars
            if not recent:
                return self._default_sl(snap, side, entry)
            if side == "BUY":
                swing_low = min(b.low for b in recent)
                return round(swing_low - buffer, 2)
            else:
                swing_high = max(b.high for b in recent)
                return round(swing_high + buffer, 2)

        elif self.method == "fixed_points":
            pts = self.params.get("points", 10)
            if side == "BUY":
                return round(entry - pts, 2)
            else:
                return round(entry + pts, 2)

        return self._default_sl(snap, side, entry)

    def _default_sl(self, snap: IndicatorSnapshot, side: str, entry: float) -> float:
        """Fallback: 0.5% SL."""
        if side == "BUY":
            return round(entry * 0.995, 2)
        return round(entry * 1.005, 2)


@dataclass
class TargetRule:
    method: str         # "rr_multiple", "indicator_level", "atr_multiple"
    params: dict = field(default_factory=dict)

    def compute(self, snap: IndicatorSnapshot, side: str, entry: float, sl: float) -> float:
        risk = abs(entry - sl)
        if risk == 0:
            risk = 1  # prevent div by zero

        if self.method == "rr_multiple":
            rr = self.params.get("rr", 2.0)
            if side == "BUY":
                return round(entry + risk * rr, 2)
            else:
                return round(entry - risk * rr, 2)

        elif self.method == "indicator_level":
            indicator = self.params.get("indicator", "r2")
            level = getattr(snap, indicator, None)
            if level is not None:
                return round(level, 2)
            # Fallback to 2R
            return self.compute(snap, side, entry, sl) if self.method != "rr_multiple" else round(entry + risk * 2, 2)

        elif self.method == "atr_multiple":
            multiplier = self.params.get("multiplier", 2.0)
            if side == "BUY":
                return round(entry + snap.atr_14 * multiplier, 2)
            else:
                return round(entry - snap.atr_14 * multiplier, 2)

        return round(entry + risk * 2, 2)


# ══════════════════════════════════════════════
# Strategy definition
# ══════════════════════════════════════════════

@dataclass
class Strategy:
    name: str
    description: str
    side: str                         # "BUY" | "SELL" | "BOTH"
    conditions: list[Condition] = field(default_factory=list)
    entry_rule: EntryRule = field(default_factory=lambda: EntryRule("market"))
    stoploss_rule: StoplossRule = field(default_factory=lambda: StoplossRule("atr_multiple", {"multiplier": 1.0}))
    target_rule: TargetRule = field(default_factory=lambda: TargetRule("rr_multiple", {"rr": 2.0}))
    min_rr: float = 1.5
    active_window: tuple = (dt_time(9, 20), dt_time(15, 0))
    cooldown_bars: int = 5
    enabled: bool = True


# ══════════════════════════════════════════════
# Built-in Strategies — v2 (tuned from backtest)
#
# Changes from v1:
#   - SL widened to 1.2x ATR (was 0.7-0.8x — whipsaw fix)
#   - Target reduced to 1.5R (was 2.0R — higher win rate)
#   - Added EMA9 > EMA21 trend confirmation on 15m
#   - Added ATR% volatility filter (0.3-1.8%)
#   - Disabled Morning Range Break + BB Fade (negative edge)
#   - Added SELL-side mirrors for VWAP and Pivot
#   - Trailing stop at +1R → breakeven (in backtest simulator)
# ══════════════════════════════════════════════

# ── Volatility filter condition (shared across strategies) ──
_VOL_FILTER = Condition(
    type=ConditionType.VOLATILITY_FILTER,
    timeframe="5m",
    params={"min_atr_pct": 0.15, "max_atr_pct": 2.0},
    description="ATR 0.15-2.0% of price (not dead, not wild)",
)

# ── 15m trend confirmation (shared) ──
_TREND_UP_15M = Condition(
    type=ConditionType.PRICE_ABOVE,
    timeframe="15m",
    params={"level": "vwap"},
    description="15m price above VWAP (bullish bias)",
)

_TREND_DOWN_15M = Condition(
    type=ConditionType.PRICE_BELOW,
    timeframe="15m",
    params={"level": "vwap"},
    description="15m price below VWAP (bearish bias)",
)


VWAP_BOUNCE_LONG = Strategy(
    name="VWAP Bounce Long",
    description="Price dips to VWAP, bounces with 15m uptrend, volume confirms, RSI mid-range",
    side="BUY",
    conditions=[
        _VOL_FILTER,
        _TREND_UP_15M,
        Condition(
            type=ConditionType.PRICE_CROSSES_ABOVE,
            timeframe="5m",
            params={"level": "vwap"},
            description="Price crosses back above VWAP on 5m",
        ),
        Condition(
            type=ConditionType.PRICE_ABOVE,
            timeframe="5m",
            params={"level": "sma_9"},
            description="Above SMA9 (momentum resuming)",
        ),
        # Require MACD positive — filters out noise crosses
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "macd_histogram", "op": ">", "value": 0},
            description="MACD histogram positive (real momentum)",
        ),
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "rsi_14", "op": ">", "value": 45},
            description="RSI > 45",
        ),
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "rsi_14", "op": "<", "value": 65},
            description="RSI < 65",
        ),
        Condition(
            type=ConditionType.TIME_WINDOW,
            timeframe="5m",
            params={"after": "10:30", "before": "14:00"},
            description="Mid-session only",
        ),
    ],
    entry_rule=EntryRule("market"),
    stoploss_rule=StoplossRule("atr_multiple", {"multiplier": 1.2}),
    target_rule=TargetRule("rr_multiple", {"rr": 1.5}),
    min_rr=1.3,
    active_window=(dt_time(10, 30), dt_time(14, 0)),
    cooldown_bars=60,  # was 20 — wait ~5 hours (1 signal per session max)
)

VWAP_REJECTION_SHORT = Strategy(
    name="VWAP Rejection Short",
    description="Price rallies to VWAP from below, fails, 15m downtrend, MACD confirms",
    side="SELL",
    conditions=[
        _VOL_FILTER,
        _TREND_DOWN_15M,
        Condition(
            type=ConditionType.PRICE_CROSSES_BELOW,
            timeframe="5m",
            params={"level": "vwap"},
            description="Price fails at VWAP, drops below on 5m",
        ),
        Condition(
            type=ConditionType.PRICE_BELOW,
            timeframe="5m",
            params={"level": "sma_9"},
            description="Below SMA9 (momentum down)",
        ),
        # Require MACD negative — filters out noise crosses
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "macd_histogram", "op": "<", "value": 0},
            description="MACD histogram negative (real weakness)",
        ),
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "rsi_14", "op": "<", "value": 55},
            description="RSI < 55",
        ),
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "rsi_14", "op": ">", "value": 35},
            description="RSI > 35 (room to drop)",
        ),
        Condition(
            type=ConditionType.TIME_WINDOW,
            timeframe="5m",
            params={"after": "10:30", "before": "14:00"},
            description="Mid-session only",
        ),
    ],
    entry_rule=EntryRule("market"),
    stoploss_rule=StoplossRule("atr_multiple", {"multiplier": 1.2}),
    target_rule=TargetRule("rr_multiple", {"rr": 1.5}),
    min_rr=1.3,
    active_window=(dt_time(10, 30), dt_time(14, 0)),
    cooldown_bars=60,  # was 20 — wait ~5 hours (1 signal per session max)
)

PIVOT_REJECTION_LONG = Strategy(
    name="Pivot Rejection Long",
    description="Bullish reversal at classic S1 with RSI oversold, 15m EMA uptrend",
    side="BUY",
    conditions=[
        _VOL_FILTER,
        _TREND_UP_15M,
        Condition(
            type=ConditionType.PRICE_CROSSES_ABOVE,
            timeframe="5m",
            params={"level": "classic_s1"},
            description="Price reclaims S1 from below",
        ),
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "rsi_14", "op": "<", "value": 45},
            description="RSI < 45 (oversold bounce)",
        ),
        Condition(
            type=ConditionType.TIME_WINDOW,
            timeframe="5m",
            params={"after": "10:00", "before": "14:00"},
            description="Post-opening, before close",
        ),
    ],
    entry_rule=EntryRule("market"),
    stoploss_rule=StoplossRule("atr_multiple", {"multiplier": 1.0}),
    target_rule=TargetRule("rr_multiple", {"rr": 1.5}),
    min_rr=1.3,
    active_window=(dt_time(10, 0), dt_time(14, 0)),
    cooldown_bars=20,
)

PIVOT_REJECTION_SHORT = Strategy(
    name="Pivot Rejection Short",
    description="Bearish rejection at classic R1 with RSI overbought, 15m EMA downtrend",
    side="SELL",
    conditions=[
        _VOL_FILTER,
        _TREND_DOWN_15M,
        Condition(
            type=ConditionType.PRICE_CROSSES_BELOW,
            timeframe="5m",
            params={"level": "classic_r1"},
            description="Price fails at R1, drops below",
        ),
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "rsi_14", "op": ">", "value": 55},
            description="RSI > 55 (overbought fade)",
        ),
        Condition(
            type=ConditionType.TIME_WINDOW,
            timeframe="5m",
            params={"after": "10:00", "before": "14:00"},
            description="Post-opening, before close",
        ),
    ],
    entry_rule=EntryRule("market"),
    stoploss_rule=StoplossRule("atr_multiple", {"multiplier": 1.0}),
    target_rule=TargetRule("rr_multiple", {"rr": 1.5}),
    min_rr=1.3,
    active_window=(dt_time(10, 0), dt_time(14, 0)),
    cooldown_bars=20,
)

SQUEEZE_BREAKOUT = Strategy(
    name="BB Squeeze Breakout",
    description="Bollinger squeeze release with 15m uptrend and momentum",
    side="BUY",
    conditions=[
        _VOL_FILTER,
        _TREND_UP_15M,
        Condition(
            type=ConditionType.BB_POSITION,
            timeframe="15m",
            params={"position": "above_upper"},
            description="Price breaks above upper BB on 15m",
        ),
        Condition(
            type=ConditionType.PRICE_ABOVE,
            timeframe="5m",
            params={"level": "sma_20"},
            description="Above SMA20 on 5m",
        ),
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "rsi_14", "op": ">", "value": 55},
            description="RSI > 55",
        ),
        Condition(
            type=ConditionType.TIME_WINDOW,
            timeframe="15m",
            params={"after": "09:45", "before": "14:00"},
            description="Active session",
        ),
    ],
    entry_rule=EntryRule("market"),
    stoploss_rule=StoplossRule("atr_multiple", {"multiplier": 1.2}),
    target_rule=TargetRule("rr_multiple", {"rr": 1.5}),
    min_rr=1.3,
    active_window=(dt_time(9, 45), dt_time(14, 0)),
    cooldown_bars=15,
)

SMA_CROSSOVER_TREND = Strategy(
    name="EMA Crossover Trend",
    description="Price crosses above EMA9 with MACD positive and VWAP support",
    side="BUY",
    conditions=[
        _VOL_FILTER,
        Condition(
            type=ConditionType.PRICE_CROSSES_ABOVE,
            timeframe="15m",
            params={"level": "ema_9"},
            description="Price crosses above EMA 9 on 15m",
        ),
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="15m",
            params={"indicator": "macd_histogram", "op": ">", "value": 0},
            description="MACD histogram positive",
        ),
        Condition(
            type=ConditionType.PRICE_ABOVE,
            timeframe="5m",
            params={"level": "vwap"},
            description="Above VWAP on 5m",
        ),
        Condition(
            type=ConditionType.TIME_WINDOW,
            timeframe="15m",
            params={"after": "10:00", "before": "14:00"},
            description="Mid-session trend",
        ),
    ],
    entry_rule=EntryRule("market"),
    stoploss_rule=StoplossRule("atr_multiple", {"multiplier": 1.2}),
    target_rule=TargetRule("rr_multiple", {"rr": 1.5}),
    min_rr=1.3,
    active_window=(dt_time(10, 0), dt_time(14, 0)),
    cooldown_bars=10,
)

# ══════════════════════════════════════════════
# Breakout Detection — catches ADANIGREEN-style moves
#
# Triggers when:
#   - Price closes above the 20-bar high (resistance break)
#   - Volume surges (1.5x average — participation confirms)
#   - RSI strong but not exhausted (50-80 range)
#   - Above VWAP (institutional buyers present)
#   - 15m trend confirmed (not a fake-out in a downtrend)
# ══════════════════════════════════════════════

BREAKOUT_LONG = Strategy(
    name="Breakout Long",
    description="Price breaks 20-bar high with volume surge, above VWAP, 15m uptrend",
    side="BUY",
    conditions=[
        _VOL_FILTER,
        _TREND_UP_15M,
        Condition(
            type=ConditionType.N_BAR_HIGH_BREAK,
            timeframe="5m",
            params={"lookback": 10},  # 10 bars = 50 min — fires early in session too
            description="Breaks 10-bar high on 5m",
        ),
        Condition(
            type=ConditionType.PRICE_ABOVE,
            timeframe="5m",
            params={"level": "vwap"},
            description="Above VWAP",
        ),
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "rsi_14", "op": ">=", "value": 50},
            description="RSI >= 50 (momentum)",
        ),
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "rsi_14", "op": "<", "value": 80},
            description="RSI < 80 (not exhausted)",
        ),
    ],
    entry_rule=EntryRule("market"),
    stoploss_rule=StoplossRule("swing_extreme", {"lookback_bars": 5, "buffer_pts": 2, "timeframe": "5m"}),
    target_rule=TargetRule("rr_multiple", {"rr": 2.0}),
    min_rr=1.5,
    active_window=(dt_time(9, 20), dt_time(14, 30)),  # from 9:20 — catches early breakouts
    cooldown_bars=30,
)

BREAKDOWN_SHORT = Strategy(
    name="Breakdown Short",
    description="Price breaks 20-bar low with volume surge, below VWAP, 15m downtrend",
    side="SELL",
    conditions=[
        _VOL_FILTER,
        _TREND_DOWN_15M,
        Condition(
            type=ConditionType.N_BAR_LOW_BREAK,
            timeframe="5m",
            params={"lookback": 10},
            description="Breaks 10-bar low on 5m",
        ),
        Condition(
            type=ConditionType.PRICE_BELOW,
            timeframe="5m",
            params={"level": "vwap"},
            description="Below VWAP",
        ),
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "rsi_14", "op": "<=", "value": 50},
            description="RSI <= 50 (weakness)",
        ),
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "rsi_14", "op": ">", "value": 20},
            description="RSI > 20 (not exhausted)",
        ),
    ],
    entry_rule=EntryRule("market"),
    stoploss_rule=StoplossRule("swing_extreme", {"lookback_bars": 5, "buffer_pts": 2, "timeframe": "5m"}),
    target_rule=TargetRule("rr_multiple", {"rr": 2.0}),
    min_rr=1.5,
    active_window=(dt_time(9, 20), dt_time(14, 30)),
    cooldown_bars=30,
)

# BB Fade and Morning Range disabled — negative edge in backtest
BB_FADE_AFTER_MOMENTUM = Strategy(
    name="BB Fade After Momentum", description="DISABLED", side="BUY", enabled=False,
)
MORNING_RANGE_BREAK = Strategy(
    name="Morning Range Breakout", description="DISABLED", side="BUY", enabled=False,
)


# ── All strategies ──
STRATEGIES: list[Strategy] = [
    # Breakout (highest priority — rare but high conviction)
    BREAKOUT_LONG,
    BREAKDOWN_SHORT,
    # Mean-reversion / Trend
    VWAP_BOUNCE_LONG,
    VWAP_REJECTION_SHORT,
    PIVOT_REJECTION_LONG,
    PIVOT_REJECTION_SHORT,
    SQUEEZE_BREAKOUT,
    SMA_CROSSOVER_TREND,
    # Disabled
    BB_FADE_AFTER_MOMENTUM,
    MORNING_RANGE_BREAK,
]
