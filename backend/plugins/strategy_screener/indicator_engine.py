"""
Indicator Engine — incremental computation on bar close.

Reuses trading.utils.indicators for all math.
Maintains a snapshot per (symbol, timeframe) pair, updated only when
a new completed bar arrives — not on every tick.
"""
from dataclasses import dataclass
from typing import Optional

from trading.utils.indicators import (
    sma, ema, bollinger_bands, rsi, atr, macd, camarilla_pivots,
)
from plugins.strategy_screener.candle_store import CandleStore


@dataclass
class IndicatorSnapshot:
    """All indicators for one symbol at one timeframe at a point in time."""
    # Moving averages
    sma_9: Optional[float] = None
    sma_20: Optional[float] = None
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None

    # Bollinger Bands
    bb_upper: float = 0
    bb_middle: float = 0
    bb_lower: float = 0
    bb_bandwidth: float = 0
    bb_squeeze: bool = False

    # Oscillators
    rsi_14: float = 50
    macd_value: float = 0
    macd_signal: float = 0
    macd_histogram: float = 0
    macd_crossover: str = "NONE"

    # Volatility
    atr_14: float = 0

    # Session
    vwap: float = 0

    # Pivots (daily — same all day)
    pivot: float = 0
    r1: float = 0
    r2: float = 0
    r3: float = 0
    r4: float = 0
    s1: float = 0
    s2: float = 0
    s3: float = 0
    s4: float = 0

    # Classic pivot points
    classic_pivot: float = 0
    classic_r1: float = 0
    classic_r2: float = 0
    classic_r3: float = 0
    classic_s1: float = 0
    classic_s2: float = 0
    classic_s3: float = 0

    # Current bar context
    last_close: float = 0
    last_high: float = 0
    last_low: float = 0
    last_open: float = 0
    bar_range: float = 0
    bar_range_vs_atr: float = 0  # current bar range / ATR

    def price_vs_sma9(self) -> str:
        if self.sma_9 is None or self.last_close == 0:
            return "unknown"
        if self.last_close > self.sma_9:
            return "above"
        return "below"

    def price_vs_bb(self) -> str:
        if self.bb_upper == 0:
            return "unknown"
        if self.last_close > self.bb_upper:
            return "above_upper"
        if self.last_close < self.bb_lower:
            return "below_lower"
        return "inside"

    def rsi_zone(self) -> str:
        if self.rsi_14 >= 70:
            return "overbought"
        if self.rsi_14 <= 30:
            return "oversold"
        return "neutral"


class IndicatorEngine:
    """
    Computes and caches indicator snapshots per (symbol, timeframe).

    Call update() when a bar closes. Call get() anytime for cached values.
    """

    def __init__(self):
        self._cache: dict[tuple[str, str], IndicatorSnapshot] = {}
        self._prev_cache: dict[tuple[str, str], IndicatorSnapshot] = {}

    def update(self, symbol: str, tf: str, store: CandleStore) -> IndicatorSnapshot:
        """Recompute all indicators for symbol@timeframe. Called on bar close."""
        closes = store.get_closes(tf)
        bars_dicts = store.get_bars_as_dicts(tf)
        last_bar = store.get_last_bar(tf)

        # Save previous snapshot for crossover detection
        prev = self._cache.get((symbol, tf))
        if prev:
            self._prev_cache[(symbol, tf)] = prev

        snap = IndicatorSnapshot()

        if not closes:
            self._cache[(symbol, tf)] = snap
            return snap

        # Moving averages
        snap.sma_9 = sma(closes, 9)
        snap.sma_20 = sma(closes, 20)
        snap.ema_9 = ema(closes, 9)
        snap.ema_21 = ema(closes, 21)

        # Bollinger Bands
        if len(closes) >= 20:
            bb = bollinger_bands(closes, 20)
            snap.bb_upper = bb["upper"]
            snap.bb_middle = bb["middle"]
            snap.bb_lower = bb["lower"]
            snap.bb_bandwidth = bb["bandwidth"]
            snap.bb_squeeze = bb["squeeze"]

        # RSI
        snap.rsi_14 = rsi(closes, 14)

        # MACD
        if len(closes) >= 35:
            m = macd(closes)
            snap.macd_value = m["macd"]
            snap.macd_signal = m["signal"]
            snap.macd_histogram = m["histogram"]
            snap.macd_crossover = m["crossover"]

        # ATR
        if len(bars_dicts) >= 5:
            snap.atr_14 = atr(bars_dicts, 14)

        # VWAP (session-level, from store)
        snap.vwap = store.vwap

        # Pivots (Camarilla — from prev day OHLC, computed once)
        if store.prev_day_high > 0:
            cam = camarilla_pivots(store.prev_day_high, store.prev_day_low, store.prev_day_close)
            snap.pivot = cam["P"]
            snap.r1, snap.r2, snap.r3, snap.r4 = cam["R1"], cam["R2"], cam["R3"], cam["R4"]
            snap.s1, snap.s2, snap.s3, snap.s4 = cam["S1"], cam["S2"], cam["S3"], cam["S4"]

            # Classic pivots
            h, l, c = store.prev_day_high, store.prev_day_low, store.prev_day_close
            cp = (h + l + c) / 3
            snap.classic_pivot = round(cp, 2)
            snap.classic_r1 = round(2 * cp - l, 2)
            snap.classic_s1 = round(2 * cp - h, 2)
            snap.classic_r2 = round(cp + (h - l), 2)
            snap.classic_s2 = round(cp - (h - l), 2)
            snap.classic_r3 = round(h + 2 * (cp - l), 2)
            snap.classic_s3 = round(l - 2 * (h - cp), 2)

        # Current bar context
        if last_bar:
            snap.last_close = last_bar.close
            snap.last_high = last_bar.high
            snap.last_low = last_bar.low
            snap.last_open = last_bar.open
            snap.bar_range = last_bar.high - last_bar.low
            if snap.atr_14 > 0:
                snap.bar_range_vs_atr = round(snap.bar_range / snap.atr_14, 2)

        self._cache[(symbol, tf)] = snap
        return snap

    def get(self, symbol: str, tf: str) -> Optional[IndicatorSnapshot]:
        """Get cached snapshot. Returns None if never computed."""
        return self._cache.get((symbol, tf))

    def get_previous(self, symbol: str, tf: str) -> Optional[IndicatorSnapshot]:
        """Get the snapshot from the bar BEFORE the current one (for crossover detection)."""
        return self._prev_cache.get((symbol, tf))

    def get_indicator_value(self, symbol: str, tf: str, indicator_name: str) -> Optional[float]:
        """Get a specific indicator value by name."""
        snap = self.get(symbol, tf)
        if snap is None:
            return None
        return getattr(snap, indicator_name, None)
