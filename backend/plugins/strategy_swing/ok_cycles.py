"""
Oliver Kell Cycle of Price Action — Core Detection Logic.

Detects 8 cycle phases on daily/weekly candles:
  Bullish: RE → WP → EC → BB
  Bearish: EX → WD → EC_BEAR → BB_BEAR

Mirrors the Pine Script indicator logic exactly.
"""
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import pandas as pd

from trading.config import OKCycleConfig, config as default_config
from trading.utils.indicators import _ema


# ══════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════

class CyclePhase(Enum):
    """Oliver Kell's 8 cycle phases + NONE."""
    REVERSAL_EXTENSION = "RE"       # Potential bottom — WATCH
    WEDGE_POP = "WP"                # Momentum entry — BUY
    EMA_CROSSBACK_BULL = "EC"       # Low-risk pullback — BUY
    BASIN_BREAK_BULL = "BB"         # Continuation — BUY
    EXHAUSTION_EXTENSION = "EX"     # Potential top — SELL
    WEDGE_DROP = "WD"               # Breakdown — SHORT/AVOID
    EMA_CROSSBACK_BEAR = "EC_BEAR"  # Failed bounce — SHORT
    BASIN_BREAK_BEAR = "BB_BEAR"    # Continuation down — SHORT
    NONE = "NONE"                   # No active phase


class TrendState(Enum):
    """EMA-stacking trend."""
    BULLISH = "bullish"     # EMA10 > EMA20 > EMA50
    BEARISH = "bearish"     # EMA10 < EMA20 < EMA50
    NEUTRAL = "neutral"     # Mixed


# Phase → trading action mapping
PHASE_ACTION = {
    CyclePhase.REVERSAL_EXTENSION: "WATCH",
    CyclePhase.WEDGE_POP: "BUY",
    CyclePhase.EMA_CROSSBACK_BULL: "BUY",
    CyclePhase.BASIN_BREAK_BULL: "BUY",
    CyclePhase.EXHAUSTION_EXTENSION: "SELL",
    CyclePhase.WEDGE_DROP: "AVOID",
    CyclePhase.EMA_CROSSBACK_BEAR: "SHORT",
    CyclePhase.BASIN_BREAK_BEAR: "SHORT",
    CyclePhase.NONE: "—",
}

# Phases that are actionable for long trades
BULLISH_ACTIONABLE = {
    CyclePhase.WEDGE_POP,
    CyclePhase.EMA_CROSSBACK_BULL,
    CyclePhase.BASIN_BREAK_BULL,
}

# Phases that are actionable for short trades
BEARISH_ACTIONABLE = {
    CyclePhase.WEDGE_DROP,
    CyclePhase.EMA_CROSSBACK_BEAR,
    CyclePhase.BASIN_BREAK_BEAR,
}


# ══════════════════════════════════════════════
# Result
# ══════════════════════════════════════════════

@dataclass
class CycleResult:
    """Analysis result for a single symbol."""
    symbol: str
    phase: CyclePhase
    trend_daily: TrendState
    trend_weekly: TrendState
    aligned: bool                         # daily + weekly trends agree
    action: str                           # BUY / SELL / WATCH / AVOID / SHORT / —
    confidence: float                     # 0-1
    last_close: float = 0.0
    ema_fast: float = 0.0                 # EMA 10
    ema_mid: float = 0.0                  # EMA 20
    ema_slow: float = 0.0                 # EMA 50
    upper_ext: float = 0.0               # Upper extension band
    lower_ext: float = 0.0               # Lower extension band
    volume_ratio: float = 0.0            # Current vol / avg vol
    phase_label: str = ""                # Human-readable phase name
    error: str = ""                       # Non-empty if analysis failed

    def __post_init__(self):
        self.phase_label = self.phase.value if self.phase != CyclePhase.NONE else "—"
        if not self.action:
            self.action = PHASE_ACTION.get(self.phase, "—")


# ══════════════════════════════════════════════
# Cycle Detector
# ══════════════════════════════════════════════

class CycleDetector:
    """
    Detects Oliver Kell cycle phases on daily candle data.

    Usage:
        detector = CycleDetector()
        df = detector.compute_indicators(daily_candles)
        phase = detector.detect_phase(df)
        result = detector.analyze("RELIANCE", daily_candles, weekly_candles)
    """

    def __init__(self, cfg: Optional[OKCycleConfig] = None):
        self.cfg = cfg or default_config.ok_cycle

    # ──────────────────────────────────────────
    # Indicator computation
    # ──────────────────────────────────────────

    def compute_indicators(self, candles: List[dict]) -> pd.DataFrame:
        """
        Compute OK cycle indicators on candle data.

        Args:
            candles: List of dicts with keys: date/timestamp, open, high, low, close, volume

        Returns:
            DataFrame with columns: date, open, high, low, close, volume,
            ema_fast, ema_mid, ema_slow, upper_ext, lower_ext, vol_avg, vol_spike
        """
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)

        # Normalize column names
        if "timestamp" in df.columns and "date" not in df.columns:
            df.rename(columns={"timestamp": "date"}, inplace=True)

        closes = df["close"].tolist()
        volumes = df["volume"].tolist()

        # EMAs — using the project's _ema() helper
        ema_fast_series = _ema(closes, self.cfg.ema_fast)
        ema_mid_series = _ema(closes, self.cfg.ema_mid)
        ema_slow_series = _ema(closes, self.cfg.ema_slow)

        # Pad shorter series with NaN to align with DataFrame index
        # _ema() returns (len - period + 1) values, starting from index (period - 1)
        df["ema_fast"] = self._align_ema(ema_fast_series, len(closes), self.cfg.ema_fast)
        df["ema_mid"] = self._align_ema(ema_mid_series, len(closes), self.cfg.ema_mid)
        df["ema_slow"] = self._align_ema(ema_slow_series, len(closes), self.cfg.ema_slow)

        # Standard deviation for extension bands (rolling)
        df["stdev"] = df["close"].rolling(window=self.cfg.stdev_period).std()

        # Extension bands: EMA_fast ± (stdev × threshold)
        df["upper_ext"] = df["ema_fast"] + (df["stdev"] * self.cfg.ext_threshold)
        df["lower_ext"] = df["ema_fast"] - (df["stdev"] * self.cfg.ext_threshold)

        # Volume analysis
        df["vol_avg"] = df["volume"].rolling(window=self.cfg.vol_avg_period).mean()
        df["vol_spike"] = df["volume"] > (df["vol_avg"] * self.cfg.vol_spike_multiplier)

        # Price range for consolidation detection
        df["range_high"] = df["high"].rolling(window=self.cfg.consolidation_bars).max()
        df["range_low"] = df["low"].rolling(window=self.cfg.consolidation_bars).min()
        df["price_range"] = df["range_high"] - df["range_low"]
        df["avg_range"] = df["price_range"].rolling(window=self.cfg.consolidation_bars).mean()

        return df

    @staticmethod
    def _align_ema(ema_values: List[float], total_len: int, period: int) -> pd.Series:
        """Align EMA series (shorter) to DataFrame index by prepending NaN."""
        pad_len = total_len - len(ema_values)
        return pd.Series([None] * pad_len + ema_values, dtype=float)

    # ──────────────────────────────────────────
    # Trend detection
    # ──────────────────────────────────────────

    def detect_trend(self, ema_fast: float, ema_mid: float, ema_slow: float) -> TrendState:
        """Determine trend from EMA stacking order."""
        if pd.isna(ema_fast) or pd.isna(ema_mid) or pd.isna(ema_slow):
            return TrendState.NEUTRAL
        if ema_fast > ema_mid > ema_slow:
            return TrendState.BULLISH
        if ema_fast < ema_mid < ema_slow:
            return TrendState.BEARISH
        return TrendState.NEUTRAL

    # ──────────────────────────────────────────
    # Phase detection
    # ──────────────────────────────────────────

    def detect_phase(self, df: pd.DataFrame) -> CyclePhase:
        """
        Evaluate all 8 cycle phase conditions on the latest bar.
        Returns the first matching phase (priority: bearish reversals first for safety).
        """
        if len(df) < max(self.cfg.ema_slow, self.cfg.stdev_period) + 5:
            return CyclePhase.NONE

        row = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) >= 3 else prev

        close = row["close"]
        ema_f = row["ema_fast"]
        ema_m = row["ema_mid"]
        ema_s = row["ema_slow"]
        vol_spike = row["vol_spike"]

        prev_close = prev["close"]
        prev_ema_f = prev["ema_fast"]
        prev_ema_m = prev["ema_mid"]

        # Skip if EMAs not computed yet
        if pd.isna(ema_f) or pd.isna(ema_m) or pd.isna(ema_s):
            return CyclePhase.NONE

        upper = row["upper_ext"]
        lower = row["lower_ext"]
        vol_above_avg = row["volume"] > row["vol_avg"] if pd.notna(row["vol_avg"]) else False

        # Above / below EMA checks
        above_emas = close > ema_f and close > ema_m
        below_emas = close < ema_f and close < ema_m
        prev_above = prev_close > prev_ema_f and prev_close > prev_ema_m
        prev_below = prev_close < prev_ema_f and prev_close < prev_ema_m

        # Consolidation check for basin break
        consolidation = False
        if pd.notna(row.get("price_range")) and pd.notna(row.get("avg_range")):
            consolidation = row["price_range"] < row["avg_range"] * 0.5

        # ── 1. Reversal Extension (RE) — potential bottom ──
        if pd.notna(lower) and close < lower and vol_spike and below_emas:
            return CyclePhase.REVERSAL_EXTENSION

        # ── 5. Exhaustion Extension (EX) — potential top ──
        if pd.notna(upper) and close > upper and vol_spike and above_emas:
            return CyclePhase.EXHAUSTION_EXTENSION

        # ── 2. Wedge Pop (WP) — crosses above EMAs with volume ──
        if above_emas and not prev_above and vol_spike:
            return CyclePhase.WEDGE_POP

        # ── 6. Wedge Drop (WD) — crosses below EMAs with volume ──
        if below_emas and not prev_below and vol_above_avg:
            return CyclePhase.WEDGE_DROP

        # ── 3. EMA Crossback Bull (EC) — pullback to EMA zone, bounces ──
        # Price is between EMA10 and EMA20 (regardless of which is higher)
        if (min(ema_f, ema_m) <= close <= max(ema_f, ema_m)
                and prev_above
                and close > prev2["close"]):
            return CyclePhase.EMA_CROSSBACK_BULL

        # ── 7. Bear EMA Crossback (EC_BEAR) — bounce to EMA, fails ──
        # Price is between EMA10 and EMA20 (regardless of which is higher)
        if (min(ema_f, ema_m) <= close <= max(ema_f, ema_m)
                and prev_below
                and close < prev2["close"]):
            return CyclePhase.EMA_CROSSBACK_BEAR

        # ── 4. Basin Break Bull (BB) — breakout from consolidation ──
        if len(df) >= self.cfg.consolidation_bars + 2:
            five_bar_high = df["high"].iloc[-(self.cfg.consolidation_bars + 1):-1].max()
            prev_consolidation = df.iloc[-2].get("price_range", float("inf"))
            prev_avg_range = df.iloc[-2].get("avg_range", 1)
            was_consolidating = (pd.notna(prev_consolidation) and pd.notna(prev_avg_range)
                                 and prev_avg_range > 0
                                 and prev_consolidation < prev_avg_range * 0.5)
            if was_consolidating and close > five_bar_high and vol_spike and close > ema_m:
                return CyclePhase.BASIN_BREAK_BULL

        # ── 8. Basin Break Bear (BB_BEAR) — breakdown from consolidation ──
        if len(df) >= self.cfg.consolidation_bars + 2:
            five_bar_low = df["low"].iloc[-(self.cfg.consolidation_bars + 1):-1].min()
            prev_consolidation = df.iloc[-2].get("price_range", float("inf"))
            prev_avg_range = df.iloc[-2].get("avg_range", 1)
            was_consolidating = (pd.notna(prev_consolidation) and pd.notna(prev_avg_range)
                                 and prev_avg_range > 0
                                 and prev_consolidation < prev_avg_range * 0.5)
            if was_consolidating and close < five_bar_low and vol_above_avg and close < ema_m:
                return CyclePhase.BASIN_BREAK_BEAR

        return CyclePhase.NONE

    # ──────────────────────────────────────────
    # Full analysis
    # ──────────────────────────────────────────

    def analyze(
        self,
        symbol: str,
        daily_candles: List[dict],
        weekly_candles: List[dict],
    ) -> CycleResult:
        """
        Full Oliver Kell analysis for a single symbol.

        Args:
            symbol: NSE symbol (e.g. "RELIANCE")
            daily_candles: List of daily OHLCV dicts
            weekly_candles: List of weekly OHLCV dicts (aggregated from daily)

        Returns:
            CycleResult with phase, trends, action, and confidence
        """
        if not daily_candles or len(daily_candles) < self.cfg.ema_slow + 5:
            return CycleResult(
                symbol=symbol,
                phase=CyclePhase.NONE,
                trend_daily=TrendState.NEUTRAL,
                trend_weekly=TrendState.NEUTRAL,
                aligned=False,
                action="—",
                confidence=0.0,
                error=f"Insufficient data: {len(daily_candles) if daily_candles else 0} daily candles",
            )

        # Compute indicators on daily
        daily_df = self.compute_indicators(daily_candles)

        # Detect daily trend and phase
        last = daily_df.iloc[-1]
        daily_trend = self.detect_trend(last["ema_fast"], last["ema_mid"], last["ema_slow"])
        phase = self.detect_phase(daily_df)

        # Compute weekly trend
        weekly_trend = TrendState.NEUTRAL
        if weekly_candles and len(weekly_candles) >= self.cfg.ema_slow:
            weekly_df = self.compute_indicators(weekly_candles)
            wk_last = weekly_df.iloc[-1]
            weekly_trend = self.detect_trend(
                wk_last["ema_fast"], wk_last["ema_mid"], wk_last["ema_slow"]
            )

        # Trend alignment
        aligned = (
            (daily_trend == TrendState.BULLISH and weekly_trend == TrendState.BULLISH)
            or (daily_trend == TrendState.BEARISH and weekly_trend == TrendState.BEARISH)
        )

        # Confidence scoring
        confidence = self._compute_confidence(phase, daily_trend, weekly_trend, aligned, last)

        # Volume ratio
        vol_ratio = 0.0
        if pd.notna(last.get("vol_avg")) and last["vol_avg"] > 0:
            vol_ratio = round(last["volume"] / last["vol_avg"], 2)

        action = PHASE_ACTION.get(phase, "—")

        return CycleResult(
            symbol=symbol,
            phase=phase,
            trend_daily=daily_trend,
            trend_weekly=weekly_trend,
            aligned=aligned,
            action=action,
            confidence=confidence,
            last_close=round(last["close"], 2),
            ema_fast=round(last["ema_fast"], 2) if pd.notna(last["ema_fast"]) else 0.0,
            ema_mid=round(last["ema_mid"], 2) if pd.notna(last["ema_mid"]) else 0.0,
            ema_slow=round(last["ema_slow"], 2) if pd.notna(last["ema_slow"]) else 0.0,
            upper_ext=round(last["upper_ext"], 2) if pd.notna(last.get("upper_ext")) else 0.0,
            lower_ext=round(last["lower_ext"], 2) if pd.notna(last.get("lower_ext")) else 0.0,
            volume_ratio=vol_ratio,
        )

    def _compute_confidence(
        self,
        phase: CyclePhase,
        daily_trend: TrendState,
        weekly_trend: TrendState,
        aligned: bool,
        last_row: pd.Series,
    ) -> float:
        """Score confidence 0-1 based on trend alignment, volume, and phase strength."""
        if phase == CyclePhase.NONE:
            return 0.0

        score = 0.3  # Base score for having a detected phase

        # Trend alignment bonus
        if aligned:
            score += 0.3
        elif daily_trend != TrendState.NEUTRAL:
            score += 0.1

        # Volume confirmation
        if last_row.get("vol_spike", False):
            score += 0.2

        # Strong phase bonus (extension phases are high conviction)
        if phase in (CyclePhase.REVERSAL_EXTENSION, CyclePhase.EXHAUSTION_EXTENSION):
            score += 0.1

        # EMA proximity bonus (tighter = better setup)
        if pd.notna(last_row.get("ema_fast")) and pd.notna(last_row.get("ema_mid")):
            ema_spread_pct = abs(last_row["ema_fast"] - last_row["ema_mid"]) / last_row["close"] * 100
            if ema_spread_pct < 1.0:
                score += 0.1

        return round(min(score, 1.0), 2)
