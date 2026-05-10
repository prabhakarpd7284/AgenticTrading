"""
Oliver Kell Intraday — cycle phases on 3m/5m/15m with RSI momentum filter.

Combines OK cycle detection (EMA 10/20/50, extension bands, volume spike)
with RSI momentum confirmation:
  - RSI EMA(3) crossing above WMA(21) = bullish momentum confirmed
  - RSI EMA(3) crossing below WMA(21) = bearish momentum confirmed

Trade types:
  1. OK Cycle phases (WP, EC, BB, WD, etc.) with momentum confirmation
  2. GREEN_AFTER_RED — buy green candle after red candle(s) during retracement
     after first bounce of the day
  3. EMA_BOUNCE_3M — buy on bounce from EMA line(s) on 3-min timeframe

Entry only when BOTH pattern AND momentum align.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from trading.config import OKCycleConfig, config as default_config
from trading.utils.indicators import _ema, _wma, _rsi_series, atr
from trading.swing.ok_cycles import CyclePhase, TrendState, PHASE_ACTION


# ══════════════════════════════════════════════
# Trade type enum (extends CyclePhase with new patterns)
# ══════════════════════════════════════════════

TRADE_TYPE_GREEN_AFTER_RED = "GAR"    # Green After Red
TRADE_TYPE_EMA_BOUNCE = "EB3"         # EMA Bounce 3m

# All trade types that generate BUY signals
EXTRA_BUY_TYPES = {TRADE_TYPE_GREEN_AFTER_RED, TRADE_TYPE_EMA_BOUNCE}


# ══════════════════════════════════════════════
# Intraday result
# ══════════════════════════════════════════════

@dataclass
class IntradayCycleResult:
    """Analysis result for a single bar."""
    symbol: str
    timestamp: str
    phase: CyclePhase
    trend: TrendState
    action: str              # BUY / SHORT / —

    # Trade type — CyclePhase.value for cycle trades, or GAR/EB3 for new types
    trade_type: str = ""

    # Momentum
    rsi: float = 50.0
    rsi_ema3: float = 50.0
    rsi_wma21: float = 50.0
    momentum_bullish: bool = False
    momentum_bearish: bool = False

    # Prices
    close: float = 0.0
    ema_fast: float = 0.0
    ema_mid: float = 0.0
    ema_slow: float = 0.0
    atr: float = 0.0

    # Risk levels
    sl: float = 0.0
    target: float = 0.0
    risk_points: float = 0.0

    @property
    def confirmed(self) -> bool:
        """Phase + momentum both confirm."""
        if self.phase == CyclePhase.NONE and not self.trade_type:
            return False
        if self.action == "BUY" and self.momentum_bullish:
            return True
        if self.action in ("SHORT", "AVOID") and self.momentum_bearish:
            return True
        return False


# ══════════════════════════════════════════════
# Intraday Cycle Detector
# ══════════════════════════════════════════════

class IntradayCycleDetector:
    """
    OK cycle detection on intraday bars + RSI momentum filter.

    Usage:
        detector = IntradayCycleDetector()
        df = detector.compute(candles)
        signals = detector.scan(symbol, candles, min_rr=2.0)
    """

    def __init__(
        self,
        cfg: Optional[OKCycleConfig] = None,
        rsi_period: int = 14,
        rsi_ema_period: int = 3,
        rsi_wma_period: int = 21,
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
    ):
        self.cfg = cfg or default_config.ok_cycle
        self.rsi_period = rsi_period
        self.rsi_ema_period = rsi_ema_period
        self.rsi_wma_period = rsi_wma_period
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult

    def compute(self, candles: List[dict]) -> pd.DataFrame:
        """
        Compute all indicators on intraday candles.

        Returns DataFrame with OK cycle columns + RSI momentum columns.
        """
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        closes = df["close"].tolist()
        volumes = df["volume"].tolist()

        n = len(closes)

        # ── EMAs for cycle detection ──
        ema_f = _ema(closes, self.cfg.ema_fast)
        ema_m = _ema(closes, self.cfg.ema_mid)
        ema_s = _ema(closes, self.cfg.ema_slow)

        df["ema_fast"] = _align(ema_f, n, self.cfg.ema_fast)
        df["ema_mid"] = _align(ema_m, n, self.cfg.ema_mid)
        df["ema_slow"] = _align(ema_s, n, self.cfg.ema_slow)

        # ── Extension bands ──
        df["stdev"] = df["close"].rolling(window=self.cfg.stdev_period).std()
        df["upper_ext"] = df["ema_fast"] + (df["stdev"] * self.cfg.ext_threshold)
        df["lower_ext"] = df["ema_fast"] - (df["stdev"] * self.cfg.ext_threshold)

        # ── Volume ──
        df["vol_avg"] = df["volume"].rolling(window=self.cfg.vol_avg_period).mean()
        df["vol_spike"] = df["volume"] > (df["vol_avg"] * self.cfg.vol_spike_multiplier)

        # ── Consolidation ──
        df["range_high"] = df["high"].rolling(window=self.cfg.consolidation_bars).max()
        df["range_low"] = df["low"].rolling(window=self.cfg.consolidation_bars).min()
        df["price_range"] = df["range_high"] - df["range_low"]
        df["avg_range"] = df["price_range"].rolling(window=self.cfg.consolidation_bars).mean()

        # ── RSI + smoothed lines ──
        rsi_vals = _rsi_series(closes, self.rsi_period)
        df["rsi"] = pd.Series(rsi_vals, dtype=float)

        # Filter out None values for EMA/WMA computation
        rsi_clean = [v if v is not None else 50.0 for v in rsi_vals]
        rsi_ema = _ema(rsi_clean, self.rsi_ema_period)
        rsi_wma = _wma(rsi_clean, self.rsi_wma_period)

        df["rsi_ema3"] = _align(rsi_ema, n, self.rsi_ema_period)
        df["rsi_wma21"] = _align(rsi_wma, n, self.rsi_wma_period)

        # Momentum crossover
        df["momentum_bull"] = (df["rsi_ema3"] > df["rsi_wma21"])
        df["momentum_bear"] = (df["rsi_ema3"] < df["rsi_wma21"])

        # ── ATR ──
        candle_dicts = df[["high", "low", "close"]].to_dict("records")
        atr_vals = []
        for i in range(len(candle_dicts)):
            if i < self.atr_period:
                atr_vals.append(candle_dicts[i]["high"] - candle_dicts[i]["low"])
            else:
                atr_vals.append(atr(candle_dicts[max(0, i - self.atr_period * 2): i + 1], self.atr_period))
        df["atr"] = pd.Series(atr_vals, dtype=float)

        # ── Candle color ──
        df["is_green"] = df["close"] > df["open"]
        df["is_red"] = df["close"] < df["open"]

        # ── Session tracking (per trading day) ──
        # Detect day boundaries from timestamps
        df["_day"] = df["timestamp"].astype(str).str[:10] if "timestamp" in df.columns else ""
        df["day_open"] = df.groupby("_day")["open"].transform("first")

        # Day high so far (expanding max within each day)
        df["day_high_so_far"] = df.groupby("_day")["high"].cummax()

        # First bounce detection:
        #   1. Price must have gone up from open (made a day high)
        #   2. Then pulled back (retracement = red candles)
        #   3. Current bar is green after those red candles
        # We track this via a simple state machine per day
        df["bounce_phase"] = "none"  # "none" | "up" | "retrace" | "signal"
        _compute_bounce_phase(df)

        # ── EMA proximity (for EMA bounce detection) ──
        # How close is the low to each EMA (as % of price)
        for ema_col in ("ema_fast", "ema_mid", "ema_slow"):
            df[f"{ema_col}_dist_pct"] = ((df["low"] - df[ema_col]) / df[ema_col] * 100).abs()
            # Touched: low went to or below EMA, close recovered above
            df[f"{ema_col}_touched"] = (df["low"] <= df[ema_col]) & (df["close"] > df[ema_col])

        return df

    def detect_phase(self, df: pd.DataFrame, idx: int = -1) -> CyclePhase:
        """Detect cycle phase at a given bar index."""
        if len(df) < max(self.cfg.ema_slow, self.cfg.stdev_period) + 5:
            return CyclePhase.NONE

        if idx < 0:
            idx = len(df) + idx
        if idx < 2:
            return CyclePhase.NONE

        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        prev2 = df.iloc[idx - 2]

        close = row["close"]
        ema_f = row["ema_fast"]
        ema_m = row["ema_mid"]

        if pd.isna(ema_f) or pd.isna(ema_m) or pd.isna(row.get("ema_slow")):
            return CyclePhase.NONE

        vol_spike = row.get("vol_spike", False)
        vol_above = row["volume"] > row.get("vol_avg", 0) if pd.notna(row.get("vol_avg")) else False
        upper = row.get("upper_ext")
        lower = row.get("lower_ext")

        above_emas = close > ema_f and close > ema_m
        below_emas = close < ema_f and close < ema_m
        prev_above = prev["close"] > prev["ema_fast"] and prev["close"] > prev["ema_mid"]
        prev_below = prev["close"] < prev["ema_fast"] and prev["close"] < prev["ema_mid"]

        # RE
        if pd.notna(lower) and close < lower and vol_spike and below_emas:
            return CyclePhase.REVERSAL_EXTENSION
        # EX
        if pd.notna(upper) and close > upper and vol_spike and above_emas:
            return CyclePhase.EXHAUSTION_EXTENSION
        # WP
        if above_emas and not prev_above and vol_spike:
            return CyclePhase.WEDGE_POP
        # WD
        if below_emas and not prev_below and vol_above:
            return CyclePhase.WEDGE_DROP
        # EC bull
        if min(ema_f, ema_m) <= close <= max(ema_f, ema_m) and prev_above and close > prev2["close"]:
            return CyclePhase.EMA_CROSSBACK_BULL
        # EC bear
        if min(ema_f, ema_m) <= close <= max(ema_f, ema_m) and prev_below and close < prev2["close"]:
            return CyclePhase.EMA_CROSSBACK_BEAR

        # Basin breaks
        if idx >= self.cfg.consolidation_bars + 2:
            start = idx - self.cfg.consolidation_bars
            five_high = df["high"].iloc[start:idx].max()
            five_low = df["low"].iloc[start:idx].min()
            prev_pr = df.iloc[idx - 1].get("price_range", float("inf"))
            prev_ar = df.iloc[idx - 1].get("avg_range", 1)
            was_cons = pd.notna(prev_pr) and pd.notna(prev_ar) and prev_ar > 0 and prev_pr < prev_ar * 0.5

            if was_cons and close > five_high and vol_spike and close > ema_m:
                return CyclePhase.BASIN_BREAK_BULL
            if was_cons and close < five_low and vol_above and close < ema_m:
                return CyclePhase.BASIN_BREAK_BEAR

        return CyclePhase.NONE

    def detect_trend(self, row: pd.Series) -> TrendState:
        ef, em, es = row.get("ema_fast"), row.get("ema_mid"), row.get("ema_slow")
        if pd.isna(ef) or pd.isna(em) or pd.isna(es):
            return TrendState.NEUTRAL
        if ef > em > es:
            return TrendState.BULLISH
        if ef < em < es:
            return TrendState.BEARISH
        return TrendState.NEUTRAL

    def scan(
        self,
        symbol: str,
        candles: List[dict],
        min_rr: float = 2.0,
    ) -> List[IntradayCycleResult]:
        """
        Walk through all bars and return confirmed signals.

        Three signal types:
          1. OK cycle phases (WP/EC/BB/WD/EC_BEAR/BB_BEAR) + RSI momentum
          2. GREEN_AFTER_RED — green candle after red retracement post first bounce
          3. EMA_BOUNCE_3M — bounce off EMA line(s) with bullish momentum
        """
        df = self.compute(candles)
        if df.empty:
            return []

        from trading.swing.ok_cycles import BULLISH_ACTIONABLE, BEARISH_ACTIONABLE

        signals: List[IntradayCycleResult] = []
        warmup = max(self.cfg.ema_slow, self.rsi_wma_period, self.cfg.stdev_period) + 5

        for i in range(warmup, len(df)):
            row = df.iloc[i]
            trend = self.detect_trend(row)
            mom_bull = bool(row.get("momentum_bull", False))
            mom_bear = bool(row.get("momentum_bear", False))
            close = row["close"]
            atr_val = row.get("atr", 0)
            ts = str(row.get("timestamp", row.get("date", "")))

            signal = None

            # ── Type 1: OK Cycle Phase ──
            phase = self.detect_phase(df, i)
            if phase != CyclePhase.NONE:
                is_buy = phase in BULLISH_ACTIONABLE
                is_short = phase in BEARISH_ACTIONABLE
                if is_buy and mom_bull:
                    signal = self._build_signal(
                        symbol, ts, phase, trend, "BUY", phase.value,
                        row, atr_val, min_rr, mom_bull, mom_bear,
                    )
                elif is_short and mom_bear:
                    signal = self._build_signal(
                        symbol, ts, phase, trend, "SHORT", phase.value,
                        row, atr_val, min_rr, mom_bull, mom_bear,
                    )

            # ── Type 2: Green After Red (retracement buy) ──
            if signal is None and row.get("bounce_phase") == "signal" and mom_bull:
                # Additional filters: price above EMA50, RSI not overbought
                ema_s = row.get("ema_slow")
                rsi_val = row.get("rsi", 50) or 50
                if pd.notna(ema_s) and close > ema_s and rsi_val < 70:
                    signal = self._build_signal(
                        symbol, ts, CyclePhase.NONE, trend, "BUY",
                        TRADE_TYPE_GREEN_AFTER_RED,
                        row, atr_val, min_rr, mom_bull, mom_bear,
                    )

            # ── Type 3: EMA Bounce (3m) ──
            if signal is None and mom_bull:
                # Buy when low touches any EMA and close recovers above it
                # Price must be in an uptrend (EMA stacking bullish)
                if trend == TrendState.BULLISH:
                    ema_touched = (
                        row.get("ema_fast_touched", False)
                        or row.get("ema_mid_touched", False)
                        or row.get("ema_slow_touched", False)
                    )
                    if ema_touched and row.get("is_green", False):
                        rsi_val = row.get("rsi", 50) or 50
                        if rsi_val > 40:  # Not in oversold collapse
                            signal = self._build_signal(
                                symbol, ts, CyclePhase.NONE, trend, "BUY",
                                TRADE_TYPE_EMA_BOUNCE,
                                row, atr_val, min_rr, mom_bull, mom_bear,
                            )

            if signal is not None:
                signals.append(signal)

        return signals

    def _build_signal(
        self,
        symbol: str,
        timestamp: str,
        phase: CyclePhase,
        trend: TrendState,
        action: str,
        trade_type: str,
        row: pd.Series,
        atr_val: float,
        min_rr: float,
        mom_bull: bool,
        mom_bear: bool,
    ) -> IntradayCycleResult:
        """Build a signal with risk levels."""
        close = row["close"]

        if action == "BUY":
            sl = close - (atr_val * self.sl_atr_mult)
            risk = close - sl
            target = close + (risk * min_rr) if risk > 0 else close
        else:
            sl = close + (atr_val * self.sl_atr_mult)
            risk = sl - close
            target = close - (risk * min_rr) if risk > 0 else close

        return IntradayCycleResult(
            symbol=symbol,
            timestamp=timestamp,
            phase=phase,
            trend=trend,
            action=action,
            trade_type=trade_type,
            rsi=round(row.get("rsi", 50) or 50, 2),
            rsi_ema3=round(row.get("rsi_ema3", 50) or 50, 2),
            rsi_wma21=round(row.get("rsi_wma21", 50) or 50, 2),
            momentum_bullish=mom_bull,
            momentum_bearish=mom_bear,
            close=round(close, 2),
            ema_fast=round(row["ema_fast"], 2) if pd.notna(row["ema_fast"]) else 0,
            ema_mid=round(row["ema_mid"], 2) if pd.notna(row["ema_mid"]) else 0,
            ema_slow=round(row["ema_slow"], 2) if pd.notna(row["ema_slow"]) else 0,
            atr=round(atr_val, 2),
            sl=round(sl, 2),
            target=round(target, 2),
            risk_points=round(risk, 2) if risk > 0 else 0,
        )


def _compute_bounce_phase(df: pd.DataFrame):
    """Tag each bar's bounce phase within its trading day.

    State machine per day:
      none    → price hasn't moved up from open yet
      up      → price made a meaningful move up (high > open + threshold)
      retrace → after 'up', price is pulling back (consecutive red candles)
      signal  → first green candle during/after retrace = entry candidate
    """
    phases = ["none"] * len(df)
    days = df["_day"].astype(str).tolist() if "_day" in df.columns else [""] * len(df)
    opens = df["open"].tolist()
    highs = df["high"].tolist()
    greens = df["is_green"].tolist()
    reds = df["is_red"].tolist()

    prev_day = ""
    phase = "none"
    day_open = 0.0
    day_high = 0.0
    red_count = 0
    bounced = False

    for i in range(len(df)):
        day = days[i]

        if day != prev_day:
            prev_day = day
            phase = "none"
            day_open = opens[i]
            day_high = highs[i]
            red_count = 0
            bounced = False

        day_high = max(day_high, highs[i])

        if phase == "none":
            if day_open > 0 and day_high > day_open * 1.003:
                phase = "up"

        elif phase == "up":
            if reds[i]:
                phase = "retrace"
                red_count = 1

        elif phase == "retrace":
            if reds[i]:
                red_count += 1
            elif greens[i] and red_count >= 1 and not bounced:
                phase = "signal"
                bounced = True

        elif phase == "signal":
            phase = "up"

        phases[i] = phase

    df["bounce_phase"] = phases


def _align(series: List[float], total: int, period: int) -> pd.Series:
    """Pad shorter indicator series with NaN to align with DataFrame."""
    pad = total - len(series)
    return pd.Series([None] * pad + series, dtype=float)
