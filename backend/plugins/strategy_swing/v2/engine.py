"""
v2 engine — robust Oliver-Kell cycle detection + factor scoring + money model.

Pure over candle lists (no broker IO), so it's fully unit-testable. The
scanner layer handles fetching; this layer turns OHLCV + HTF + benchmark
into a graded, risk-defined SwingSignalV2.

What's different from v1 (plugins.strategy_swing.ok_cycles):
  • Regime is a hard gate (primary + HTF EMA stack *and slope*), not a bonus.
  • Extension is ATR-normalised, not a raw-stdev band.
  • A real multi-bar contraction base is required for pop/breakout phases.
  • Phases decay: a trigger older than freshness_max_bars is ignored (no
    sticky "current_phase" that never resets).
  • Exhaustion in an uptrend is TRIM, never an auto-short.
  • Every actionable signal carries entry / structural stop / measured target /
    R-multiple / fixed-fractional size / time-stop.
  • Composite grade from 8 transparent factors instead of an ad-hoc score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

from trading.utils.indicators import _ema, _wma, _rsi_series
from plugins.strategy_swing.v2.config import SwingV2Config, get_tier_config, SwingTier
from plugins.strategy_swing.v2 import factors as F


# ══════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════

class Regime(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Phase(str, Enum):
    REVERSAL_EXTENSION = "RE"        # washout below — WATCH for long setup
    WEDGE_POP = "WP"                 # reclaim of EMAs out of a base — long
    EMA_CROSSBACK = "EC"             # pullback to rising EMA — long (best R:R)
    BASIN_BREAK = "BB"               # breakout from contraction base — long/add
    EXHAUSTION_EXTENSION = "EX"      # over-extended up — TRIM/trail
    WEDGE_DROP = "WD"                # breakdown of EMAs out of a base — short
    EMA_CROSSBACK_BEAR = "ECB"       # failed bounce to falling EMA — short
    BASIN_BREAK_BEAR = "BBB"         # breakdown from contraction base — short
    NONE = "NONE"


class Action(str, Enum):
    BUY = "BUY"
    ADD = "ADD"
    SHORT = "SHORT"
    TRIM = "TRIM"
    WATCH = "WATCH"
    AVOID = "AVOID"
    NONE = "—"


# Phase metadata: direction it implies, and whether it wants volume expansion
# (breakout/breakdown) vs quiet dry-up (pullback).
_LONG_ENTRY = {Phase.WEDGE_POP, Phase.EMA_CROSSBACK, Phase.BASIN_BREAK}
_SHORT_ENTRY = {Phase.WEDGE_DROP, Phase.EMA_CROSSBACK_BEAR, Phase.BASIN_BREAK_BEAR}
_EXPANSION_PHASES = {Phase.WEDGE_POP, Phase.BASIN_BREAK, Phase.WEDGE_DROP,
                     Phase.BASIN_BREAK_BEAR}
_PHASE_LABEL = {
    Phase.REVERSAL_EXTENSION: "Reversal Extension",
    Phase.WEDGE_POP: "Wedge Pop",
    Phase.EMA_CROSSBACK: "EMA Crossback",
    Phase.BASIN_BREAK: "Basin Break",
    Phase.EXHAUSTION_EXTENSION: "Exhaustion Extension",
    Phase.WEDGE_DROP: "Wedge Drop",
    Phase.EMA_CROSSBACK_BEAR: "Bear EMA Crossback",
    Phase.BASIN_BREAK_BEAR: "Bear Basin Break",
    Phase.NONE: "—",
}


def phase_direction(phase: Phase) -> Optional[str]:
    if phase in _LONG_ENTRY or phase == Phase.REVERSAL_EXTENSION:
        return "long"
    if phase in _SHORT_ENTRY:
        return "short"
    if phase == Phase.EXHAUSTION_EXTENSION:
        return "long"   # exhaustion of an up-move (manage a long)
    return None


# ══════════════════════════════════════════════════════════════════════
# Result
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SwingSignalV2:
    symbol: str
    tier: str
    timestamp: str = ""
    phase: Phase = Phase.NONE
    phase_label: str = "—"
    action: Action = Action.NONE
    direction: str = ""                 # long | short | ""

    # Regime
    regime_primary: Regime = Regime.NEUTRAL
    regime_htf: Regime = Regime.NEUTRAL
    aligned: bool = False

    # Snapshot
    close: float = 0.0
    ema_fast: float = 0.0
    ema_mid: float = 0.0
    ema_slow: float = 0.0
    atr: float = 0.0
    ext_atr: float = 0.0
    rsi: float = 50.0
    rsi_ema3: float = 50.0
    rsi_wma21: float = 50.0
    roc: float = 0.0
    rs_slope: float = 0.0

    # Structure / time
    bars_since_phase: int = 0
    base_len: int = 0
    contraction_ratio: float = 1.0
    vol_ratio: float = 0.0

    # Money model
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    target_measured: float = 0.0
    risk_per_share: float = 0.0
    risk_pct: float = 0.0
    rr: float = 0.0
    qty: int = 0
    notional: float = 0.0
    time_stop_bars: int = 0

    # Score
    score: float = 0.0
    grade: str = "—"                    # A | B | C | — (not actionable)
    factors: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def actionable(self) -> bool:
        return self.action in (Action.BUY, Action.ADD, Action.SHORT) and self.grade in ("A", "B", "C")

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "tier": self.tier, "timestamp": self.timestamp,
            "phase": self.phase.value, "phase_label": self.phase_label,
            "action": self.action.value, "direction": self.direction,
            "regime_primary": self.regime_primary.value, "regime_htf": self.regime_htf.value,
            "aligned": self.aligned, "close": self.close,
            "ema_fast": self.ema_fast, "ema_mid": self.ema_mid, "ema_slow": self.ema_slow,
            "atr": self.atr, "ext_atr": self.ext_atr, "rsi": self.rsi,
            "rsi_ema3": self.rsi_ema3, "rsi_wma21": self.rsi_wma21, "roc": self.roc,
            "rs_slope": self.rs_slope, "bars_since_phase": self.bars_since_phase,
            "base_len": self.base_len, "contraction_ratio": self.contraction_ratio,
            "vol_ratio": self.vol_ratio, "entry": self.entry, "stop": self.stop,
            "target": self.target, "target_measured": self.target_measured,
            "risk_per_share": self.risk_per_share, "risk_pct": self.risk_pct,
            "rr": self.rr, "qty": self.qty, "notional": self.notional,
            "time_stop_bars": self.time_stop_bars, "score": self.score,
            "grade": self.grade, "factors": self.factors, "reasons": self.reasons,
            "error": self.error,
        }


# ══════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════

class SwingEngineV2:
    def __init__(self, cfg: Optional[SwingV2Config] = None, capital: float = 500_000.0):
        self.cfg = cfg or get_tier_config(SwingTier.MEDIUM)
        self.capital = capital

    # ── indicator frame ───────────────────────────────────────────────

    def compute(self, candles: List[dict]) -> pd.DataFrame:
        c = self.cfg
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        if "timestamp" not in df.columns and "date" in df.columns:
            df["timestamp"] = df["date"]
        closes = df["close"].astype(float).tolist()
        n = len(closes)

        df["ema_fast"] = _pad(_ema(closes, c.ema_fast), n)
        df["ema_mid"] = _pad(_ema(closes, c.ema_mid), n)
        df["ema_slow"] = _pad(_ema(closes, c.ema_slow), n)

        # ATR (Wilder), as a full aligned series
        df["atr"] = _atr_series(df, c.atr_period)
        df["ext_atr"] = (df["close"] - df["ema_fast"]) / df["atr"].replace(0, math.nan)

        # Slope of the slow EMA over 5 bars (% per bar)
        df["ema_slow_slope"] = df["ema_slow"].pct_change(periods=5) * 100.0

        # Momentum
        rsi_vals = _rsi_series(closes, c.rsi_period)
        rsi_clean = [v if v is not None else 50.0 for v in rsi_vals]
        df["rsi"] = pd.Series(rsi_vals, dtype=float)
        df["rsi_ema3"] = _pad(_ema(rsi_clean, c.rsi_ema_period), n)
        df["rsi_wma21"] = _pad(_wma(rsi_clean, c.rsi_wma_period), n)
        df["roc"] = df["close"].pct_change(periods=c.roc_period) * 100.0

        # Volume — median is robust to single-bar spikes
        df["vol_median"] = df["volume"].rolling(window=c.vol_median_period).median()
        df["vol_ratio"] = df["volume"] / df["vol_median"].replace(0, math.nan)

        # Swing pivots (prior bars only — exclude current to avoid look-ahead)
        df["swing_high"] = df["high"].shift(1).rolling(window=c.pivot_lookback).max()
        df["swing_low"] = df["low"].shift(1).rolling(window=c.pivot_lookback).min()

        df["above_emas"] = (df["close"] > df["ema_fast"]) & (df["close"] > df["ema_mid"])
        df["below_emas"] = (df["close"] < df["ema_fast"]) & (df["close"] < df["ema_mid"])
        return df

    # ── regime ────────────────────────────────────────────────────────

    @staticmethod
    def regime_at(ef, em, es, slope) -> Regime:
        if any(pd.isna(x) for x in (ef, em, es, slope)):
            return Regime.NEUTRAL
        if ef > em > es and slope >= -0.02:
            return Regime.BULLISH
        if ef < em < es and slope <= 0.02:
            return Regime.BEARISH
        return Regime.NEUTRAL

    # ── base / contraction metrics ────────────────────────────────────

    def _base_metrics(self, df: pd.DataFrame, idx: int) -> tuple[float, int]:
        """Return (contraction_ratio, base_len) for the base preceding bar idx."""
        c = self.cfg
        w = c.base_window
        if idx < 2 * w + 1:
            return 1.0, 0
        recent_hi = df["high"].iloc[idx - w:idx].max()
        recent_lo = df["low"].iloc[idx - w:idx].min()
        earlier_hi = df["high"].iloc[idx - 2 * w:idx - w].max()
        earlier_lo = df["low"].iloc[idx - 2 * w:idx - w].min()
        recent_range = recent_hi - recent_lo
        earlier_range = earlier_hi - earlier_lo
        ratio = (recent_range / earlier_range) if earlier_range > 0 else 1.0

        # base_len: tight bars in the recent window (TR below earlier median TR)
        earlier_tr = (df["high"].iloc[idx - 2 * w:idx - w]
                      - df["low"].iloc[idx - 2 * w:idx - w])
        thresh = earlier_tr.median()
        recent_tr = (df["high"].iloc[idx - w:idx] - df["low"].iloc[idx - w:idx])
        base_len = int((recent_tr <= thresh).sum()) if thresh and thresh == thresh else 0
        return float(ratio), base_len

    # ── phase detection at a single bar ───────────────────────────────

    def detect_phase(self, df: pd.DataFrame, idx: int) -> Phase:
        c = self.cfg
        if idx < c.warmup_bars():
            return Phase.NONE
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        prev2 = df.iloc[idx - 2]

        close = row["close"]
        ef, em, es = row["ema_fast"], row["ema_mid"], row["ema_slow"]
        if any(pd.isna(x) for x in (ef, em, es)):
            return Phase.NONE
        ext_atr = row["ext_atr"]
        vol_ratio = row["vol_ratio"] if pd.notna(row["vol_ratio"]) else 0.0
        swing_hi = row["swing_high"]
        swing_lo = row["swing_low"]
        above = bool(row["above_emas"])
        below = bool(row["below_emas"])
        prev_above = bool(prev["above_emas"])
        prev_below = bool(prev["below_emas"])

        ratio, base_len = self._base_metrics(df, idx)
        has_base = base_len >= c.base_min_bars and ratio <= c.contraction_ratio
        expansion = vol_ratio >= c.vol_breakout_mult * 0.85
        in_ema_zone = min(ef, em) <= close <= max(ef, em)

        # 1. Extensions first (management / watch) ------------------------
        if pd.notna(ext_atr):
            if ext_atr >= c.ext_atr_chase and above:
                return Phase.EXHAUSTION_EXTENSION
            if ext_atr <= -c.ext_atr_chase and below:
                return Phase.REVERSAL_EXTENSION

        # 2. Long breakouts ----------------------------------------------
        if pd.notna(swing_hi) and close > swing_hi and has_base and expansion and close > em:
            return Phase.BASIN_BREAK
        if above and not prev_above and has_base and expansion:
            return Phase.WEDGE_POP
        # 3. Long pullback (best R:R) — pull into rising EMA zone, turn up
        if (in_ema_zone and es > 0 and close > es
                and row["ema_slow_slope"] >= 0
                and close > prev["close"] and prev["close"] <= prev2["close"]):
            return Phase.EMA_CROSSBACK

        # 4. Short breakdowns --------------------------------------------
        if pd.notna(swing_lo) and close < swing_lo and has_base and expansion and close < em:
            return Phase.BASIN_BREAK_BEAR
        if below and not prev_below and has_base and expansion:
            return Phase.WEDGE_DROP
        if (in_ema_zone and es > 0 and close < es
                and row["ema_slow_slope"] <= 0
                and close < prev["close"] and prev["close"] >= prev2["close"]):
            return Phase.EMA_CROSSBACK_BEAR

        return Phase.NONE

    # ── relative strength ───────────────────────────────────────────────

    def _rs_slope(self, df: pd.DataFrame, idx: int,
                  benchmark: Optional[List[dict]]) -> float:
        c = self.cfg
        if not benchmark or idx < c.rs_lookback:
            return 0.0
        bench = {str(b.get("timestamp", b.get("date", ""))): float(b["close"])
                 for b in benchmark}
        # RS at window start and end (match by timestamp; skip unmatched)
        def rs_at(i: int) -> Optional[float]:
            ts = str(df.iloc[i].get("timestamp", df.iloc[i].get("date", "")))
            bc = bench.get(ts)
            return (df.iloc[i]["close"] / bc) if bc else None
        rs_end = rs_at(idx)
        rs_start = rs_at(idx - c.rs_lookback)
        if rs_end and rs_start and rs_start != 0:
            return round((rs_end / rs_start - 1.0) * 100.0, 3)
        return 0.0

    # ── full analysis ───────────────────────────────────────────────────

    def analyze(
        self,
        symbol: str,
        candles: List[dict],
        htf_candles: Optional[List[dict]] = None,
        benchmark_candles: Optional[List[dict]] = None,
    ) -> SwingSignalV2:
        c = self.cfg
        sig = SwingSignalV2(symbol=symbol, tier=c.tier.value)
        if not candles or len(candles) < c.warmup_bars():
            sig.error = f"insufficient data: {len(candles) if candles else 0} bars"
            return sig

        df = self.compute(candles)
        last = len(df) - 1
        row = df.iloc[last]

        # Primary regime
        reg_p = self.regime_at(row["ema_fast"], row["ema_mid"], row["ema_slow"],
                               row["ema_slow_slope"])
        # HTF regime
        reg_h = Regime.NEUTRAL
        if htf_candles and len(htf_candles) >= c.ema_slow + 6:
            hdf = self.compute(htf_candles)
            hl = hdf.iloc[-1]
            reg_h = self.regime_at(hl["ema_fast"], hl["ema_mid"], hl["ema_slow"],
                                   hl["ema_slow_slope"])

        # Find the freshest actionable phase within the freshness window
        phase = Phase.NONE
        bars_since = 0
        for back in range(0, c.freshness_max_bars + 1):
            i = last - back
            if i < c.warmup_bars():
                break
            p = self.detect_phase(df, i)
            if p != Phase.NONE:
                phase, bars_since = p, back
                break

        ratio, base_len = self._base_metrics(df, last)
        rs_slope = self._rs_slope(df, last, benchmark_candles)

        self._populate_snapshot(sig, row, reg_p, reg_h, phase, bars_since,
                                ratio, base_len, rs_slope)

        if phase == Phase.NONE:
            sig.action = Action.NONE
            return sig

        direction = phase_direction(phase)
        sig.direction = direction or ""

        # Resolve action with the regime gate
        sig.action = self._resolve_action(phase, direction, reg_p, reg_h)
        if sig.action in (Action.WATCH, Action.TRIM, Action.AVOID, Action.NONE):
            # Non-entry phases still report, but carry no money model / grade.
            sig.reasons.append(self._action_reason(phase, sig.action, reg_p, reg_h))
            return sig

        # Money model for entries
        self._build_money_model(sig, df, last, direction)

        # Factor scorecard
        self._score(sig, phase, direction, reg_p, reg_h, row, rs_slope,
                    ratio, base_len, bars_since)
        return sig

    # ── helpers ─────────────────────────────────────────────────────────

    def _populate_snapshot(self, sig, row, reg_p, reg_h, phase, bars_since,
                           ratio, base_len, rs_slope):
        sig.timestamp = str(row.get("timestamp", row.get("date", "")))
        sig.phase = phase
        sig.phase_label = _PHASE_LABEL[phase]
        sig.regime_primary = reg_p
        sig.regime_htf = reg_h
        sig.aligned = (reg_p == reg_h and reg_p != Regime.NEUTRAL)
        sig.close = _r(row["close"])
        sig.ema_fast = _r(row["ema_fast"])
        sig.ema_mid = _r(row["ema_mid"])
        sig.ema_slow = _r(row["ema_slow"])
        sig.atr = _r(row["atr"])
        sig.ext_atr = _r(row["ext_atr"], 2)
        sig.rsi = _r(row["rsi"])
        sig.rsi_ema3 = _r(row["rsi_ema3"])
        sig.rsi_wma21 = _r(row["rsi_wma21"])
        sig.roc = _r(row["roc"], 2)
        sig.rs_slope = rs_slope
        sig.bars_since_phase = bars_since
        sig.base_len = base_len
        sig.contraction_ratio = round(ratio, 3)
        sig.vol_ratio = _r(row["vol_ratio"], 2)
        sig.time_stop_bars = self.cfg.max_hold_bars

    def _resolve_action(self, phase, direction, reg_p, reg_h) -> Action:
        c = self.cfg
        if phase in _LONG_ENTRY:
            # Gate: don't take a long when the regime is bearish on either TF.
            if reg_p == Regime.BEARISH or reg_h == Regime.BEARISH:
                return Action.AVOID
            return Action.ADD if reg_p == Regime.BULLISH and reg_h == Regime.BULLISH and phase == Phase.BASIN_BREAK else Action.BUY
        if phase in _SHORT_ENTRY:
            if reg_p == Regime.BULLISH or reg_h == Regime.BULLISH:
                return Action.AVOID
            return Action.SHORT
        if phase == Phase.EXHAUSTION_EXTENSION:
            return Action.TRIM
        if phase == Phase.REVERSAL_EXTENSION:
            return Action.WATCH
        return Action.NONE

    def _build_money_model(self, sig: SwingSignalV2, df, idx, direction):
        c = self.cfg
        row = df.iloc[idx]
        close = float(row["close"])
        atr = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
        em = float(row["ema_mid"])
        swing_hi = row["swing_high"]
        swing_lo = row["swing_low"]
        # base height for the measured move
        recent_hi = df["high"].iloc[max(0, idx - c.base_window):idx].max()
        recent_lo = df["low"].iloc[max(0, idx - c.base_window):idx].min()
        base_height = float(recent_hi - recent_lo) if recent_hi == recent_hi else atr

        if direction == "long":
            struct = min(em, float(swing_lo) if pd.notna(swing_lo) else em)
            struct_stop = struct * (1 - c.sl_buffer_pct)
            atr_stop = close - atr * c.sl_atr_mult
            stop = max(struct_stop, atr_stop)
            if stop >= close:
                stop = close - atr * c.sl_atr_mult
            risk = close - stop
            measured = close + base_height
            target = max(measured, close + risk * 1.5)
        else:
            struct = max(em, float(swing_hi) if pd.notna(swing_hi) else em)
            struct_stop = struct * (1 + c.sl_buffer_pct)
            atr_stop = close + atr * c.sl_atr_mult
            stop = min(struct_stop, atr_stop)
            if stop <= close:
                stop = close + atr * c.sl_atr_mult
            risk = stop - close
            measured = close - base_height
            target = min(measured, close - risk * 1.5)

        risk = max(risk, 1e-6)
        rr = abs(target - close) / risk
        risk_capital = self.capital * c.risk_per_trade_pct / 100.0
        qty = int(risk_capital / risk) if risk > 0 else 0
        max_notional = self.capital * c.max_position_pct / 100.0
        if qty * close > max_notional:
            qty = int(max_notional / close) if close > 0 else 0

        sig.entry = _r(close)
        sig.stop = _r(stop)
        sig.target = _r(target)
        sig.target_measured = _r(measured)
        sig.risk_per_share = _r(risk)
        sig.risk_pct = _r(risk / close * 100, 2) if close else 0.0
        sig.rr = round(rr, 2)
        sig.qty = qty
        sig.notional = _r(qty * close)

    def _score(self, sig, phase, direction, reg_p, reg_h, row, rs_slope,
               ratio, base_len, bars_since):
        c = self.cfg
        needs_expansion = phase in _EXPANSION_PHASES
        # base dry-up proxy: recent base median vol vs overall median (vol_ratio<1 ⇒ dry)
        base_dryup = min(1.0, float(sig.vol_ratio) if sig.vol_ratio else 1.0)

        fs = F.FactorScores()
        fs.regime = F.regime_score(reg_p.value, reg_h.value, direction)
        fs.momentum = F.momentum_score(sig.rsi, sig.rsi_ema3, sig.rsi_wma21, sig.roc, direction)
        fs.rel_strength = F.rel_strength_score(rs_slope, direction)
        fs.extension = F.extension_score(float(row["ext_atr"]) if pd.notna(row["ext_atr"]) else 0.0,
                                         direction, c.ext_atr_ideal, c.ext_atr_chase)
        fs.base = F.base_score(ratio, base_len, c.base_min_bars, c.contraction_ratio)
        fs.volume = F.volume_score(sig.vol_ratio or 0.0, base_dryup, needs_expansion,
                                   c.vol_breakout_mult, c.vol_dryup_mult)
        fs.freshness = F.freshness_score(bars_since, c.freshness_max_bars)
        fs.risk_geometry = F.risk_geometry_score(sig.risk_pct / 100.0, sig.rr, c.rr_target)

        sig.factors = fs.as_dict()
        sig.score = fs.composite(c.weights.as_dict())
        sig.grade = (
            "A" if sig.score >= c.grade_a else
            "B" if sig.score >= c.grade_b else
            "C" if sig.score >= c.grade_c else "—"
        )
        sig.reasons = self._reasons(sig, fs)

    @staticmethod
    def _reasons(sig: SwingSignalV2, fs: F.FactorScores) -> List[str]:
        r = [f"{sig.phase_label} ({sig.phase.value}) · {sig.direction}",
             f"regime {sig.regime_primary.value}/{sig.regime_htf.value}"
             f"{' aligned' if sig.aligned else ''}",
             f"R:R {sig.rr} · risk {sig.risk_pct}% · qty {sig.qty}"]
        top = sorted(fs.as_dict().items(), key=lambda kv: kv[1], reverse=True)[:3]
        r.append("top: " + ", ".join(f"{k} {v:.2f}" for k, v in top))
        if sig.rs_slope:
            r.append(f"RS {'+' if sig.rs_slope > 0 else ''}{sig.rs_slope}%")
        return r

    @staticmethod
    def _action_reason(phase, action, reg_p, reg_h) -> str:
        if action == Action.AVOID:
            return f"{_PHASE_LABEL[phase]} but regime {reg_p.value}/{reg_h.value} blocks the trade"
        if action == Action.TRIM:
            return "Exhaustion extension — trim / trail, do not chase"
        if action == Action.WATCH:
            return "Reversal extension — washout, watch for a base to form"
        return _PHASE_LABEL[phase]


# ══════════════════════════════════════════════════════════════════════
# small numeric helpers
# ══════════════════════════════════════════════════════════════════════

def _r(x, nd: int = 2) -> float:
    try:
        return round(float(x), nd) if x == x else 0.0
    except (TypeError, ValueError):
        return 0.0


def _pad(series: List[float], total: int) -> pd.Series:
    pad = total - len(series)
    if pad < 0:
        series = series[-total:]
        pad = 0
    return pd.Series([None] * pad + list(series), dtype=float)


def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder smoothing via EWM alpha=1/period
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
