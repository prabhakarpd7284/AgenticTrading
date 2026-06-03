"""
v2 factor scoring — eight orthogonal, dimensionless sub-scores (each 0..1).

Every function here is pure numeric math over primitives (no pandas, no IO),
so each factor is unit-testable in isolation. The engine extracts the raw
inputs from its indicator frame and calls these; the composite score is a
weighted sum (weights live in config.FactorWeights).

Design principle: each factor answers one independent question, so a weak
score in one dimension can't be masked by another. The raw inputs are kept
on the result so a human (or the Pine table) can see *why* a setup graded
the way it did.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict


def clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _tanh01(x: float, scale: float) -> float:
    """Map a signed magnitude to 0..1 via a half-tanh (0 → 0.5, +∞ → 1)."""
    return 0.5 * (1.0 + math.tanh(x / scale)) if scale else 0.5


# ── 1. Regime ─────────────────────────────────────────────────────────
# Also the hard directional gate. Longs need an up-regime, shorts a
# down-regime, on BOTH the primary timeframe and the HTF.

def regime_score(primary: str, htf: str, direction: str) -> float:
    """
    primary / htf ∈ {"bullish","bearish","neutral"}; direction ∈ {"long","short"}.
    """
    want = "bullish" if direction == "long" else "bearish"
    opp = "bearish" if direction == "long" else "bullish"
    if primary == want and htf == want:
        return 1.0
    if primary == want and htf == "neutral":
        return 0.65
    if primary == "neutral" and htf == want:
        return 0.5
    if primary == want and htf == opp:
        return 0.2          # primary leads HTF — counter-trend, low conviction
    if primary == "neutral" and htf == "neutral":
        return 0.25
    return 0.0              # primary against the trade → gate fails


# ── 2. Momentum ───────────────────────────────────────────────────────

def momentum_score(
    rsi: float, rsi_ema3: float, rsi_wma21: float, roc: float, direction: str
) -> float:
    """
    Blend three independent momentum reads:
      • RSI regime  — is RSI on the right side of 50, with room to run?
      • RSI cross   — fast smoothing above/below slow smoothing (turn).
      • ROC         — raw rate of change, sign + magnitude.
    """
    if direction == "long":
        regime = clamp01((rsi - 45) / 35)          # 45→0, 80→1
        cross = clamp01(0.5 + (rsi_ema3 - rsi_wma21) / 20)
        roc_c = _tanh01(roc, 6.0)
    else:
        regime = clamp01((55 - rsi) / 35)          # 55→0, 20→1
        cross = clamp01(0.5 + (rsi_wma21 - rsi_ema3) / 20)
        roc_c = _tanh01(-roc, 6.0)
    return clamp01(0.4 * regime + 0.35 * cross + 0.25 * roc_c)


# ── 3. Relative strength vs benchmark (Mansfield slope) ───────────────

def rel_strength_score(rs_slope_pct: float, direction: str) -> float:
    """
    rs_slope_pct: % change of the RS line (stock/benchmark) over rs_lookback.
    Positive = outperforming. Longs want it up, shorts want it down.
    """
    signed = rs_slope_pct if direction == "long" else -rs_slope_pct
    return _tanh01(signed, 4.0)


# ── 4. Extension (ATR units) ──────────────────────────────────────────
# Replaces the scale-dependent stdev bands. We measure how far price sits
# from the anchor EMA in ATR units. For ENTRY setups, being near the EMA is
# good (better R:R, not chasing); being many ATRs out is exhaustion.

def extension_score(ext_atr: float, direction: str, ideal: float, chase: float) -> float:
    """
    ext_atr: (close - ema_fast) / atr  (signed; + above EMA).
    For longs, the "good" side is small-positive to mildly-negative (pullback
    to support). Penalise being > `chase` ATRs in the trade direction.
    """
    d = ext_atr if direction == "long" else -ext_atr
    # Deep counter-side pullbacks (d very negative) are also risky (knife).
    if d < -chase:
        return 0.1
    if d <= ideal:
        # Near/below the EMA up to the ideal band → best entries.
        return clamp01(0.85 + 0.15 * (1 - abs(d) / max(ideal, 1e-9)))
    # Beyond ideal, decay linearly to ~0 at the chase threshold.
    span = max(chase - ideal, 1e-9)
    return clamp01(0.85 * (1 - (d - ideal) / span))


# ── 5. Base / VCP contraction quality (TIME factor) ───────────────────

def base_score(contraction_ratio: float, base_len: int, min_bars: int,
               target_ratio: float) -> float:
    """
    contraction_ratio: recent_range / earlier_range over the base window
                       (< 1 means it's tightening — VCP-style).
    base_len:          how many bars the base has held.
    Longer + tighter bases (more *time* absorbing supply) score higher.
    """
    if base_len < min_bars:
        return clamp01(0.2 * base_len / max(min_bars, 1))
    # Tightness: fully tight at/below target_ratio, 0 by ratio≈1.0.
    span = max(1.0 - target_ratio, 1e-9)
    tight = clamp01((1.0 - contraction_ratio) / span)
    # Length: saturates around 3× the minimum base length.
    length = clamp01(base_len / (3.0 * min_bars))
    return clamp01(0.6 * tight + 0.4 * length)


# ── 6. Volume confirmation ────────────────────────────────────────────

def volume_score(breakout_ratio: float, base_dryup_ratio: float,
                 needs_expansion: bool, breakout_mult: float,
                 dryup_mult: float) -> float:
    """
    breakout_ratio:    current vol / median vol.
    base_dryup_ratio:  median vol during the base / overall median.
    needs_expansion:   breakout/wedge phases want expansion; pullback
                       (crossback) phases want quiet dry-up instead.
    """
    if needs_expansion:
        return clamp01((breakout_ratio - 1.0) / max(breakout_mult - 1.0, 1e-9))
    # Pullback: reward dry-up (low base volume) + no distribution spike.
    dryup = clamp01((1.0 - base_dryup_ratio) / max(1.0 - dryup_mult, 1e-9))
    calm = clamp01(1.0 - max(0.0, breakout_ratio - 1.2) / 1.0)
    return clamp01(0.6 * dryup + 0.4 * calm)


# ── 7. Freshness (TIME factor) ────────────────────────────────────────

def freshness_score(bars_since_phase: int, max_bars: int) -> float:
    """Linear decay: signal on this bar = 1.0, stale by max_bars = 0.0."""
    if bars_since_phase <= 0:
        return 1.0
    if bars_since_phase >= max_bars:
        return 0.0
    return clamp01(1.0 - bars_since_phase / max_bars)


# ── 8. Risk geometry ──────────────────────────────────────────────────

def risk_geometry_score(risk_pct: float, rr: float, rr_target: float) -> float:
    """
    risk_pct: structural stop distance as % of entry (tighter = better).
    rr:       reward:risk to the logical target.
    """
    # Tight stops (≤2% of price) ideal; 8%+ poor.
    tight = clamp01((0.08 - risk_pct) / 0.06)
    rr_c = clamp01(rr / max(rr_target, 1e-9))
    return clamp01(0.5 * tight + 0.5 * rr_c)


# ══════════════════════════════════════════════════════════════════════
# Aggregate
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FactorScores:
    """The 8 normalised sub-scores + the composite."""
    regime: float = 0.0
    momentum: float = 0.0
    rel_strength: float = 0.0
    extension: float = 0.0
    base: float = 0.0
    volume: float = 0.0
    freshness: float = 0.0
    risk_geometry: float = 0.0
    raw: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return {
            "regime": round(self.regime, 3),
            "momentum": round(self.momentum, 3),
            "rel_strength": round(self.rel_strength, 3),
            "extension": round(self.extension, 3),
            "base": round(self.base, 3),
            "volume": round(self.volume, 3),
            "freshness": round(self.freshness, 3),
            "risk_geometry": round(self.risk_geometry, 3),
        }

    def composite(self, weights: Dict[str, float]) -> float:
        s = self.as_dict()
        total = sum(weights.values()) or 1.0
        return round(sum(s[k] * w for k, w in weights.items()) / total, 4)
