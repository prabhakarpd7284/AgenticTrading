"""
v2 configuration — self-contained so the v2 engine has zero coupling to the
v1 OKCycleConfig in trading/config.py.

A SwingV2Config bundles every tunable for one *holding tier*. The three
presets (SMALL/MEDIUM/LONG) differ only in timeframe, lookback, and the
risk/time constants that scale with holding period — the detection logic and
factor weights are shared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class SwingTier(str, Enum):
    """Holding-period tier, selected by candle timeframe."""
    SMALL = "small"     # 15m candles → multi-day swings
    MEDIUM = "medium"   # 1h candles  → ~1–2 week swings
    LONG = "long"       # daily candles → weeks–months position swings


# ══════════════════════════════════════════════════════════════════════
# Factor weights — the composite score is a weighted sum of 8 orthogonal
# sub-scores, each normalised to 0..1. Weights sum to 1.0. These are the
# single place to recalibrate against enrich_signals outcome data later.
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FactorWeights:
    regime: float = 0.20        # primary + HTF trend alignment (also a hard gate)
    momentum: float = 0.15      # RSI regime + EMA3×WMA21 + ROC
    rel_strength: float = 0.15  # RS vs NIFTY (Mansfield slope)
    extension: float = 0.10     # ATR-normalised distance from anchor EMA (inverted for entries)
    base: float = 0.15          # VCP contraction quality + base length (TIME)
    volume: float = 0.10        # breakout volume vs median + dry-up
    freshness: float = 0.05     # bars-since-phase decay (TIME)
    risk_geometry: float = 0.10  # tightness of structural stop / resulting R:R

    def as_dict(self) -> Dict[str, float]:
        return {
            "regime": self.regime,
            "momentum": self.momentum,
            "rel_strength": self.rel_strength,
            "extension": self.extension,
            "base": self.base,
            "volume": self.volume,
            "freshness": self.freshness,
            "risk_geometry": self.risk_geometry,
        }

    def total(self) -> float:
        return sum(self.as_dict().values())


@dataclass(frozen=True)
class SwingV2Config:
    """All tunables for one holding tier."""
    tier: SwingTier

    # ── Data / timeframe ──
    interval: str                       # Angel One interval for the primary TF
    htf_interval: str                   # higher timeframe for the regime gate
    htf_from_weekly_agg: bool = False   # if True, derive HTF by weekly aggregation
    lookback_days: int = 120            # calendar days of history to fetch

    # ── EMAs (Kell's 10/20/50, applied on the tier's TF) ──
    ema_fast: int = 10
    ema_mid: int = 20
    ema_slow: int = 50

    # ── Momentum ──
    rsi_period: int = 14
    rsi_ema_period: int = 3
    rsi_wma_period: int = 21
    roc_period: int = 10                # rate-of-change lookback

    # ── Relative strength ──
    rs_lookback: int = 20               # bars for RS-line slope
    benchmark: str = "NIFTY"

    # ── Extension (ATR units, replaces stdev bands) ──
    atr_period: int = 14
    ext_atr_chase: float = 4.0          # > this many ATRs from EMA = "chasing"/exhausted
    ext_atr_ideal: float = 1.5          # entries within ~this distance score best

    # ── Base / VCP contraction (TIME) ──
    base_window: int = 12               # bars to scan for a contraction base
    base_min_bars: int = 4              # a base must be at least this many bars
    contraction_ratio: float = 0.6      # recent range must be < ratio × earlier range
    pivot_lookback: int = 5             # swing high/low lookback for breakout & stops

    # ── Volume ──
    vol_median_period: int = 20
    vol_breakout_mult: float = 1.4      # breakout vol vs median to confirm
    vol_dryup_mult: float = 0.85        # base vol below this × median = healthy dry-up

    # ── Freshness (TIME) ──
    freshness_max_bars: int = 6         # a phase older than this is stale (score→0)

    # ── Risk / money model ──
    sl_atr_mult: float = 1.8            # ATR stop distance
    sl_buffer_pct: float = 0.003        # buffer below structural pivot
    rr_target: float = 2.5              # primary R-multiple target
    risk_per_trade_pct: float = 1.0     # fixed-fractional risk (% of capital)
    max_position_pct: float = 15.0      # notional cap (% of capital)
    max_hold_bars: int = 40             # time-stop: exit if not working within N bars

    # ── Scoring ──
    weights: FactorWeights = field(default_factory=FactorWeights)
    grade_a: float = 0.75
    grade_b: float = 0.60
    grade_c: float = 0.45               # below grade_c → not actionable

    def warmup_bars(self) -> int:
        """Minimum bars before any factor is trustworthy."""
        return max(
            self.ema_slow,
            self.rsi_wma_period,
            self.vol_median_period,
            self.base_window + self.pivot_lookback,
        ) + 5


# ══════════════════════════════════════════════════════════════════════
# Tier presets
# ══════════════════════════════════════════════════════════════════════

TIER_PRESETS: Dict[SwingTier, SwingV2Config] = {
    SwingTier.SMALL: SwingV2Config(
        tier=SwingTier.SMALL,
        interval="FIFTEEN_MINUTE",
        htf_interval="ONE_HOUR",
        lookback_days=30,
        roc_period=10,
        rs_lookback=26,            # ~one session of 15m bars
        sl_atr_mult=1.5,
        rr_target=2.0,
        max_hold_bars=50,          # ~2 sessions of 15m bars
        freshness_max_bars=8,
    ),
    SwingTier.MEDIUM: SwingV2Config(
        tier=SwingTier.MEDIUM,
        interval="ONE_HOUR",
        htf_interval="ONE_DAY",
        lookback_days=120,
        roc_period=10,
        rs_lookback=20,
        sl_atr_mult=1.8,
        rr_target=2.5,
        max_hold_bars=40,          # ~1.5 weeks of ~6.25 1h bars/day
        freshness_max_bars=6,
    ),
    SwingTier.LONG: SwingV2Config(
        tier=SwingTier.LONG,
        interval="ONE_DAY",
        htf_interval="ONE_WEEK",   # derived by weekly aggregation
        htf_from_weekly_agg=True,
        lookback_days=500,
        roc_period=20,
        rs_lookback=20,
        sl_atr_mult=2.0,
        rr_target=3.0,
        max_hold_bars=30,          # ~6 trading weeks
        freshness_max_bars=8,
    ),
}


def get_tier_config(tier: SwingTier | str) -> SwingV2Config:
    """Resolve a tier (enum or string) to its preset config."""
    if isinstance(tier, str):
        tier = SwingTier(tier.lower())
    return TIER_PRESETS[tier]
