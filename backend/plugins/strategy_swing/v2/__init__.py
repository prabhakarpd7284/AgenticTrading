"""
Oliver Kell Cycle — v2 (robust, timeframe-tiered swing engine).

Fully isolated from the v1 `plugins.strategy_swing` modules. v1 keeps running
the live scanner untouched; v2 is the hardened re-derivation that adds:

  • A regime gate (primary + HTF EMA stack & slope) on every entry.
  • Momentum (RSI regime + EMA3×WMA21 cross + ROC) on the swing timeframe.
  • Relative strength vs NIFTY (Mansfield RS slope).
  • ATR-normalised extension (replaces the scale-dependent stdev bands).
  • Base / VCP contraction quality + base length (a TIME factor).
  • Phase freshness decay (fixes the sticky-phase bug + adds a TIME factor).
  • A full money model: structural stop, R-multiple target, fixed-fractional
    position size, and a time-stop.

Three holding tiers, selected by candle timeframe:
  SMALL  — 15m candles (HTF 1h)        → days
  MEDIUM — 1h candles  (HTF daily)     → ~1–2 weeks
  LONG   — daily candles (HTF weekly)  → weeks–months

Every trade rolls up into the monthly ledger via apps.strategies.Signal
(source=OK_SCANNER, tagged v2 in the strategy/indicators fields).
"""

from plugins.strategy_swing.v2.config import (  # noqa: F401
    SwingV2Config,
    SwingTier,
    TIER_PRESETS,
    get_tier_config,
)
from plugins.strategy_swing.v2.engine import (  # noqa: F401
    SwingEngineV2,
    SwingSignalV2,
    Phase,
    Regime,
    Action,
)
