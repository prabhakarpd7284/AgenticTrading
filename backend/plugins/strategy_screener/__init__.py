"""Screener plugin — intraday opportunity detection across NIFTY 50.

Architecture:
  TickStream (websocket/polling) → CandleStore (in-memory) → IndicatorEngine
    → ConditionEngine (strategies) → Signals → Telegram/Dashboard/CLI
"""
from plugins.strategy_screener.plugin import ScreenerStrategy

__all__ = ["ScreenerStrategy"]
