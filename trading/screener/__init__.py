"""
Live Screener — real-time opportunity detection across NIFTY 50.

Architecture:
  TickStream (websocket/polling) → CandleStore (in-memory) → IndicatorEngine
    → ConditionEngine (strategies) → Signals → Telegram/Dashboard/CLI
"""
