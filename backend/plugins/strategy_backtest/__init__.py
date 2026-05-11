"""Backtest plugin — historical replay engine.

Unlike the other strategy plugins, this is infrastructure: a generic
event-driven backtester that drives entry/exit adapters over historical
candles. Strategies (screener, OK swing, intraday cycle, basket) wire
into it via the entry/exit adapter contracts in entry.py + exits.py.

Exposed via the alphadesk.strategies entry-point group so the registry
knows it exists, but the build_graph wraps trading.backtester.compat
helpers rather than running a "strategy" per se.
"""
from plugins.strategy_backtest.plugin import BacktestStrategy

# Re-export the public surface so callers can do
# `from plugins.strategy_backtest import BacktestEngine, Trade, ...`.
from plugins.strategy_backtest.engine import BacktestEngine, EngineConfig
from plugins.strategy_backtest.trade import Trade
from plugins.strategy_backtest.types import (
    Bar, EntrySignal, PnLMode, TimeframeMode, TradeSide,
)
from plugins.strategy_backtest.stats import BacktestStats

__all__ = [
    "BacktestStrategy",
    "BacktestEngine",
    "EngineConfig",
    "Trade",
    "Bar",
    "EntrySignal",
    "PnLMode",
    "TimeframeMode",
    "TradeSide",
    "BacktestStats",
]
