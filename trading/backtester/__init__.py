"""
Unified Backtester Engine — event-driven, live-ready, composable.

Usage:
    from trading.backtester import BacktestEngine, EngineConfig
    from trading.backtester.exits import BreakevenExit, StoplossExit, TargetExit
    from trading.backtester.entry import OKCycleAdapter

    engine = BacktestEngine(
        config=EngineConfig(capital=500_000),
        entry=OKCycleAdapter(...),
        exits=[BreakevenExit(0.5), StoplossExit(), TargetExit()],
    )
    stats = engine.run(data)
"""
from trading.backtester.engine import BacktestEngine, EngineConfig
from trading.backtester.trade import Trade
from trading.backtester.types import PnLMode, TimeframeMode, TradeSide, Bar, EntrySignal
from trading.backtester.stats import BacktestStats

__all__ = [
    "BacktestEngine", "EngineConfig", "Trade",
    "PnLMode", "TimeframeMode", "TradeSide", "Bar", "EntrySignal",
    "BacktestStats",
]
