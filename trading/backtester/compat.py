"""Compatibility layer — drop-in replacements for old backtest functions.

Existing management commands update their imports to point here.
The old modules are kept with deprecation warnings.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from logzero import logger

from trading.backtester.engine import BacktestEngine, EngineConfig
from trading.backtester.entry import IntradayCycleAdapter, OKCycleAdapter, ScreenerEntryAdapter
from trading.backtester.exits import (
    BreakevenExit,
    EODExit,
    MaxHoldExit,
    StoplossExit,
    TargetExit,
    TrailingSLExit,
)
from trading.backtester.report import ReportFormatter
from trading.backtester.stats import BacktestStats
from trading.backtester.types import Bar, PnLMode, TimeframeMode


def run_ok_backtest(
    symbols: List[str],
    from_date: str,
    to_date: str,
    capital: float = None,
    max_risk_pct: float = None,
    max_position_pct: float = None,
    max_positions: int = None,
    min_rr: float = 2.0,
    max_hold_bars: int = 10,
    slippage_pct: float = 0.1,
    data_svc=None,
) -> BacktestStats:
    """Drop-in replacement for trading.swing.ok_backtest.run_ok_backtest().

    Returns BacktestStats (new type) instead of the old BacktestResult.
    The old BacktestResult attributes map 1:1 to BacktestStats fields.
    """
    from trading.config import config as tc

    capital = capital or tc.risk.default_capital
    max_risk_pct = max_risk_pct or tc.risk.max_risk_per_trade_pct
    max_position_pct = max_position_pct or tc.risk.max_position_size_pct
    max_positions = max_positions or tc.risk.max_open_positions

    # Fetch data
    warmup_days = tc.ok_cycle.lookback_days
    fetch_from = (
        datetime.strptime(from_date, "%Y-%m-%d").date() - timedelta(days=warmup_days)
    ).strftime("%Y-%m-%d")

    if data_svc is None:
        from trading.services.data_service import DataService
        data_svc = DataService()

    logger.info(f"OK Backtest (v2): {from_date} → {to_date} | {len(symbols)} symbols")
    data: Dict[str, List[Bar]] = {}
    for symbol in symbols:
        try:
            candles = data_svc.fetch_historical(symbol, fetch_from, to_date, interval="ONE_DAY")
            if candles and len(candles) >= tc.ok_cycle.ema_slow + 10:
                data[symbol] = [Bar.from_dict(c) for c in candles]
        except Exception as e:
            logger.warning(f"Skip {symbol}: {e}")

    # Build engine
    engine = BacktestEngine(
        config=EngineConfig(
            pnl_mode=PnLMode.RUPEES,
            mode=TimeframeMode.DAILY,
            capital=capital,
            max_risk_pct=max_risk_pct,
            max_position_pct=max_position_pct,
            max_positions=max_positions,
            slippage_pct=slippage_pct,
            daily_loss_limit_pct=tc.risk.max_daily_loss_pct,
        ),
        entry=OKCycleAdapter(min_rr=min_rr, slippage_pct=slippage_pct),
        exits=[
            BreakevenExit(trigger_r=tc.trade_manager.breakeven_at_r, buffer=0.5),
            TrailingSLExit(
                trigger_r=tc.trade_manager.partial_at_r,
                atr_factor=tc.trade_manager.trail_atr_factor,
                tight_r=tc.trade_manager.tight_trail_at_r,
                tight_factor=tc.trade_manager.tight_trail_atr_factor,
            ),
            MaxHoldExit(max_hold_bars),
            StoplossExit(),
            TargetExit(),
        ],
    )

    # Filter data to backtest range only for bar matching
    stats = engine.run(data)
    logger.info(f"OK Backtest (v2): {stats.total_trades} trades, PF {stats.profit_factor:.2f}")
    return stats


def run_intraday_backtest(
    symbols: List[str],
    from_date: str,
    to_date: str,
    timeframes: List[str] = None,
    sl_atr_mults: List[float] = None,
    rr_ratios: List[float] = None,
    capital: float = None,
    max_risk_pct: float = None,
    max_position_pct: float = None,
    cooldown_bars: int = 5,
    data_svc=None,
) -> List[dict]:
    """Drop-in replacement for trading.swing.ok_intraday_backtest.run_intraday_backtest().

    Returns a list of result dicts (one per TF×SL×RR combo) instead of MultiTFReport.
    Each dict: {tf, sl_atr, rr, stats: BacktestStats}
    """
    from trading.config import config as tc

    timeframes = timeframes or ["3m", "5m", "15m"]
    sl_atr_mults = sl_atr_mults or [1.0, 1.5, 2.0]
    rr_ratios = rr_ratios or [1.5, 2.0, 2.5]
    capital = capital or tc.risk.default_capital
    max_risk_pct = max_risk_pct or tc.risk.max_risk_per_trade_pct
    max_position_pct = max_position_pct or tc.risk.max_position_size_pct

    INTERVAL_MAP = {"3m": "THREE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE"}

    if data_svc is None:
        from trading.services.data_service import DataService
        data_svc = DataService()

    # Fetch data per TF
    tf_data: Dict[str, Dict[str, List[dict]]] = {}
    for tf in timeframes:
        interval = INTERVAL_MAP.get(tf)
        if not interval:
            continue
        tf_data[tf] = {}
        for symbol in symbols:
            try:
                candles = data_svc.fetch_intraday_candles(symbol, from_date, to_date, interval=interval)
                if candles and len(candles) >= 60:
                    tf_data[tf][symbol] = candles
            except Exception as e:
                logger.warning(f"Skip {symbol} {tf}: {e}")

    # Grid search: one engine per (TF, SL, RR)
    results = []
    for tf in timeframes:
        if tf not in tf_data:
            continue
        for sl_atr in sl_atr_mults:
            for rr in rr_ratios:
                adapter = IntradayCycleAdapter(sl_atr_mult=sl_atr, min_rr=rr)

                # Pre-compute signals for all symbols
                bars_data: Dict[str, List[Bar]] = {}
                for symbol, candles in tf_data[tf].items():
                    adapter.precompute(symbol, candles)
                    bars_data[symbol] = [Bar.from_dict(c) for c in candles]

                engine = BacktestEngine(
                    config=EngineConfig(
                        pnl_mode=PnLMode.RUPEES,
                        mode=TimeframeMode.INTRADAY,
                        capital=capital,
                        max_risk_pct=max_risk_pct,
                        max_position_pct=max_position_pct,
                        slippage_pct=0.05,
                        cooldown_bars=cooldown_bars,
                    ),
                    entry=adapter,
                    exits=[
                        BreakevenExit(trigger_r=1.0, buffer=0.5),
                        StoplossExit(),
                        TargetExit(),
                        EODExit(15, 20),
                    ],
                )

                stats = engine.run(bars_data)
                results.append({"tf": tf, "sl_atr": sl_atr, "rr": rr, "stats": stats})

                logger.info(
                    f"  {tf} SL:{sl_atr} RR:{rr} → "
                    f"{stats.total_trades}t, {stats.win_rate:.0%}W, "
                    f"PF:{stats.profit_factor:.2f}"
                )

    return results


def run_screener_backtest(
    symbols: List[str],
    from_date: str,
    to_date: str,
    strategies=None,
    slippage_pct: float = 0.05,
) -> BacktestStats:
    """Drop-in replacement for trading.screener.backtest.run_backtest().

    Uses the old screener engine for signal generation (tick replay
    through ScreenerEngine), but the new BacktestEngine for trade
    simulation (exits, P&L, stats).
    """
    from plugins.strategy_screener.backtest import run_backtest as _old_replay
    from plugins.strategy_screener.backtest import BacktestResult as OldResult

    # Step 1: Use old screener replay to collect signals
    # (the screener's tick-by-tick replay + strategy evaluation is
    # tightly coupled and not worth reimplementing)
    old_result = _old_replay(symbols, from_date, to_date, strategies, slippage_pct)

    # Step 2: Convert old BacktestTrade results to unified BacktestStats
    # The old result already simulated trades with its own exit logic,
    # so we import the completed trades directly into stats.
    from trading.backtester.trade import Trade
    from trading.backtester.types import TradeSide, PnLMode

    trades: List[Trade] = []
    for old_t in old_result.trades:
        t = Trade()
        t.symbol = old_t.signal.symbol
        t.side = TradeSide.BUY if old_t.signal.side == "BUY" else TradeSide.SHORT
        t.entry_price = old_t.signal.entry
        t.entry_time = str(old_t.signal.timestamp)
        t.exit_price = old_t.exit_price
        t.exit_time = str(old_t.exit_time or "")
        t.exit_reason = old_t.exit_reason
        t.stoploss = old_t.signal.stoploss
        t.target = old_t.signal.target
        t.risk_points = old_t.signal.risk_points
        t.quantity = 1
        t.remaining_qty = 1
        t.bars_held = 0
        t.max_favorable = old_t.max_favorable
        t.max_adverse = old_t.max_adverse
        t.pnl = old_t.pnl
        t.pnl_pct = old_t.pnl_pct
        t.rr_achieved = old_t.pnl / old_t.signal.risk_points if old_t.signal.risk_points > 0 else 0
        t.metadata = {"trade_type": old_t.signal.strategy, "strategy": old_t.signal.strategy}
        t._pnl_mode = PnLMode.POINTS
        from trading.backtester.types import TradeState
        t.state = TradeState.CLOSED
        trades.append(t)

    from trading.backtester.stats import StatsAggregator
    agg = StatsAggregator(0, PnLMode.POINTS)
    return agg.compute(trades, total_signals=old_result.total_signals)


def run_basket_backtest(
    symbols: List[str],
    from_date: str,
    to_date: str,
    capital: float = None,
    interval: str = "FIVE_MINUTE",
    min_rr: float = 2.0,
    data_svc=None,
) -> BacktestStats:
    """Backtest the morning basket strategy on historical intraday data.

    For each symbol:
      1. Fetch intraday candles (5m by default)
      2. Pre-compute basket signals in the 9:45-10:30 window
      3. Run through BacktestEngine with breakeven + SL + target + EOD exits
    """
    from trading.config import config as tc
    from trading.basket.entry_adapter import BasketEntryAdapter

    capital = capital or tc.risk.default_capital

    if data_svc is None:
        from trading.services.data_service import DataService
        data_svc = DataService()

    logger.info(f"Basket backtest: {from_date} → {to_date} | {len(symbols)} symbols | {interval}")

    # Fetch intraday data
    all_data: Dict[str, List[Bar]] = {}
    adapter = BasketEntryAdapter(min_rr=min_rr)

    for i, symbol in enumerate(symbols):
        if (i + 1) % 10 == 0:
            logger.info(f"  Fetching: {i + 1}/{len(symbols)}")
        try:
            candles = data_svc.fetch_intraday_candles(symbol, from_date, to_date, interval=interval)
            if candles and len(candles) >= 30:
                adapter.precompute(symbol, candles)
                all_data[symbol] = [Bar.from_dict(c) for c in candles]
        except Exception as e:
            logger.warning(f"  Skip {symbol}: {e}")

    logger.info(f"  Data: {len(all_data)}/{len(symbols)} symbols")

    # Build engine
    engine = BacktestEngine(
        config=EngineConfig(
            pnl_mode=PnLMode.RUPEES,
            mode=TimeframeMode.INTRADAY,
            capital=capital,
            max_risk_pct=tc.basket.max_risk_per_leg_pct,
            max_position_pct=15.0,
            slippage_pct=0.05,
            cooldown_bars=5,
        ),
        entry=adapter,
        exits=[
            BreakevenExit(trigger_r=tc.basket.breakeven_at_r, buffer=0.5),
            StoplossExit(),
            TargetExit(),
            EODExit(15, 15),
        ],
    )

    stats = engine.run(all_data)
    logger.info(
        f"Basket backtest: {stats.total_trades}t, {stats.win_rate:.0%}W, "
        f"PF:{stats.profit_factor:.2f}, ₹{stats.total_pnl:+,.0f}"
    )
    return stats
