"""BacktestEngine — the core walk-forward and replay engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from logzero import logger

from plugins.strategy_backtest.entry import EntryDetector
from plugins.strategy_backtest.exits import ExitCheck, ExitManager
from plugins.strategy_backtest.sizing import PositionSizer
from plugins.strategy_backtest.stats import BacktestStats, StatsAggregator
from plugins.strategy_backtest.trade import Trade
from plugins.strategy_backtest.types import (
    Bar,
    EntrySignal,
    PnLMode,
    TimeframeMode,
    TradeState,
    TradeSide,
)


@dataclass
class EngineConfig:
    """All engine parameters as a single config object.

    New strategy = new EngineConfig + new EntryDetector. Zero engine changes.
    """
    pnl_mode: PnLMode = PnLMode.RUPEES
    mode: TimeframeMode = TimeframeMode.DAILY
    capital: float = 500_000
    max_risk_pct: float = 1.0
    max_position_pct: float = 15.0
    max_positions: int = 11
    slippage_pct: float = 0.1
    daily_loss_limit_pct: float = 3.0
    cooldown_bars: int = 0          # Min bars between entries (per symbol)
    max_concurrent_per_symbol: int = 1  # Max open trades per symbol
    # Entry window (inclusive, 'YYYY-MM-DD'). Bars OUTSIDE this window are still
    # processed — for indicator warmup and for exiting open trades — but NO new
    # entries are opened outside it. Lets a backtest fetch a long lookback for
    # warmup yet only trade the requested period. None = no restriction.
    entry_from: Optional[str] = None
    entry_to: Optional[str] = None


class BacktestEngine:
    """Strategy-agnostic backtester with composable entry/exit management.

    Modes:
      DAILY: Walk-forward on daily bars. For each trading day:
             1. Check exits on open trades
             2. Scan for new entries across all symbols
      INTRADAY: Bar-by-bar replay from pre-computed signals.
                Walk forward from each signal timestamp.

    Usage:
        engine = BacktestEngine(
            config=EngineConfig(capital=500_000),
            entry=OKCycleAdapter(),
            exits=[BreakevenExit(0.5), StoplossExit(), TargetExit()],
        )
        stats = engine.run({"RELIANCE": bars, "TCS": bars})
    """

    def __init__(
        self,
        config: EngineConfig,
        entry: EntryDetector,
        exits: List[ExitCheck],
    ):
        self.config = config
        self.entry = entry
        self.exit_manager = ExitManager(exits)
        self.sizer = PositionSizer(
            config.capital, config.max_risk_pct,
            config.max_position_pct, config.pnl_mode,
        )

        self._open: List[Trade] = []
        self._closed: List[Trade] = []
        self._total_signals = 0
        self._skipped = 0
        self._daily_pnl = 0.0
        self._last_exit_bar: Dict[str, int] = {}  # symbol → last exit bar index

    def run(
        self,
        data: Dict[str, List[Bar]],
        context: Optional[dict] = None,
    ) -> BacktestStats:
        """Main entry point. Routes to appropriate mode.

        Args:
            data: {symbol: [Bar, ...]} — pre-fetched candle data
            context: Strategy-specific context (weekly bars, etc.)
        """
        context = context or {}

        if self.config.mode == TimeframeMode.DAILY:
            self._walk_forward(data, context)
        else:
            self._replay(data, context)

        # Close any remaining open trades at last price
        self._close_remaining(data)

        # Compute stats
        agg = StatsAggregator(self.config.capital, self.config.pnl_mode)
        return agg.compute(self._closed, self._total_signals, self._skipped)

    # ──────────────────────────────────────────
    # Mode 1: Walk-forward (daily swing)
    # ──────────────────────────────────────────

    def _walk_forward(self, data: Dict[str, List[Bar]], context: dict):
        """For each trading day: check exits, then scan entries."""
        # Build sorted list of all trading days
        all_dates = set()
        for bars in data.values():
            for b in bars:
                all_dates.add(b.timestamp[:10])
        trading_days = sorted(all_dates)

        for day in trading_days:
            self._daily_pnl = 0.0

            # Whether new entries are allowed on this day. Bars before the
            # window still run (warmup + exits) but don't open trades, so a
            # 120-day lookback doesn't leak warmup-period trades into the
            # requested window's stats.
            entries_allowed = self._in_entry_window(day)

            # 1. Check exits on open trades
            for trade in list(self._open):
                bar = self._find_bar_for_day(data.get(trade.symbol, []), day)
                if bar is None:
                    continue

                trade.on_bar(bar)
                exit_sig = self.exit_manager.process(trade, bar)

                if exit_sig and exit_sig.partial:
                    trade.on_partial(exit_sig.price, exit_sig.partial_fraction, exit_sig.reason)
                elif exit_sig:
                    trade.on_exit(exit_sig.price, bar.timestamp, exit_sig.reason)
                    self._open.remove(trade)
                    self._closed.append(trade)
                    self.sizer.update_capital(trade.pnl)
                    self._daily_pnl += trade.pnl

            # 2. Check daily loss limit
            loss_limit = self.config.capital * self.config.daily_loss_limit_pct / 100
            if self._daily_pnl < -loss_limit:
                continue

            # 3. Scan for new entries (only inside the entry window)
            if not entries_allowed:
                continue
            for symbol, bars in data.items():
                if self._count_open(symbol) >= self.config.max_concurrent_per_symbol:
                    continue
                if len(self._open) >= self.config.max_positions:
                    break

                idx = self._find_bar_index_for_day(bars, day)
                if idx < 0:
                    continue

                signals = self.entry.detect(symbol, bars, idx, context)
                for sig in signals:
                    self._try_open(sig)

    # ──────────────────────────────────────────
    # Mode 2: Replay (intraday)
    # ──────────────────────────────────────────

    def _replay(self, data: Dict[str, List[Bar]], context: dict):
        """Bar-by-bar: detect signals, walk forward from each."""
        for symbol, bars in data.items():
            # Pre-compute signals (intraday adapter caches them)
            if hasattr(self.entry, "precompute"):
                candle_dicts = [{"timestamp": b.timestamp, "open": b.open, "high": b.high,
                                 "low": b.low, "close": b.close, "volume": b.volume}
                                for b in bars]
                self.entry.precompute(symbol, candle_dicts)  # type: ignore[attr-defined]

            last_exit_idx = -self.config.cooldown_bars

            for i in range(len(bars)):
                # Only open entries inside the window; bars outside still feed
                # the detector (warmup) and exit walk-forward below.
                if not self._in_entry_window(bars[i].timestamp[:10]):
                    continue
                signals = self.entry.detect(symbol, bars, i, context)
                if not signals:
                    continue

                for sig in signals:
                    # Cooldown check
                    if i - last_exit_idx < self.config.cooldown_bars:
                        continue

                    self._total_signals += 1
                    size = self.sizer.compute(sig.entry_price, sig.risk_points)
                    if size.quantity <= 0:
                        self._skipped += 1
                        continue

                    trade = Trade()
                    trade.on_entry(sig, size.quantity, self.config.slippage_pct, self.config.pnl_mode)

                    # Walk forward from signal bar
                    exited = False
                    for j in range(i + 1, len(bars)):
                        trade.on_bar(bars[j])
                        exit_sig = self.exit_manager.process(trade, bars[j])

                        if exit_sig and exit_sig.partial:
                            trade.on_partial(exit_sig.price, exit_sig.partial_fraction, exit_sig.reason)
                        elif exit_sig:
                            trade.on_exit(exit_sig.price, bars[j].timestamp, exit_sig.reason)
                            last_exit_idx = j
                            exited = True
                            break

                    if not exited:
                        trade.on_exit(bars[-1].close, bars[-1].timestamp, "End of test")
                        last_exit_idx = len(bars) - 1

                    self._closed.append(trade)
                    self.sizer.update_capital(trade.pnl)

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _try_open(self, signal: EntrySignal):
        """Attempt to open a trade from a signal."""
        self._total_signals += 1

        size = self.sizer.compute(signal.entry_price, signal.risk_points)
        if size.quantity <= 0:
            self._skipped += 1
            return

        trade = Trade()
        trade.on_entry(signal, size.quantity, self.config.slippage_pct, self.config.pnl_mode)
        self._open.append(trade)

    def _count_open(self, symbol: str) -> int:
        return sum(1 for t in self._open if t.symbol == symbol)

    def _in_entry_window(self, day: str) -> bool:
        """True if `day` ('YYYY-MM-DD') is within the configured entry window.

        No window configured → always True (back-compat). Comparison is on the
        date string, which is lexicographically ordered for ISO dates."""
        if self.config.entry_from and day < self.config.entry_from:
            return False
        if self.config.entry_to and day > self.config.entry_to:
            return False
        return True

    def _close_remaining(self, data: Dict[str, List[Bar]]):
        """Close all open trades at last available price."""
        for trade in list(self._open):
            bars = data.get(trade.symbol, [])
            if bars:
                trade.on_exit(bars[-1].close, bars[-1].timestamp, "End of test")
            else:
                trade.on_exit(trade.entry_price, trade.entry_time, "No data")
            self._open.remove(trade)
            self._closed.append(trade)

    @staticmethod
    def _find_bar_for_day(bars: List[Bar], day: str) -> Optional[Bar]:
        for b in bars:
            if b.timestamp[:10] == day:
                return b
        return None

    @staticmethod
    def _find_bar_index_for_day(bars: List[Bar], day: str) -> int:
        for i, b in enumerate(bars):
            if b.timestamp[:10] == day:
                return i
        return -1
