"""BacktestEngine — the core walk-forward and replay engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

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

    # ── TIMELINE mode only ──────────────────────────────────────────────
    # Signals are detected on a bar's CLOSE and filled at the NEXT bar's open
    # of the same symbol (no same-bar look-ahead). Signal levels are either
    # "relative" (stop/target distances re-anchored to the fill) or "absolute"
    # (structural prices; a fill that opens beyond either level is rejected) —
    # set per signal in ``EntrySignal.metadata["levels"]`` (default relative).
    session_flat: bool = False       # intraday: flat by the last bar of each session
    eod_minute: Optional[int] = None  # IST minutes; exit at the close of the bar ENDING at/after it
    bar_minutes: int = 0             # bar length (for the eod check); 0 = daily
    last_entry_minute: Optional[int] = None  # no fills on a bar starting after this (IST minutes)
    max_trades_per_day: int = 0      # 0 = unlimited
    exit_slippage_pct: Optional[float] = None  # None → same as slippage_pct
    # Round-trip costs for a closed trade (₹). Deducted from trade.pnl; the
    # gross figure is kept in trade.metadata["gross_pnl"].
    cost_fn: Optional[Callable[[Any], float]] = None
    # Optional owner-rule gate (duck-typed): ``status(day) -> (state, mult)``
    # before entries, ``close_day(day, net_pnl)`` after a day with trades.
    day_gate: Optional[Any] = None


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
        self.rejections: Dict[str, int] = {}       # TIMELINE: skipped signals by reason

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
        elif self.config.mode == TimeframeMode.TIMELINE:
            self._timeline(data, context)
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
    # Mode 3: Timeline (all symbols chronologically)
    # ──────────────────────────────────────────

    def _timeline(self, data: Dict[str, List[Bar]], context: dict):
        """Every bar of every symbol in time order. Per bar:
        1. fill a pending entry for this symbol at the bar's OPEN,
        2. run exits for this symbol's open trades (incl. the one just filled),
        3. detect new signals on the bar's CLOSE → pending for the next bar.

        Portfolio limits (max_positions, daily loss, trades/day, owner gate)
        apply across symbols, which bar-by-bar ``_replay`` cannot do.
        """
        cfg = self.config
        events = sorted(
            ((b.timestamp, sym, i) for sym, bars in data.items() for i, b in enumerate(bars)),
            key=lambda e: (e[0], e[1]),
        )
        pending: Dict[str, tuple] = {}
        last_exit: Dict[str, int] = {}
        loss_limit = cfg.capital * cfg.daily_loss_limit_pct / 100
        self.rejections: Dict[str, int] = {}
        day = None
        day_net = 0.0
        day_trades = 0
        day_closed = 0

        def reject(reason: str):
            self._skipped += 1
            self.rejections[reason] = self.rejections.get(reason, 0) + 1

        for ts, sym, i in events:
            d = ts[:10]
            if d != day:
                if day is not None and day_closed and cfg.day_gate is not None:
                    cfg.day_gate.close_day(day, day_net)
                day, day_net, day_trades, day_closed = d, 0.0, 0, 0
            bars = data[sym]
            bar = bars[i]
            last_of_session = i + 1 >= len(bars) or bars[i + 1].timestamp[:10] != d

            # 1. fill pending
            p = pending.pop(sym, None)
            if p is not None:
                sig, mult = p
                if cfg.session_flat and sig.timestamp[:10] != d:
                    reject("no_next_bar_in_session")
                else:
                    trade = self._fill(sig, bar, mult, reject)
                    if trade is not None:
                        self._open.append(trade)
                        day_trades += 1

            # 2. exits
            for trade in [t for t in self._open if t.symbol == sym]:
                trade.on_bar(bar)
                ex = self.exit_manager.process(trade, bar)
                if ex is None and cfg.session_flat and (last_of_session or self._past_eod(bar)):
                    from plugins.strategy_backtest.types import ExitSignal
                    ex = ExitSignal(price=bar.close, reason="EOD close")
                if ex is not None and ex.partial:
                    trade.on_partial(ex.price, ex.partial_fraction, ex.reason)
                elif ex is not None:
                    self._close(trade, ex.price, bar.timestamp, ex.reason)
                    last_exit[sym] = i
                    day_net += trade.pnl
                    day_closed += 1

            # 3. entries (decided on this bar's close, filled next bar)
            if not self._in_entry_window(d) or i + 1 >= len(bars):
                continue
            if cfg.session_flat and last_of_session:
                continue
            nxt = bars[i + 1]
            if cfg.last_entry_minute is not None and _minute(nxt.timestamp) > cfg.last_entry_minute:
                continue
            if cfg.session_flat and self._past_eod(nxt):
                continue
            if sym in pending or self._count_open(sym) >= cfg.max_concurrent_per_symbol:
                continue
            if i - last_exit.get(sym, -10**9) < cfg.cooldown_bars:
                continue
            signals = self.entry.detect(sym, bars, i, context)
            if not signals:
                continue
            sig = signals[0]
            self._total_signals += 1
            if len(self._open) + len(pending) >= cfg.max_positions:
                reject("max_positions"); continue
            if day_net < -loss_limit:
                reject("daily_loss_limit"); continue
            if cfg.max_trades_per_day and day_trades + sum(
                    1 for s, _ in pending.values() if s.timestamp[:10] == d) >= cfg.max_trades_per_day:
                reject("max_trades_per_day"); continue
            mult = 1.0
            if cfg.day_gate is not None:
                _state, mult = cfg.day_gate.status(d)
                if mult <= 0:
                    reject("owner_cooldown"); continue
            pending[sym] = (sig, mult)

        if day is not None and day_closed and cfg.day_gate is not None:
            cfg.day_gate.close_day(day, day_net)

    def _past_eod(self, bar: Bar) -> bool:
        cfg = self.config
        if cfg.eod_minute is None:
            return False
        return _minute(bar.timestamp) + cfg.bar_minutes >= cfg.eod_minute + 1 \
            if cfg.bar_minutes else _minute(bar.timestamp) >= cfg.eod_minute

    def _fill(self, sig: EntrySignal, bar: Bar, mult: float, reject) -> Optional[Trade]:
        """Open a trade at ``bar.open`` from a signal decided on the previous bar."""
        cfg = self.config
        fill = bar.open
        absolute = sig.metadata.get("levels") == "absolute"
        long = sig.side == TradeSide.BUY
        if absolute:
            stop, target = sig.stoploss, sig.target
            if (long and (fill <= stop or (target and fill >= target))) or \
                    (not long and (fill >= stop or (target and fill <= target))):
                reject("opened_beyond_levels")
                return None
        else:
            shift = fill - sig.entry_price
            stop = sig.stoploss + shift
            target = sig.target + shift if sig.target else 0.0
        slip = cfg.slippage_pct / 100
        exp_fill = fill * (1 + slip) if long else fill * (1 - slip)
        risk = (exp_fill - stop) if long else (stop - exp_fill)
        if risk <= 0:
            reject("non_positive_risk")
            return None
        size = self.sizer.compute(exp_fill, risk)
        qty = int(size.quantity * mult)
        if qty <= 0:
            reject("size_zero")
            return None
        meta = dict(sig.metadata)
        meta.update(signal_time=sig.timestamp, signal_price=sig.entry_price,
                    initial_stop=round(stop, 4), risk_mult=mult)
        filled = EntrySignal(symbol=sig.symbol, timestamp=bar.timestamp, side=sig.side,
                             entry_price=fill, stoploss=round(stop, 2),
                             target=round(target, 2) if target else 0.0,
                             risk_points=risk, atr=sig.atr, metadata=meta)
        trade = Trade()
        trade.on_entry(filled, qty, cfg.slippage_pct, cfg.pnl_mode)
        trade.risk_points = abs(trade.entry_price - trade.stoploss)
        if not trade.target:
            # No target: park it where TargetExit can never trigger.
            trade.target = float("inf") if long else float("-inf")
        trade.metadata["initial_risk"] = round(trade.risk_points, 4)
        return trade

    def _close(self, trade: Trade, price: float, ts: str, reason: str):
        cfg = self.config
        pct = cfg.slippage_pct if cfg.exit_slippage_pct is None else cfg.exit_slippage_pct
        slip = pct / 100
        # exiting a long sells (price down); exiting a short buys (price up)
        fill = price * (1 - slip) if trade.side == TradeSide.BUY else price * (1 + slip)
        trade.on_exit(fill, ts, reason)
        gross = trade.pnl
        costs = float(cfg.cost_fn(trade)) if cfg.cost_fn else 0.0
        trade.metadata["gross_pnl"] = gross
        trade.metadata["costs"] = round(costs, 2)
        trade.pnl = round(gross - costs, 2)
        risk = trade.metadata.get("initial_risk") or trade.risk_points
        if risk and trade.quantity:
            trade.rr_achieved = round(trade.pnl / (risk * trade.quantity), 4)
        if trade in self._open:
            self._open.remove(trade)
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
        if self.config.mode == TimeframeMode.TIMELINE:
            for trade in list(self._open):
                bars = data.get(trade.symbol, [])
                last = bars[-1] if bars else None
                self._close(trade, last.close if last else trade.entry_price,
                            last.timestamp if last else trade.entry_time, "End of test")
            return
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


def _minute(ts: str) -> int:
    """IST minute-of-day from an ISO timestamp ('YYYY-MM-DDTHH:MM…'); 0 for dates."""
    try:
        return int(ts[11:13]) * 60 + int(ts[14:16])
    except (ValueError, IndexError):
        return 0
