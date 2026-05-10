"""
Backtest Engine — replay historical candles through the screener.

Feeds historical 1m candles into ScreenerEngine tick-by-tick,
captures all signals, then computes P&L and performance stats.

Usage:
    results = run_backtest(["RELIANCE", "TCS"], "2026-03-10", "2026-03-20")
    print(results["summary"])
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from logzero import logger

from trading.screener.engine import ScreenerEngine
from trading.screener.signals import Signal
from trading.screener.strategies import Strategy, STRATEGIES


@dataclass
class BacktestTrade:
    """A simulated trade from backtest."""
    signal: Signal
    exit_price: float = 0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl: float = 0
    pnl_pct: float = 0
    hit_target: bool = False
    hit_sl: bool = False
    max_favorable: float = 0   # max favorable excursion in points
    max_adverse: float = 0     # max adverse excursion in points


@dataclass
class BacktestResult:
    """Aggregate backtest performance."""
    trades: list[BacktestTrade] = field(default_factory=list)
    total_signals: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0
    win_rate: float = 0
    avg_rr_achieved: float = 0
    profit_factor: float = 0
    max_drawdown: float = 0
    per_strategy: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"═══ Backtest Results ═══",
            f"Total signals: {self.total_signals}",
            f"Trades taken:  {len(self.trades)}",
            f"Winners:       {self.winning_trades} ({self.win_rate:.0%})",
            f"Losers:        {self.losing_trades}",
            f"Total P&L:     {self.total_pnl:+.2f} pts",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"Avg R:R:       {self.avg_rr_achieved:.2f}",
            f"Max Drawdown:  {self.max_drawdown:.2f} pts",
            f"",
            f"── Per Strategy ──",
        ]
        for name, stats in self.per_strategy.items():
            lines.append(
                f"  {name}: {stats['trades']} trades, "
                f"{stats['win_rate']:.0%} win, "
                f"{stats['pnl']:+.2f} pts"
            )
        return "\n".join(lines)


def run_backtest(
    symbols: list[str],
    from_date: str,
    to_date: str,
    strategies: list[Strategy] = None,
    slippage_pct: float = 0.05,
) -> BacktestResult:
    """
    Run screener strategies against historical data.

    Fetches 1m candles for each symbol/day, replays through ScreenerEngine,
    then simulates trade outcomes using subsequent bars.

    Args:
        symbols: list of NSE tickers
        from_date: "YYYY-MM-DD"
        to_date: "YYYY-MM-DD"
        strategies: custom strategies (default: STRATEGIES)
        slippage_pct: % slippage on entry (applied against trade direction)
    """
    from trading.services.data_service import BrokerClient, DataService
    from trading.services.ticker_service import ticker_service

    strategies = strategies or STRATEGIES
    broker = BrokerClient.get_instance()
    broker.ensure_login()

    all_signals: list[Signal] = []
    all_candles_by_symbol: dict[str, list] = {}

    # Collect signals
    def signal_collector(signal: Signal):
        all_signals.append(signal)

    start = datetime.strptime(from_date, "%Y-%m-%d").date()
    end = datetime.strptime(to_date, "%Y-%m-%d").date()

    logger.info(f"Backtest: {len(symbols)} symbols, {from_date} → {to_date}")

    # Fetch all candles first
    for symbol in symbols:
        token = ticker_service.get_token(symbol)
        if not token:
            continue

        symbol_candles = []
        current = start
        while current <= end:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            day_str = current.strftime("%Y-%m-%d")
            raw = broker.fetch_candles(
                token, f"{day_str} 09:15", f"{day_str} 15:30", "ONE_MINUTE"
            )
            if raw:
                symbol_candles.extend(raw)
                logger.info(f"  {symbol} {day_str}: {len(raw)} candles")

            current += timedelta(days=1)

        all_candles_by_symbol[symbol] = symbol_candles

    # Replay through engine day by day
    current = start
    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        day_str = current.strftime("%Y-%m-%d")

        # Create fresh engine per day
        engine = ScreenerEngine(symbols, strategies)
        engine.add_output_handler(signal_collector)

        # Seed previous day data for pivots
        prev_day = current - timedelta(days=1)
        while prev_day.weekday() >= 5:
            prev_day -= timedelta(days=1)

        for symbol in symbols:
            token = ticker_service.get_token(symbol)
            if not token:
                continue

            # Set prev day OHLC
            prev_str = prev_day.strftime("%Y-%m-%d")
            prev_candles = [
                c for c in all_candles_by_symbol.get(symbol, [])
                if isinstance(c[0], str) and c[0].startswith(prev_str)
            ]
            if prev_candles:
                highs = [float(c[2]) for c in prev_candles]
                lows = [float(c[3]) for c in prev_candles]
                closes = [float(c[4]) for c in prev_candles]
                engine.stores[symbol].prev_day_high = max(highs)
                engine.stores[symbol].prev_day_low = min(lows)
                engine.stores[symbol].prev_day_close = closes[-1]

        # Replay today's candles interleaved by timestamp across all symbols
        # This ensures multi-TF alignment works correctly
        all_day_candles = []
        for symbol in symbols:
            day_candles = [
                c for c in all_candles_by_symbol.get(symbol, [])
                if isinstance(c[0], str) and c[0].startswith(day_str)
            ]
            for candle in day_candles:
                all_day_candles.append((symbol, candle))

        # Sort by timestamp for chronological replay
        all_day_candles.sort(key=lambda x: x[1][0])

        for symbol, candle in all_day_candles:
            ts = _parse_ts(candle[0])
            # Feed high and low as separate ticks for realistic SL/target simulation
            high = float(candle[2])
            low = float(candle[3])
            close = float(candle[4])
            vol = int(candle[5]) if len(candle) > 5 else 0
            # Feed close as the tick (bar aggregation happens in candle_store)
            engine.on_tick(symbol, close, vol, ts)

        current += timedelta(days=1)

    # Simulate trade outcomes
    trades = []
    for signal in all_signals:
        trade = _simulate_trade(signal, all_candles_by_symbol, slippage_pct)
        if trade:
            trades.append(trade)

    # Compute stats
    return _compute_stats(all_signals, trades, strategies)


def _simulate_trade(
    signal: Signal,
    all_candles: dict[str, list],
    slippage_pct: float,
) -> Optional[BacktestTrade]:
    """Simulate a trade by walking forward through subsequent candles."""
    candles = all_candles.get(signal.symbol, [])
    if not candles:
        return None

    # Find entry point — first candle after signal
    signal_ts = signal.timestamp
    entry_idx = None
    for i, c in enumerate(candles):
        ts = _parse_ts(c[0])
        if ts > signal_ts:
            entry_idx = i
            break

    if entry_idx is None:
        return None

    # Apply slippage
    entry = signal.entry
    if signal.side == "BUY":
        entry *= (1 + slippage_pct / 100)
    else:
        entry *= (1 - slippage_pct / 100)

    trade = BacktestTrade(signal=signal)
    risk = abs(entry - signal.stoploss)
    current_sl = signal.stoploss
    trailing_activated = False

    # Walk forward to see if SL or target hit
    for c in candles[entry_idx:]:
        ts = _parse_ts(c[0])
        open_p = float(c[1])
        high = float(c[2])
        low = float(c[3])
        close = float(c[4])

        if signal.side == "BUY":
            trade.max_favorable = max(trade.max_favorable, high - entry)
            trade.max_adverse = max(trade.max_adverse, entry - low)

            # Trailing stop: move SL to breakeven once +1R is reached
            if not trailing_activated and high >= entry + risk:
                current_sl = entry + 0.5  # breakeven + tiny buffer
                trailing_activated = True

            # Fix: determine SL/target check order from open direction
            # If open is closer to SL, check SL first; if closer to target, check target first
            sl_first = (open_p - current_sl) < (signal.target - open_p)

            if sl_first:
                if low <= current_sl:
                    trade.exit_price = current_sl
                    trade.exit_time = ts
                    trade.exit_reason = "Trailing BE" if trailing_activated else "SL hit"
                    trade.hit_sl = True
                    break
                if high >= signal.target:
                    trade.exit_price = signal.target
                    trade.exit_time = ts
                    trade.exit_reason = "Target hit"
                    trade.hit_target = True
                    break
            else:
                if high >= signal.target:
                    trade.exit_price = signal.target
                    trade.exit_time = ts
                    trade.exit_reason = "Target hit"
                    trade.hit_target = True
                    break
                if low <= current_sl:
                    trade.exit_price = current_sl
                    trade.exit_time = ts
                    trade.exit_reason = "Trailing BE" if trailing_activated else "SL hit"
                    trade.hit_sl = True
                    break
        else:
            # SELL side
            trade.max_favorable = max(trade.max_favorable, entry - low)
            trade.max_adverse = max(trade.max_adverse, high - entry)

            if not trailing_activated and low <= entry - risk:
                current_sl = entry - 0.5
                trailing_activated = True

            sl_first = (current_sl - open_p) < (open_p - signal.target)

            if sl_first:
                if high >= current_sl:
                    trade.exit_price = current_sl
                    trade.exit_time = ts
                    trade.exit_reason = "Trailing BE" if trailing_activated else "SL hit"
                    trade.hit_sl = True
                    break
                if low <= signal.target:
                    trade.exit_price = signal.target
                    trade.exit_time = ts
                    trade.exit_reason = "Target hit"
                    trade.hit_target = True
                    break
            else:
                if low <= signal.target:
                    trade.exit_price = signal.target
                    trade.exit_time = ts
                    trade.exit_reason = "Target hit"
                    trade.hit_target = True
                    break
                if high >= current_sl:
                    trade.exit_price = current_sl
                    trade.exit_time = ts
                    trade.exit_reason = "Trailing BE" if trailing_activated else "SL hit"
                    trade.hit_sl = True
                    break

        # End of day — close at last price
        if ts.hour == 15 and ts.minute >= 25:
            trade.exit_price = close
            trade.exit_time = ts
            trade.exit_reason = "EOD close"
            break

    if trade.exit_price == 0:
        return None

    # Compute P&L
    if signal.side == "BUY":
        trade.pnl = trade.exit_price - entry
    else:
        trade.pnl = entry - trade.exit_price
    trade.pnl_pct = round(trade.pnl / entry * 100, 2) if entry > 0 else 0

    return trade


def _compute_stats(
    signals: list[Signal],
    trades: list[BacktestTrade],
    strategies: list[Strategy],
) -> BacktestResult:
    """Aggregate backtest performance metrics."""
    result = BacktestResult(
        trades=trades,
        total_signals=len(signals),
    )

    if not trades:
        return result

    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]

    result.winning_trades = len(winners)
    result.losing_trades = len(losers)
    result.total_pnl = sum(t.pnl for t in trades)
    result.win_rate = len(winners) / len(trades) if trades else 0

    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    result.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')

    # Average R:R achieved
    rr_list = []
    for t in trades:
        risk = t.signal.risk_points
        if risk > 0:
            rr_list.append(t.pnl / risk)
    result.avg_rr_achieved = round(sum(rr_list) / len(rr_list), 2) if rr_list else 0

    # Max drawdown
    equity = 0
    peak = 0
    max_dd = 0
    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    result.max_drawdown = round(max_dd, 2)

    # Per-strategy breakdown
    for strategy in strategies:
        strat_trades = [t for t in trades if t.signal.strategy == strategy.name]
        if not strat_trades:
            continue
        strat_winners = [t for t in strat_trades if t.pnl > 0]
        result.per_strategy[strategy.name] = {
            "trades": len(strat_trades),
            "winners": len(strat_winners),
            "win_rate": len(strat_winners) / len(strat_trades),
            "pnl": round(sum(t.pnl for t in strat_trades), 2),
        }

    return result


def _parse_ts(ts_str) -> datetime:
    if isinstance(ts_str, datetime):
        return ts_str
    clean = ts_str.replace("+05:30", "").replace("T", " ")
    try:
        return datetime.strptime(clean, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.strptime(clean[:16], "%Y-%m-%d %H:%M")
