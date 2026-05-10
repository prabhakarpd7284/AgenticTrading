"""
Oliver Kell Cycle Backtester — walk-forward simulation on daily candles.

Simulates swing trades triggered by OK cycle phases:
  - ENTRY: On BUY phases (WP, EC, BB) when daily+weekly trends are aligned bullish
  - STOP LOSS: Below EMA20 (mid EMA) minus 0.5% buffer
  - TARGET: 2:1 risk-reward from entry
  - EXIT: Target hit | SL hit | Opposite phase (EX/WD) | Max hold (10 bars)

Position sizing: risk 1% of capital per trade, max 15% notional.

Usage:
    from trading.swing.ok_backtest import run_ok_backtest
    result = run_ok_backtest(symbols, "2026-03-01", "2026-04-30")
    print(result.summary())
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from logzero import logger

from dashboard_utils.candle_cache import _aggregate_to_weekly
from trading.config import config as trading_config
from trading.swing.ok_cycles import (
    BULLISH_ACTIONABLE,
    BEARISH_ACTIONABLE,
    CycleDetector,
    CyclePhase,
    CycleResult,
    TrendState,
)


# ══════════════════════════════════════════════
# Trade tracking
# ══════════════════════════════════════════════

@dataclass
class SwingTrade:
    """A single simulated swing trade."""
    symbol: str
    side: str                    # BUY or SHORT
    phase: str                   # Phase code that triggered entry
    entry_date: str
    entry_price: float
    stoploss: float
    target: float
    quantity: int
    notional: float              # entry * quantity

    exit_date: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""        # "Target", "SL", "Phase reversal", "Max hold", "End of test"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    risk_points: float = 0.0
    rr_achieved: float = 0.0
    bars_held: int = 0
    max_favorable: float = 0.0   # Best unrealized P&L in points
    max_adverse: float = 0.0     # Worst unrealized drawdown in points

    @property
    def won(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    """Aggregate results from the backtest."""
    from_date: str
    to_date: str
    capital: float
    symbols_scanned: int
    trading_days: int

    trades: List[SwingTrade] = field(default_factory=list)
    total_signals: int = 0
    skipped_signals: int = 0     # Skipped due to max positions / risk

    # Computed in finalize()
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_bars_held: float = 0.0

    # Per-phase breakdown
    phase_stats: Dict[str, dict] = field(default_factory=dict)

    # Weekly P&L
    weekly_pnl: Dict[str, float] = field(default_factory=dict)

    def finalize(self):
        """Compute aggregate stats from trade list."""
        self.total_trades = len(self.trades)
        if self.total_trades == 0:
            return

        wins = [t for t in self.trades if t.won]
        losses = [t for t in self.trades if not t.won]

        self.winners = len(wins)
        self.losers = len(losses)
        self.win_rate = self.winners / self.total_trades

        self.total_pnl = sum(t.pnl for t in self.trades)
        self.total_pnl_pct = self.total_pnl / self.capital * 100

        self.avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        self.avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        rr_values = [t.rr_achieved for t in self.trades if t.risk_points > 0]
        self.avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0

        self.best_trade = max(t.pnl for t in self.trades)
        self.worst_trade = min(t.pnl for t in self.trades)

        bars = [t.bars_held for t in self.trades]
        self.avg_bars_held = sum(bars) / len(bars) if bars else 0

        # Max drawdown (peak-to-trough equity curve)
        equity = self.capital
        peak = equity
        max_dd = 0.0
        for t in sorted(self.trades, key=lambda x: x.entry_date):
            equity += t.pnl
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        self.max_drawdown = max_dd
        self.max_drawdown_pct = max_dd / self.capital * 100

        # Per-phase breakdown
        phases: Dict[str, list] = {}
        for t in self.trades:
            phases.setdefault(t.phase, []).append(t)
        for phase, trades in phases.items():
            phase_wins = [t for t in trades if t.won]
            self.phase_stats[phase] = {
                "trades": len(trades),
                "win_rate": len(phase_wins) / len(trades) if trades else 0,
                "pnl": sum(t.pnl for t in trades),
                "avg_rr": sum(t.rr_achieved for t in trades if t.risk_points > 0) / max(1, len(trades)),
            }

        # Weekly P&L
        for t in self.trades:
            # Get the Monday of the entry week
            try:
                d = datetime.strptime(t.entry_date, "%Y-%m-%d").date()
                week_start = d - timedelta(days=d.weekday())
                key = week_start.isoformat()
                self.weekly_pnl[key] = self.weekly_pnl.get(key, 0) + t.pnl
            except ValueError:
                pass

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"{'═' * 55}",
            f"  Oliver Kell Cycle Backtest",
            f"  {self.from_date} → {self.to_date}",
            f"{'═' * 55}",
            f"  Capital: ₹{self.capital:,.0f}",
            f"  Symbols scanned: {self.symbols_scanned}",
            f"  Trading days: {self.trading_days}",
            f"{'─' * 55}",
            f"  Signals generated: {self.total_signals}",
            f"  Skipped (risk): {self.skipped_signals}",
            f"  Trades taken: {self.total_trades}",
            f"  Winners: {self.winners} ({self.win_rate:.0%})",
            f"  Losers: {self.losers}",
            f"{'─' * 55}",
            f"  Total P&L: ₹{self.total_pnl:+,.0f} ({self.total_pnl_pct:+.2f}%)",
            f"  Avg winner: ₹{self.avg_win:+,.0f}",
            f"  Avg loser: ₹{self.avg_loss:+,.0f}",
            f"  Best trade: ₹{self.best_trade:+,.0f}",
            f"  Worst trade: ₹{self.worst_trade:+,.0f}",
            f"{'─' * 55}",
            f"  Profit factor: {self.profit_factor:.2f}",
            f"  Avg R:R achieved: {self.avg_rr:.2f}",
            f"  Max drawdown: ₹{self.max_drawdown:,.0f} ({self.max_drawdown_pct:.2f}%)",
            f"  Avg holding: {self.avg_bars_held:.1f} days",
        ]

        if self.phase_stats:
            lines.append(f"{'─' * 55}")
            lines.append("  Per-Phase Breakdown:")
            for phase, s in sorted(self.phase_stats.items()):
                lines.append(
                    f"    {phase:>8}: {s['trades']} trades, "
                    f"{s['win_rate']:.0%} win, ₹{s['pnl']:+,.0f}"
                )

        if self.weekly_pnl:
            lines.append(f"{'─' * 55}")
            lines.append("  Weekly P&L:")
            for week, pnl in sorted(self.weekly_pnl.items()):
                icon = "🟢" if pnl >= 0 else "🔴"
                lines.append(f"    {week}: {icon} ₹{pnl:+,.0f}")

        lines.append(f"{'═' * 55}")
        return "\n".join(lines)

    def telegram_message(self) -> str:
        """Formatted Telegram message (HTML)."""
        def r(n: float) -> str:
            """Format rupee amount without problematic chars."""
            sign = "+" if n >= 0 else "-"
            return f"{sign}₹{abs(n):,.0f}"

        lines = [
            f"📊 <b>Oliver Kell Backtest Report</b>",
            f"<i>{self.from_date} → {self.to_date}</i>\n",
            f"💰 Capital: ₹{self.capital:,.0f}",
            f"📈 Symbols: {self.symbols_scanned} | Days: {self.trading_days}\n",
            f"<b>Results:</b>",
            f"  Trades: {self.total_trades} ({self.total_signals} signals, {self.skipped_signals} skipped)",
            f"  Winners: {self.winners} ({self.win_rate:.0%})",
            f"  Total P&amp;L: <b>{r(self.total_pnl)}</b> ({self.total_pnl_pct:+.2f}%)",
            f"  Profit Factor: {self.profit_factor:.2f}",
            f"  Avg R:R: {self.avg_rr:.2f}",
            f"  Max Drawdown: ₹{self.max_drawdown:,.0f} ({self.max_drawdown_pct:.2f}%)\n",
            f"<b>Avg Trade:</b>",
            f"  Winner: {r(self.avg_win)} | Loser: {r(self.avg_loss)}",
            f"  Best: {r(self.best_trade)} | Worst: {r(self.worst_trade)}",
            f"  Hold: {self.avg_bars_held:.1f} days",
        ]

        if self.phase_stats:
            lines.append(f"\n<b>Per Phase:</b>")
            for phase, s in sorted(self.phase_stats.items()):
                emoji = "🚀" if phase == "WP" else "📈" if phase == "EC" else "💥" if phase == "BB" else "📉"
                lines.append(
                    f"  {emoji} {phase}: {s['trades']}t, "
                    f"{s['win_rate']:.0%}W, {r(s['pnl'])}"
                )

        if self.weekly_pnl:
            lines.append(f"\n<b>Weekly P&amp;L:</b>")
            for week, pnl in sorted(self.weekly_pnl.items()):
                icon = "🟢" if pnl >= 0 else "🔴"
                lines.append(f"  {week}: {icon} {r(pnl)}")

        verdict = "✅ PROFITABLE" if self.total_pnl > 0 else "❌ LOSS"
        lines.append(f"\n<b>Verdict: {verdict}</b>")

        return "\n".join(lines)


# ══════════════════════════════════════════════
# Backtest engine
# ══════════════════════════════════════════════

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
) -> BacktestResult:
    """
    Walk-forward backtest of the Oliver Kell cycle strategy.

    Args:
        symbols: List of NSE symbols to scan
        from_date: Backtest start date (YYYY-MM-DD). Data before this is warmup only.
        to_date: Backtest end date (YYYY-MM-DD)
        capital: Starting capital (default: from config)
        max_risk_pct: Max % of capital risked per trade (default: 1%)
        max_position_pct: Max % of capital per position (default: 15%)
        max_positions: Max concurrent positions (default: 11)
        min_rr: Minimum risk:reward ratio for targets (default: 2.0)
        max_hold_bars: Max bars to hold a position (default: 10 days)
        slippage_pct: Slippage % applied to entry (default: 0.1%)
        data_svc: Optional DataService override

    Returns:
        BacktestResult with all trades, stats, and formatted summaries
    """
    risk_cfg = trading_config.risk
    capital = capital or risk_cfg.default_capital
    max_risk_pct = max_risk_pct or risk_cfg.max_risk_per_trade_pct
    max_position_pct = max_position_pct or risk_cfg.max_position_size_pct
    max_positions = max_positions or risk_cfg.max_open_positions

    detector = CycleDetector()

    # Need warmup period before from_date for EMA50
    warmup_days = trading_config.ok_cycle.lookback_days
    fetch_from = (
        datetime.strptime(from_date, "%Y-%m-%d").date()
        - timedelta(days=warmup_days)
    ).strftime("%Y-%m-%d")

    logger.info(
        f"OK Backtest: {from_date} → {to_date} | "
        f"{len(symbols)} symbols | capital ₹{capital:,.0f} | "
        f"data from {fetch_from}"
    )

    # ── Fetch all daily data ──
    if data_svc is None:
        from trading.services.data_service import DataService
        data_svc = DataService()

    all_data: Dict[str, List[dict]] = {}
    fetch_start = time.time()

    for i, symbol in enumerate(symbols):
        if (i + 1) % 20 == 0:
            logger.info(f"  Fetching data: {i + 1}/{len(symbols)}")
        try:
            candles = data_svc.fetch_historical(
                symbol, fetch_from, to_date, interval="ONE_DAY"
            )
            if candles and len(candles) >= trading_config.ok_cycle.ema_slow + 10:
                all_data[symbol] = candles
        except Exception as e:
            logger.warning(f"  Skip {symbol}: {e}")

    fetch_elapsed = time.time() - fetch_start
    logger.info(
        f"  Data fetched: {len(all_data)}/{len(symbols)} symbols "
        f"in {fetch_elapsed:.1f}s"
    )

    # ── Build trading day list ──
    from_dt = datetime.strptime(from_date, "%Y-%m-%d").date()
    to_dt = datetime.strptime(to_date, "%Y-%m-%d").date()

    # Get all unique dates from the data
    all_dates = set()
    for candles in all_data.values():
        for c in candles:
            d = c["date"][:10] if isinstance(c["date"], str) else str(c["date"])[:10]
            all_dates.add(d)

    trading_days = sorted(d for d in all_dates if from_date <= d <= to_date)
    logger.info(f"  Trading days in range: {len(trading_days)}")

    # ── Walk-forward simulation ──
    result = BacktestResult(
        from_date=from_date,
        to_date=to_date,
        capital=capital,
        symbols_scanned=len(all_data),
        trading_days=len(trading_days),
    )

    open_trades: List[SwingTrade] = []
    equity = capital
    daily_pnl_today = 0.0

    for day_idx, trade_date in enumerate(trading_days):
        daily_pnl_today = 0.0

        # ── Check exits on open trades (with trailing SL + breakeven) ──
        mgr = trading_config.trade_manager
        closed_today = []
        for trade in open_trades:
            candles = all_data.get(trade.symbol, [])
            today_candle = _find_candle(candles, trade_date)
            if not today_candle:
                continue

            trade.bars_held += 1
            open_p = today_candle["open"]
            high = today_candle["high"]
            low = today_candle["low"]
            close = today_candle["close"]

            # Track excursions
            if trade.side == "BUY":
                fav = high - trade.entry_price
                adv = trade.entry_price - low
            else:
                fav = trade.entry_price - low
                adv = high - trade.entry_price
            trade.max_favorable = max(trade.max_favorable, fav)
            trade.max_adverse = max(trade.max_adverse, adv)

            # Current R multiple
            risk = trade.risk_points
            if risk > 0:
                current_r = fav / risk
            else:
                current_r = 0

            # ── Breakeven: move SL to entry when +0.5R ──
            if not getattr(trade, "_be_done", False) and current_r >= mgr.breakeven_at_r:
                if trade.side == "BUY":
                    trade.stoploss = max(trade.stoploss, trade.entry_price + 0.5)
                else:
                    trade.stoploss = min(trade.stoploss, trade.entry_price - 0.5)
                trade._be_done = True  # type: ignore[attr-defined]

            # ── Trailing SL after +1R ──
            if current_r >= mgr.partial_at_r:
                trail_factor = mgr.trail_atr_factor
                if current_r >= mgr.tight_trail_at_r:
                    trail_factor = mgr.tight_trail_atr_factor

                if trade.side == "BUY":
                    trail_sl = high - (risk * trail_factor * 2)
                    trade.stoploss = max(trade.stoploss, trail_sl)
                else:
                    trail_sl = low + (risk * trail_factor * 2)
                    trade.stoploss = min(trade.stoploss, trail_sl)

            # ── Smart exit priority based on open direction ──
            if trade.side == "BUY":
                sl_first = (open_p - trade.stoploss) < (trade.target - open_p)
            else:
                sl_first = (trade.stoploss - open_p) < (open_p - trade.target)

            hit_sl = False
            hit_target = False

            if trade.side == "BUY":
                hit_sl = low <= trade.stoploss
                hit_target = high >= trade.target
            else:
                hit_sl = high >= trade.stoploss
                hit_target = low <= trade.target

            if hit_sl and hit_target:
                # Both hit — use priority
                if sl_first:
                    hit_target = False
                else:
                    hit_sl = False

            if hit_sl:
                reason = "Trail SL" if getattr(trade, "_be_done", False) else "SL hit"
                _close_trade(trade, trade.stoploss, trade_date, reason)
                closed_today.append(trade)
                continue
            if hit_target:
                _close_trade(trade, trade.target, trade_date, "Target hit")
                closed_today.append(trade)
                continue

            # Max hold
            if trade.bars_held >= max_hold_bars:
                _close_trade(trade, close, trade_date, "Max hold")
                closed_today.append(trade)
                continue

        # Remove closed trades
        for t in closed_today:
            open_trades.remove(t)
            equity += t.pnl
            daily_pnl_today += t.pnl
            result.trades.append(t)

        # ── Check daily loss limit ──
        daily_loss_limit = capital * trading_config.risk.max_daily_loss_pct / 100
        if daily_pnl_today < -daily_loss_limit:
            continue  # Skip new entries today

        # ── Scan for new entries ──
        for symbol, candles in all_data.items():
            # Skip if already in a trade for this symbol
            if any(t.symbol == symbol for t in open_trades):
                continue

            # Max positions check
            if len(open_trades) >= max_positions:
                break

            # Get candles up to and including today
            today_idx = _find_candle_idx(candles, trade_date)
            if today_idx < 0:
                continue

            daily_slice = candles[: today_idx + 1]
            if len(daily_slice) < trading_config.ok_cycle.ema_slow + 10:
                continue

            # Compute weekly from available daily
            weekly_slice = _to_weekly(daily_slice)

            # Analyze
            cycle_result = detector.analyze(symbol, daily_slice, weekly_slice)

            # Enter on bullish actionable phases (BUY) or bearish (SHORT)
            is_buy = cycle_result.phase in BULLISH_ACTIONABLE
            is_short = cycle_result.phase in BEARISH_ACTIONABLE
            if not is_buy and not is_short:
                continue

            result.total_signals += 1
            side = "BUY" if is_buy else "SHORT"

            today_candle = candles[today_idx]
            ema_mid = cycle_result.ema_mid

            if side == "BUY":
                entry_price = today_candle["close"] * (1 + slippage_pct / 100)
                # SL: below EMA20 with 0.5% buffer
                sl_price = ema_mid * 0.995
                risk_points = entry_price - sl_price
            else:
                entry_price = today_candle["close"] * (1 - slippage_pct / 100)
                # SL: above EMA20 with 0.5% buffer
                sl_price = ema_mid * 1.005
                risk_points = sl_price - entry_price

            if risk_points <= 0:
                result.skipped_signals += 1
                continue

            # Target: min_rr × risk
            if side == "BUY":
                target_price = entry_price + (risk_points * min_rr)
            else:
                target_price = entry_price - (risk_points * min_rr)

            # Position sizing: risk max_risk_pct of capital
            max_risk_rupees = equity * max_risk_pct / 100
            qty = int(max_risk_rupees / risk_points)
            if qty <= 0:
                result.skipped_signals += 1
                continue

            # Cap notional at max_position_pct
            notional = qty * entry_price
            max_notional = equity * max_position_pct / 100
            if notional > max_notional:
                qty = int(max_notional / entry_price)
                notional = qty * entry_price

            if qty <= 0:
                result.skipped_signals += 1
                continue

            trade = SwingTrade(
                symbol=symbol,
                side=side,
                phase=cycle_result.phase.value,
                entry_date=trade_date,
                entry_price=round(entry_price, 2),
                stoploss=round(sl_price, 2),
                target=round(target_price, 2),
                quantity=qty,
                notional=round(notional, 2),
                risk_points=round(risk_points, 2),
            )
            open_trades.append(trade)

    # ── Close remaining open trades at last available price ──
    for trade in open_trades:
        candles = all_data.get(trade.symbol, [])
        if candles:
            last = candles[-1]
            _close_trade(trade, last["close"], last["date"][:10], "End of test")
        result.trades.append(trade)

    result.finalize()
    logger.info(f"OK Backtest complete: {result.total_trades} trades, ₹{result.total_pnl:+,.0f} P&L")
    return result


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def _find_candle(candles: List[dict], trade_date: str) -> Optional[dict]:
    """Find candle matching a date."""
    for c in candles:
        d = c["date"][:10] if isinstance(c["date"], str) else str(c["date"])[:10]
        if d == trade_date:
            return c
    return None


def _find_candle_idx(candles: List[dict], trade_date: str) -> int:
    """Find index of candle matching a date."""
    for i, c in enumerate(candles):
        d = c["date"][:10] if isinstance(c["date"], str) else str(c["date"])[:10]
        if d == trade_date:
            return i
    return -1


def _close_trade(trade: SwingTrade, exit_price: float, exit_date: str, reason: str):
    """Close a trade and compute P&L."""
    trade.exit_price = round(exit_price, 2)
    trade.exit_date = exit_date
    trade.exit_reason = reason

    if trade.side == "BUY":
        pnl_per_share = exit_price - trade.entry_price
    else:
        pnl_per_share = trade.entry_price - exit_price

    trade.pnl = round(pnl_per_share * trade.quantity, 2)
    trade.pnl_pct = round(pnl_per_share / trade.entry_price * 100, 2)

    if trade.risk_points > 0:
        trade.rr_achieved = round(pnl_per_share / trade.risk_points, 2)


def _to_weekly(daily_candles: List[dict]) -> List[dict]:
    """Convert daily candles to weekly (adapts date key to timestamp key)."""
    adapted = []
    for c in daily_candles:
        adapted.append({
            "timestamp": c.get("date", c.get("timestamp", "")),
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "volume": c["volume"],
        })
    return _aggregate_to_weekly(adapted)
