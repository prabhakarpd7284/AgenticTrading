"""Unified statistics — one implementation for all backtestors."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from trading.backtester.types import PnLMode


@dataclass
class PhaseStats:
    """Per-phase/strategy breakdown."""
    trades: int = 0
    win_rate: float = 0.0
    pnl: float = 0.0
    avg_rr: float = 0.0


@dataclass
class BacktestStats:
    """Aggregate statistics computed from a trade list."""
    total_signals: int = 0
    skipped_signals: int = 0
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

    # Breakdowns
    per_phase: Dict[str, PhaseStats] = field(default_factory=dict)
    weekly_pnl: Dict[str, float] = field(default_factory=dict)

    # Equity curve for charting
    equity_curve: List[Dict] = field(default_factory=list)


class StatsAggregator:
    """Computes BacktestStats from a list of Trade objects.

    Replaces duplicated finalize() logic across all three backtestors.
    """

    def __init__(self, capital: float, pnl_mode: PnLMode = PnLMode.RUPEES):
        self.capital = capital
        self.pnl_mode = pnl_mode

    def compute(
        self,
        trades: list,
        total_signals: int = 0,
        skipped: int = 0,
    ) -> BacktestStats:
        """Compute all stats from completed trades."""
        stats = BacktestStats(
            total_signals=total_signals,
            skipped_signals=skipped,
            total_trades=len(trades),
        )

        if not trades:
            return stats

        wins = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]

        stats.winners = len(wins)
        stats.losers = len(losses)
        stats.win_rate = stats.winners / stats.total_trades

        stats.total_pnl = sum(t.pnl for t in trades)
        if self.capital > 0:
            stats.total_pnl_pct = stats.total_pnl / self.capital * 100

        stats.avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        stats.avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        stats.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        rr_vals = [t.rr_achieved for t in trades if t.risk_points > 0]
        stats.avg_rr = sum(rr_vals) / len(rr_vals) if rr_vals else 0

        stats.best_trade = max(t.pnl for t in trades)
        stats.worst_trade = min(t.pnl for t in trades)

        bars = [t.bars_held for t in trades]
        stats.avg_bars_held = sum(bars) / len(bars) if bars else 0

        # Drawdown
        stats.max_drawdown, stats.max_drawdown_pct = self._drawdown(trades)

        # Equity curve
        stats.equity_curve = self._equity_curve(trades)

        # Per-phase breakdown
        stats.per_phase = self._phase_breakdown(trades)

        # Weekly P&L
        stats.weekly_pnl = self._weekly_pnl(trades)

        return stats

    def _drawdown(self, trades: list) -> Tuple[float, float]:
        """Peak-to-trough equity drawdown."""
        eq = self.capital
        peak = eq
        max_dd = 0.0

        for t in sorted(trades, key=lambda x: x.entry_time or ""):
            eq += t.pnl
            peak = max(peak, eq)
            dd = peak - eq
            max_dd = max(max_dd, dd)

        dd_pct = max_dd / self.capital * 100 if self.capital > 0 else 0
        return round(max_dd, 2), round(dd_pct, 2)

    def _equity_curve(self, trades: list) -> List[Dict]:
        """Build equity curve for charting."""
        curve = []
        eq = self.capital

        for t in sorted(trades, key=lambda x: x.exit_time or x.entry_time or ""):
            eq += t.pnl
            ts = t.exit_time or t.entry_time or ""
            curve.append({"t": ts, "v": round(eq, 0)})

        return curve

    def _phase_breakdown(self, trades: list) -> Dict[str, PhaseStats]:
        """Per-phase/trade_type breakdown."""
        buckets: Dict[str, list] = {}
        for t in trades:
            key = t.trade_type or t.metadata.get("phase", "unknown")
            buckets.setdefault(key, []).append(t)

        result = {}
        for phase, ts in buckets.items():
            phase_wins = [t for t in ts if t.won]
            rr_vals = [t.rr_achieved for t in ts if t.risk_points > 0]
            result[phase] = PhaseStats(
                trades=len(ts),
                win_rate=len(phase_wins) / len(ts) if ts else 0,
                pnl=round(sum(t.pnl for t in ts), 2),
                avg_rr=round(sum(rr_vals) / len(rr_vals), 2) if rr_vals else 0,
            )

        return result

    def _weekly_pnl(self, trades: list) -> Dict[str, float]:
        """Aggregate P&L by ISO week."""
        weekly: Dict[str, float] = {}
        for t in trades:
            ts = t.entry_time or ""
            try:
                d = datetime.fromisoformat(ts.replace("+05:30", "").replace("T", " ")[:10]).date()
                week_start = d - timedelta(days=d.weekday())
                key = week_start.isoformat()
                weekly[key] = weekly.get(key, 0) + t.pnl
            except (ValueError, IndexError):
                pass
        return {k: round(v, 2) for k, v in weekly.items()}
