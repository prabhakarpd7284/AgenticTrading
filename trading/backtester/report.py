"""Report formatting — CLI + Telegram HTML."""
from __future__ import annotations

from trading.backtester.stats import BacktestStats
from trading.backtester.types import PnLMode


class ReportFormatter:
    """Generate human-readable reports from BacktestStats."""

    def __init__(self, title: str = "Backtest", pnl_mode: PnLMode = PnLMode.RUPEES):
        self.title = title
        self.pnl_mode = pnl_mode

    def _fmt(self, n: float) -> str:
        """Format a P&L value."""
        if self.pnl_mode == PnLMode.POINTS:
            return f"{n:+,.2f} pts"
        sign = "+" if n >= 0 else "-"
        return f"{sign}₹{abs(n):,.0f}"

    def cli_summary(self, stats: BacktestStats, meta: dict = None) -> str:
        """Terminal output."""
        meta = meta or {}
        lines = [
            f"{'═' * 60}",
            f"  {self.title}",
        ]
        if meta.get("from_date") and meta.get("to_date"):
            lines.append(f"  {meta['from_date']} → {meta['to_date']}")
        lines += [
            f"{'═' * 60}",
            f"  Capital: ₹{meta.get('capital', 0):,.0f}" if meta.get("capital") else "",
            f"  Symbols: {meta.get('symbols', '?')} | Days: {meta.get('trading_days', '?')}",
            f"{'─' * 60}",
            f"  Signals: {stats.total_signals} | Skipped: {stats.skipped_signals}",
            f"  Trades: {stats.total_trades}",
            f"  Winners: {stats.winners} ({stats.win_rate:.0%})",
            f"  Losers: {stats.losers}",
            f"{'─' * 60}",
            f"  Total P&L: {self._fmt(stats.total_pnl)} ({stats.total_pnl_pct:+.2f}%)",
            f"  Avg winner: {self._fmt(stats.avg_win)}",
            f"  Avg loser: {self._fmt(stats.avg_loss)}",
            f"  Best: {self._fmt(stats.best_trade)} | Worst: {self._fmt(stats.worst_trade)}",
            f"{'─' * 60}",
            f"  Profit Factor: {stats.profit_factor:.2f}",
            f"  Avg R:R: {stats.avg_rr:.2f}",
            f"  Max Drawdown: {self._fmt(stats.max_drawdown)} ({stats.max_drawdown_pct:.2f}%)",
            f"  Avg Hold: {stats.avg_bars_held:.1f} bars",
        ]

        if stats.per_phase:
            lines.append(f"{'─' * 60}")
            lines.append("  Per Phase:")
            for phase, s in sorted(stats.per_phase.items()):
                lines.append(
                    f"    {phase:>8}: {s.trades}t, {s.win_rate:.0%}W, {self._fmt(s.pnl)}"
                )

        if stats.weekly_pnl:
            lines.append(f"{'─' * 60}")
            lines.append("  Weekly P&L:")
            for week, pnl in sorted(stats.weekly_pnl.items()):
                icon = "+" if pnl >= 0 else "-"
                lines.append(f"    {week}: {self._fmt(pnl)}")

        lines.append(f"{'═' * 60}")
        return "\n".join(l for l in lines if l)

    def telegram_html(self, stats: BacktestStats, meta: dict = None) -> str:
        """HTML formatted for Telegram sendMessage parse_mode='HTML'."""
        meta = meta or {}

        def r(n: float) -> str:
            sign = "+" if n >= 0 else "-"
            return f"{sign}₹{abs(n):,.0f}"

        lines = [
            f"<b>{self.title}</b>",
        ]
        if meta.get("from_date"):
            lines.append(f"<i>{meta.get('from_date', '')} → {meta.get('to_date', '')}</i>")
        if meta.get("capital"):
            lines.append(f"Capital: ₹{meta['capital']:,.0f}")
        lines.append("")

        lines += [
            f"<b>Results:</b>",
            f"  Trades: {stats.total_trades} ({stats.total_signals} signals)",
            f"  Winners: {stats.winners} ({stats.win_rate:.0%})",
            f"  PnL: <b>{r(stats.total_pnl)}</b> ({stats.total_pnl_pct:+.1f}%)",
            f"  PF: {stats.profit_factor:.2f} | R:R: {stats.avg_rr:.2f}",
            f"  MaxDD: {r(-stats.max_drawdown)} ({stats.max_drawdown_pct:.1f}%)",
            "",
            f"<b>Avg:</b> W:{r(stats.avg_win)} L:{r(stats.avg_loss)} Hold:{stats.avg_bars_held:.0f}bars",
        ]

        if stats.per_phase:
            lines.append(f"\n<b>Phases:</b>")
            for ph, s in sorted(stats.per_phase.items(), key=lambda x: -x[1].pnl):
                lines.append(f"  {ph}: {s.trades}t {s.win_rate:.0%}W {r(s.pnl)}")

        verdict = "PROFITABLE" if stats.total_pnl > 0 else "LOSS"
        lines.append(f"\n<b>Verdict: {verdict}</b>")

        return "\n".join(lines)

    def trade_table(self, trades: list, max_rows: int = 50) -> str:
        """Tabular trade log for CLI."""
        if not trades:
            return "  No trades."

        header = (
            f"{'Symbol':<12} {'Type':>5} {'Entry':>10} {'SL':>10} "
            f"{'Target':>10} {'Exit':>10} {'Reason':<14} "
            f"{'P&L':>12} {'R:R':>5} {'Bars':>4}"
        )
        lines = [f"{'─' * len(header)}", header, f"{'─' * len(header)}"]

        for t in trades[:max_rows]:
            pnl_str = self._fmt(t.pnl)
            lines.append(
                f"{t.symbol:<12} {t.trade_type:>5} {t.entry_price:>10,.2f} "
                f"{t.stoploss:>10,.2f} {t.target:>10,.2f} "
                f"{t.exit_price:>10,.2f} {t.exit_reason:<14} "
                f"{pnl_str:>12} {t.rr_achieved:>5.1f} {t.bars_held:>4}"
            )

        if len(trades) > max_rows:
            lines.append(f"  ... and {len(trades) - max_rows} more trades")

        return "\n".join(lines)
