"""Regression tests for two backtest-correctness bugs surfaced on the Monthly
page's run_ok_backtest output (2026-06-26):

1. Entry-window leak — the engine fetched a 120-day warmup lookback but ran
   entries across ALL of it, so a June backtest booked trades from Feb–May
   (Weekly P&L started 2026-05-04 for a `--from 2026-06-01` run). Entries must
   be restricted to the requested window; pre-window bars are warmup only.

2. Drawdown sign — the CLI report rendered the positive drawdown magnitude
   through `_fmt()`, printing it as a GAIN (`+₹18,637`) instead of the loss it
   is (`-₹18,637`). The telegram report already negated correctly.
"""
from plugins.strategy_backtest.engine import BacktestEngine, EngineConfig
from plugins.strategy_backtest.exits import ExitCheck
from plugins.strategy_backtest.report import ReportFormatter
from plugins.strategy_backtest.stats import BacktestStats
from plugins.strategy_backtest.types import Bar, EntrySignal, ExitSignal, TradeSide


class _EveryBarLong:
    """Stub detector: emit a BUY on every bar (so the window is the only thing
    that can stop an entry)."""

    def detect(self, symbol, bars, bar_index, context):
        b = bars[bar_index]
        return [EntrySignal(
            symbol=symbol,
            timestamp=b.timestamp,
            side=TradeSide.BUY,
            entry_price=b.close,
            stoploss=b.close * 0.97,
            target=b.close * 1.06,
            risk_points=b.close * 0.03,
            metadata={"phase": "WP", "trade_type": "WP"},
        )]


class _ExitNextBar(ExitCheck):
    """Close any open trade on the next bar so a fresh entry can open daily."""

    def check(self, trade, bar):
        return ExitSignal(price=bar.close, reason="test-exit")


def _bars(symbol, dates):
    return [Bar(timestamp=f"{d} 00:00", open=100.0, high=101.0,
                low=99.0, close=100.0, volume=1000) for d in dates]


def test_entries_restricted_to_window_warmup_bars_excluded():
    """Bars before entry_from feed warmup/exits but must NOT open trades."""
    dates = [
        "2026-05-28", "2026-05-29",          # before the window (warmup)
        "2026-06-01", "2026-06-02", "2026-06-03",  # inside the window
    ]
    data = {"TEST": _bars("TEST", dates)}

    engine = BacktestEngine(
        config=EngineConfig(
            capital=500_000, max_risk_pct=1.0, max_position_pct=15.0,
            entry_from="2026-06-01", entry_to="2026-06-30",
        ),
        entry=_EveryBarLong(),
        exits=[_ExitNextBar()],
    )
    stats = engine.run(data)

    assert stats.total_trades > 0, "expected in-window entries"
    # Every booked trade's entry date must be inside [from, to].
    assert all(d not in stats.weekly_pnl for d in ("2026-05-25",)), \
        "no May week should appear"
    for week in stats.weekly_pnl:
        assert week >= "2026-06-01", f"week {week} leaked from before the window"


def test_no_window_keeps_legacy_behaviour():
    """Without a window every bar can enter (back-compat preserved)."""
    dates = ["2026-05-28", "2026-05-29", "2026-06-01"]
    data = {"TEST": _bars("TEST", dates)}

    engine = BacktestEngine(
        config=EngineConfig(capital=500_000),
        entry=_EveryBarLong(),
        exits=[_ExitNextBar()],
    )
    stats = engine.run(data)
    # May entries ARE allowed when no window is set.
    assert any(w < "2026-06-01" for w in stats.weekly_pnl), \
        "unbounded run should still trade pre-June bars"


def test_cli_report_renders_drawdown_as_loss():
    """Max Drawdown must print as a loss (-₹), never a gain (+₹)."""
    stats = BacktestStats(max_drawdown=18_637.0, max_drawdown_pct=3.73)
    out = ReportFormatter(title="OK Backtest").cli_summary(stats, meta={})
    assert "Max Drawdown: -₹18,637" in out
    assert "+₹18,637" not in out
