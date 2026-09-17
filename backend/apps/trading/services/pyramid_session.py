"""Run the pyramid engine repeatedly across a session.

`run_pyramid` returns on its first exit — one position, then done for the day.
On 2026-09-09 that was measurably costly: the CE leg stopped out at 11:55 and
the option then ran 140.40 → 175.00 (+25%) with the engine already finished,
while the PE leg closed at 12:25 and never looked again. A one-shot engine
structurally cannot trade a day that chops and re-trends.

This wraps it in a session loop: after each exit, hunt a fresh entry in the
candles that remain. Two brakes keep re-entry from becoming revenge-trading —
`max_trades` and `max_daily_loss`. The engine is untouched, so the existing
CLI and UI keep their current one-shot behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import structlog

log = structlog.get_logger(__name__)

# Conservative defaults. Intraday options chop; without a cap a re-arming
# engine will happily take ten losing entries in a sideways hour.
DEFAULT_MAX_TRADES = 3


@dataclass
class SessionResult:
    symbol: str
    trades: list = field(default_factory=list)
    realized_pnl: float = 0.0          # rupees, across all trades
    stopped_reason: str = ""

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if _rupees(t) > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if _rupees(t) < 0)


def _rupees(result) -> float:
    """Realized P&L of one pyramid trade, in rupees."""
    return result.pnl_per_lot * result.total_lots * result.lot_size


def run_pyramid_session(
    candles,
    symbol: str,
    config,
    *,
    max_trades: int = DEFAULT_MAX_TRADES,
    max_daily_loss: float | None = None,
    _runner=None,
) -> SessionResult:
    """Take up to ``max_trades`` pyramid positions across ``candles``.

    ``max_daily_loss`` is a positive rupee figure; once cumulative realized P&L
    falls to -that, the session stops taking new entries.
    """
    if _runner is None:
        from plugins.strategy_pyramid.strategy import run_pyramid as _runner

    session = SessionResult(symbol=symbol)
    series = list(candles)
    start_index = 0

    while True:
        if len(session.trades) >= max_trades:
            session.stopped_reason = "max_trades"
            break
        if start_index >= len(series):
            session.stopped_reason = "session_end"
            break

        # Always hand over the FULL series. The engine needs the whole history
        # for EMA/RSI warm-up; only the bar it may enter from moves forward.
        result = _runner(series, symbol, config, start_index=start_index)

        if not result.entries:
            session.stopped_reason = "no_more_setups"
            break

        session.trades.append(result)
        session.realized_pnl += _rupees(result)

        if max_daily_loss is not None and session.realized_pnl <= -abs(max_daily_loss):
            session.stopped_reason = "max_daily_loss"
            break

        if not result.exit_time:
            # Still holding at the end of the data — nothing left to re-enter.
            session.stopped_reason = "still_open"
            break

        # Hunt forward only — re-entering at or before the exit bar would
        # re-take the trade we just closed.
        next_start = next(
            (i for i, c in enumerate(series) if c.timestamp > result.exit_time),
            None,
        )
        if next_start is None:
            session.stopped_reason = "session_end"
            break
        start_index = next_start

    log.info(
        "pyramid_session.done",
        symbol=symbol, trades=len(session.trades),
        pnl=round(session.realized_pnl, 2), reason=session.stopped_reason,
    )
    return session
