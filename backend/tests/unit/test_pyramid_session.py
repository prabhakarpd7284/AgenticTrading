"""Re-arming the pyramid engine after each exit.

`run_pyramid` returns on its first exit — one position per run, then it is done
for the day. On 2026-09-09 that cost real money: the CE leg stopped at 11:55
and the option then ran 140.40 → 175.00 (+25%) with the engine already
finished, while the PE leg closed at 12:25 and never looked again.

This wraps the engine in a session loop: after each exit, hunt a fresh entry in
the remaining candles. Two brakes stop it revenge-trading a chop —
``max_trades`` and ``max_daily_loss``.

The engine itself is not re-tested here; a stub runner lets these tests pin the
orchestration (slicing, accumulation, stop conditions) exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from apps.trading.services.pyramid_session import run_pyramid_session


@dataclass
class _Candle:
    timestamp: str


@dataclass
class _StubResult:
    """Minimal stand-in for PyramidResult."""
    entries: list = field(default_factory=lambda: [object()])
    exit_time: str = ""
    exit_price: float = 0.0
    total_lots: int = 1
    total_cost: float = 100.0
    lot_size: int = 65

    @property
    def avg_entry(self):
        return self.total_cost / self.total_lots if self.total_lots else 0

    @property
    def pnl_per_lot(self):
        return self.exit_price - self.avg_entry if self.total_lots else 0


def _candles(n=50):
    return [_Candle(timestamp=f"T{i:02d}") for i in range(n)]


def _runner_yielding(*results):
    """Stub that returns each result in turn."""
    seq = list(results)
    calls = []

    def runner(candles, symbol=None, config=None, start_index=0):
        calls.append((candles, start_index))
        return seq.pop(0) if seq else _StubResult(entries=[])

    runner.calls = calls
    return runner


def test_single_setup_yields_one_trade():
    runner = _runner_yielding(
        _StubResult(exit_time="T10", exit_price=110.0),
        _StubResult(entries=[]),          # nothing more to find
    )
    s = run_pyramid_session(_candles(), "CE", None, _runner=runner)

    assert len(s.trades) == 1
    assert s.stopped_reason == "no_more_setups"


def test_re_enters_after_an_exit():
    """The whole point: a second setup after the first exit is taken."""
    runner = _runner_yielding(
        _StubResult(exit_time="T10", exit_price=110.0),
        _StubResult(exit_time="T30", exit_price=120.0),
        _StubResult(entries=[]),
    )
    s = run_pyramid_session(_candles(), "CE", None, _runner=runner)

    assert len(s.trades) == 2


def test_second_run_hunts_forward_from_the_previous_exit():
    """Re-entry must not re-take the trade it just closed — the start index
    moves past the exit bar, while the series itself stays whole."""
    runner = _runner_yielding(
        _StubResult(exit_time="T10", exit_price=110.0),
        _StubResult(entries=[]),
    )
    run_pyramid_session(_candles(), "CE", None, _runner=runner)

    (first_candles, first_start) = runner.calls[0]
    (second_candles, second_start) = runner.calls[1]
    assert first_start == 0
    assert second_start == 11
    assert len(first_candles) == len(second_candles) == 50


def test_stops_at_max_trades():
    runner = _runner_yielding(
        _StubResult(exit_time="T05", exit_price=110.0),
        _StubResult(exit_time="T15", exit_price=110.0),
        _StubResult(exit_time="T25", exit_price=110.0),
        _StubResult(exit_time="T35", exit_price=110.0),
    )
    s = run_pyramid_session(_candles(), "CE", None, max_trades=2, _runner=runner)

    assert len(s.trades) == 2
    assert s.stopped_reason == "max_trades"


def test_stops_when_daily_loss_cap_is_breached():
    """A chopping day must not be allowed to bleed indefinitely."""
    # entry 100, exit 90 → -10/lot × 1 lot × 65 = -650 a trade
    runner = _runner_yielding(
        _StubResult(exit_time="T05", exit_price=90.0),
        _StubResult(exit_time="T15", exit_price=90.0),
        _StubResult(exit_time="T25", exit_price=90.0),
    )
    s = run_pyramid_session(
        _candles(), "CE", None, max_daily_loss=1000.0, _runner=runner,
    )

    assert s.stopped_reason == "max_daily_loss"
    assert len(s.trades) == 2          # -650, then -1300 breaches
    assert s.realized_pnl == pytest.approx(-1300.0)


def test_accumulates_pnl_across_trades():
    runner = _runner_yielding(
        _StubResult(exit_time="T05", exit_price=110.0),   # +10 × 65 = +650
        _StubResult(exit_time="T15", exit_price=95.0),    # -5  × 65 = -325
        _StubResult(entries=[]),
    )
    s = run_pyramid_session(_candles(), "CE", None, _runner=runner)

    assert s.realized_pnl == pytest.approx(325.0)


def test_open_position_at_session_end_stops_the_loop():
    """No exit_time means still holding — there is nothing left to re-enter."""
    runner = _runner_yielding(_StubResult(exit_time="", exit_price=0.0))
    s = run_pyramid_session(_candles(), "CE", None, _runner=runner)

    assert len(s.trades) == 1
    assert s.stopped_reason == "still_open"


def test_no_setup_at_all_yields_no_trades():
    runner = _runner_yielding(_StubResult(entries=[]))
    s = run_pyramid_session(_candles(), "CE", None, _runner=runner)

    assert s.trades == []
    assert s.realized_pnl == 0.0


def test_loop_terminates_when_exit_is_the_last_candle():
    """Guard against an infinite loop when the slice would be empty."""
    runner = _runner_yielding(
        _StubResult(exit_time="T49", exit_price=110.0),
        _StubResult(exit_time="T49", exit_price=110.0),
    )
    s = run_pyramid_session(_candles(50), "CE", None, _runner=runner)

    assert len(s.trades) == 1
    assert s.stopped_reason == "session_end"


# ---------------------------------------------------------------------------
# Indicator warm-up.
#
# The first cut of this wrapper sliced the candle list after each exit. That
# silently broke the engine: EMA/RSI were then recomputed from the slice with no
# prior history, so no signal could form until warm-up completed — by which time
# the move was over. On 2026-09-09 the CE leg met every entry condition at 12:30
# (close 179.35 > ema5 160.03, rsi 61.7) and the sliced session still reported
# "no_more_setups".
#
# Re-entry must therefore hand the engine the FULL history and move only the
# index it is allowed to enter from.
# ---------------------------------------------------------------------------
def test_re_entry_preserves_indicator_history():
    seen = []

    def runner(candles, symbol=None, config=None, start_index=0):
        seen.append((len(candles), start_index))
        if len(seen) == 1:
            return _StubResult(exit_time="T10", exit_price=110.0)
        return _StubResult(entries=[])

    run_pyramid_session(_candles(50), "CE", None, _runner=runner)

    first_len, first_start = seen[0]
    second_len, second_start = seen[1]
    assert first_len == 50 and first_start == 0
    assert second_len == 50, (
        "second run must see the whole series — slicing destroys EMA/RSI warm-up"
    )
    assert second_start == 11, "hunting must resume after the previous exit"
