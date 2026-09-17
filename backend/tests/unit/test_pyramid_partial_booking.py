"""Scale out part of the position at a fixed R multiple.

Measured on 2026-09-09 (NIFTY 23500 PE, entry 97.80, risk 12.07/lot, 12 lots):

    book at 1R (109.87, hit 12:00)  +9,415
    pyramid (what it did)           +1,775
    entry only, trail               -5,499

Booking beat pyramiding 5x on that trade. But a fixed target caps the outsized
move pyramiding exists to capture, so the rule here is a *hybrid*: take
``book_fraction`` of the lots off at ``book_at_r``, let the rest pyramid and
trail. Banks the chop days, still rides the trend days.

``book_at_r = 0`` disables it, so existing behaviour is unchanged by default.
"""
from __future__ import annotations

import pytest

from plugins.strategy_pyramid.strategy import Candle, PyramidConfig, run_pyramid


def _rally(n_flat=26, n_up=14, base=100.0, step=3.0):
    """Flat warm-up, then a clean rally — enough to trigger an entry and run."""
    candles = []
    for i in range(n_flat):
        p = base + (i % 2) * 0.2          # tiny noise so indicators are defined
        candles.append(Candle(
            timestamp=f"2026-09-09T09:{15 + i:02d}:00+05:30",
            open=p, high=p + 0.3, low=p - 0.3, close=p, volume=1000,
        ))
    p = base
    for i in range(n_up):
        p += step
        candles.append(Candle(
            timestamp=f"2026-09-09T10:{i:02d}:00+05:30",
            open=p - step, high=p + 1.0, low=p - step - 0.5, close=p, volume=2000,
        ))
    return candles


def _cfg(**kw):
    base = dict(lot_size=65, initial_capital=500000, initial_risk_pct=2.0,
                max_pyramids=5, eod_hour=23, eod_minute=59)
    base.update(kw)
    return PyramidConfig(**base)


def test_the_fixture_actually_triggers_an_entry():
    """Guard the test's own premise — everything below is meaningless if the
    engine never enters on this series."""
    r = run_pyramid(_rally(), "TEST", _cfg())
    assert r.entries, "fixture must produce an entry"


def test_booking_is_off_by_default():
    r = run_pyramid(_rally(), "TEST", _cfg())
    assert r.booked_lots == 0
    assert r.booked_pnl_points == 0.0


def test_booking_takes_a_fraction_off_at_the_r_multiple():
    r = run_pyramid(_rally(), "TEST", _cfg(book_at_r=1.0, book_fraction=0.5))

    assert r.booked_lots > 0, "half the position should have been booked at 1R"
    # Compare against the INITIAL entry, not avg_entry: the runner pyramids at
    # higher prices after the booking, so the final average is above the level
    # we booked at and tells us nothing about whether booking was profitable.
    assert r.booked_price > r.entries[0].price
    assert r.booked_pnl_points > 0


def test_booked_lots_leave_the_position():
    """The runner must be smaller than the original size, or nothing was sold."""
    full = run_pyramid(_rally(), "TEST", _cfg())
    hybrid = run_pyramid(_rally(), "TEST", _cfg(book_at_r=1.0, book_fraction=0.5))

    original_lots = hybrid.entries[0].lots
    assert hybrid.booked_lots < original_lots or hybrid.booked_lots == original_lots
    assert full.entries[0].lots == original_lots


def test_total_pnl_includes_both_the_booked_part_and_the_runner():
    r = run_pyramid(_rally(), "TEST", _cfg(book_at_r=1.0, book_fraction=0.5))

    runner_points = (r.exit_price - r.avg_entry) * r.total_lots
    assert r.total_pnl_points_all == pytest.approx(
        r.booked_pnl_points + runner_points
    )


def test_booking_everything_closes_the_trade():
    """book_fraction=1.0 is the pure-target rule — no runner is left."""
    r = run_pyramid(_rally(), "TEST", _cfg(book_at_r=1.0, book_fraction=1.0))

    assert r.booked_lots > 0
    assert r.total_lots == 0
    assert r.exit_reason in ("Booked at target", "EOD", "Trail SL")


def test_a_target_that_is_never_reached_books_nothing():
    r = run_pyramid(_rally(), "TEST", _cfg(book_at_r=50.0, book_fraction=0.5))

    assert r.booked_lots == 0
    assert r.booked_pnl_points == 0.0
