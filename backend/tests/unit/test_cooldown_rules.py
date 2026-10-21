"""Losing-streak cooldown state machine (TRADING_RULES.md §1) — pure."""
from __future__ import annotations

from datetime import date

import pytest

from apps.trading.services.cooldown import (
    COOLDOWN, NORMAL, REDUCED, DayPnl, estimate_costs, evaluate_cooldown,
)

# Calendar: weekends + a fake holiday on Wed 2026-10-28.
HOLIDAY = date(2026, 10, 28)


def tday(d: date) -> bool:
    return d.weekday() < 5 and d != HOLIDAY


def ev(days, as_of):
    return evaluate_cooldown([DayPnl(date.fromisoformat(d), p) for d, p in days],
                             date.fromisoformat(as_of), is_trading_day=tday,
                             loss_days=3, cooldown_days=2, reduced_multiplier=0.5)


def test_no_history_is_normal():
    st = ev([], "2026-10-21")
    assert st.status == NORMAL and st.risk_multiplier == 1.0 and not st.blocks_entries


def test_two_losses_is_not_enough():
    st = ev([("2026-10-19", -100), ("2026-10-20", -50)], "2026-10-21")
    assert st.status == NORMAL and st.loss_streak == 2


def test_three_losses_trigger_two_trading_day_cooldown_across_weekend():
    # Wed, Thu, Fri losses → cooldown Mon 26 + Tue 27.
    days = [("2026-10-21", -1), ("2026-10-22", -1), ("2026-10-23", -1)]
    for as_of in ("2026-10-26", "2026-10-27"):
        st = ev(days, as_of)
        assert st.status == COOLDOWN and st.blocks_entries
        assert st.cooldown_until == date(2026, 10, 27)
        assert st.risk_multiplier == 0.0
    # Wed 28 is a holiday; next trading day Thu 29 → reduced risk, entries allowed.
    st = ev(days, "2026-10-29")
    assert st.status == REDUCED and st.risk_multiplier == 0.5 and not st.blocks_entries


def test_holiday_inside_cooldown_is_skipped():
    # Mon, Tue, Wed? Wed 28 is a holiday → losses Fri 23, Mon 26, Tue 27.
    days = [("2026-10-23", -5), ("2026-10-26", -5), ("2026-10-27", -5)]
    # Cooldown = next 2 trading days after Tue 27 → Thu 29, Fri 30 (Wed holiday).
    assert ev(days, "2026-10-29").status == COOLDOWN
    st = ev(days, "2026-10-30")
    assert st.status == COOLDOWN and st.cooldown_until == date(2026, 10, 30)
    assert ev(days, "2026-11-02").status == REDUCED


def test_no_trade_days_do_not_break_the_streak():
    # Losses on Mon 19, Wed 21 (no trades Tue), Fri 23 → still 3 consecutive loss days.
    days = [("2026-10-19", -1), ("2026-10-21", -1), ("2026-10-23", -1)]
    assert ev(days, "2026-10-26").status == COOLDOWN


def test_profit_day_breaks_streak():
    days = [("2026-10-19", -1), ("2026-10-20", -1), ("2026-10-21", 5), ("2026-10-22", -1)]
    st = ev(days, "2026-10-23")
    assert st.status == NORMAL and st.loss_streak == 1


def test_flat_day_breaks_streak_but_does_not_end_reduced_mode():
    days = [("2026-10-14", -1), ("2026-10-15", -1), ("2026-10-16", -1),   # trigger Fri 16
            ("2026-10-21", 0.0)]                                          # after cooldown (19, 20)
    st = ev(days, "2026-10-22")
    assert st.status == REDUCED


def test_reduced_mode_resets_on_first_profitable_day():
    days = [("2026-10-14", -1), ("2026-10-15", -1), ("2026-10-16", -1),
            ("2026-10-21", -3), ("2026-10-22", 10)]
    assert ev(days[:4], "2026-10-22").status == REDUCED
    st = ev(days, "2026-10-23")
    assert st.status == NORMAL and st.risk_multiplier == 1.0


def test_manual_trade_days_inside_cooldown_are_ignored():
    days = [("2026-10-14", -1), ("2026-10-15", -1), ("2026-10-16", -1),
            ("2026-10-19", 50),     # profitable manual day INSIDE cooldown
            ("2026-10-20", -1)]     # loss inside cooldown — not a streak day either
    st = ev(days, "2026-10-21")
    assert st.status == REDUCED and st.loss_streak == 0
    assert [h["in_cooldown"] for h in st.history][-2:] == [True, True]


def test_today_in_progress_is_not_counted():
    days = [("2026-10-19", -1), ("2026-10-20", -1), ("2026-10-21", -1)]
    assert ev(days, "2026-10-21").status == NORMAL    # as_of day itself excluded
    assert ev(days, "2026-10-22").status == COOLDOWN


def test_second_streak_after_reduced_triggers_new_cooldown():
    days = [("2026-10-12", -1), ("2026-10-13", -1), ("2026-10-14", -1),   # cd 15,16
            ("2026-10-19", -1), ("2026-10-20", -1), ("2026-10-21", -1)]   # cd 22,23
    st = ev(days, "2026-10-22")
    assert st.status == COOLDOWN and st.trigger_date == date(2026, 10, 21)


@pytest.mark.parametrize("product,exchange", [("INTRADAY", "NSE"), ("DELIVERY", "NSE"),
                                               ("INTRADAY", "NFO")])
def test_cost_estimate_is_positive_and_small(product, exchange):
    c = estimate_costs(side="BUY", entry=1000, exit=1010, units=100,
                       exchange=exchange, product=product)
    assert 0 < c < 0.01 * 1000 * 100 * 2


def test_cost_estimate_turns_scratch_into_loss():
    from types import SimpleNamespace

    from apps.trading.services.cooldown import trade_net_pnl

    t = SimpleNamespace(realized_pnl=5, exit_price=1000.05, quantity=100, exit_quantity=100,
                        lot_size=1, side="BUY", fill_price=1000, entry_price=1000,
                        exchange="NSE", product="INTRADAY")
    assert trade_net_pnl(t) < 0
