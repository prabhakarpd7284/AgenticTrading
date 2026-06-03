"""Unit tests for lifecycle rules — profit-take, hard-stop, re-entry gates."""
from __future__ import annotations

import pytest

from plugins.strategy_vertical_spread.lifecycle import (
    LifecycleAction, MIN_CREDIT_WIDTH_RATIO, MIN_DTE_FOR_REENTER,
    PROFIT_TAKE_PCT,
    can_reenter, evaluate_open_spread,
)


def test_profit_take_fires_at_70pct():
    """Entry credit 100, spread now worth 30 → 70% captured → CLOSE."""
    d = evaluate_open_spread(entry_credit=100.0, current_spread_value=30.0,
                              max_profit=100 * 75, dte=3)
    assert d.action == LifecycleAction.CLOSE
    assert "profit-take" in d.reason
    assert d.pnl_pct_of_max >= PROFIT_TAKE_PCT


def test_hold_below_profit_take():
    """20% of max — HOLD."""
    d = evaluate_open_spread(entry_credit=100.0, current_spread_value=80.0,
                              max_profit=100 * 75, dte=4)
    assert d.action == LifecycleAction.HOLD


def test_hard_stop_fires_above_1_5x_credit():
    d = evaluate_open_spread(entry_credit=100.0, current_spread_value=160.0,
                              max_profit=100 * 75, dte=3)
    assert d.action == LifecycleAction.CLOSE
    assert "hard-stop" in d.reason


def test_time_stop_at_dte_zero():
    d = evaluate_open_spread(entry_credit=100.0, current_spread_value=50.0,
                              max_profit=100 * 75, dte=0)
    assert d.action == LifecycleAction.CLOSE


def test_zero_credit_does_not_crash():
    d = evaluate_open_spread(entry_credit=0.0, current_spread_value=0.0,
                              max_profit=0.0, dte=3)
    assert d.action == LifecycleAction.HOLD


def test_reenter_gate_blocks_retracement():
    """Apr 22 scenario — spot fell 200 pts from last close. Block."""
    ok, reason = can_reenter(spot_now=24378, spot_at_last_close=24577,
                              dte=5, new_credit=35.20, new_width=100)
    assert ok is False
    assert "retraced" in reason


def test_reenter_gate_allows_momentum_intact():
    ok, reason = can_reenter(spot_now=24600, spot_at_last_close=24577,
                              dte=5, new_credit=35.20, new_width=100)
    assert ok is True


def test_reenter_gate_blocks_low_dte():
    ok, reason = can_reenter(spot_now=24600, spot_at_last_close=24577,
                              dte=2, new_credit=35.20, new_width=100)
    assert ok is False
    assert "DTE" in reason


def test_reenter_gate_blocks_low_credit_quality():
    """Width 100, credit 10 → 10% credit/width — below 25% threshold."""
    ok, reason = can_reenter(spot_now=24600, spot_at_last_close=24577,
                              dte=5, new_credit=10.0, new_width=100)
    assert ok is False
    assert "credit/width" in reason


def test_reenter_first_entry_no_prior_close():
    """First trade of the day has no prior close — momentum gate skipped."""
    ok, _ = can_reenter(spot_now=24050, spot_at_last_close=None,
                         dte=5, new_credit=30, new_width=100)
    assert ok is True
