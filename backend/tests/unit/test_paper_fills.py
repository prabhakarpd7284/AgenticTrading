"""Paper fill simulation — the decision layer.

`PaperBrokerAdapter.place()` returns an order id and nothing else ever moves
the order on, so paper orders sat at `sent` forever: no Trade rows, no P&L, no
data to learn from. This is the missing half.

Deliberately pessimistic. A paper book that flatters entries teaches the wrong
lesson, so where reality is ambiguous we take the trader's worse side:

  * a LIMIT order fills at its limit, never at a better touch price
  * an unknown price (0.0) never fills anything
  * if stop and target are both touched in one tick, the stop wins
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from apps.trading.services.paper_fills import exit_for, fill_price_for


class _Order:
    def __init__(self, side, order_type="MARKET", price=None):
        self.side = side
        self.order_type = order_type
        self.price = Decimal(str(price)) if price is not None else None


class _Trade:
    def __init__(self, side, stop_loss, target):
        self.side = side
        self.stop_loss = Decimal(str(stop_loss))
        self.target = Decimal(str(target))


# ── Fills ──────────────────────────────────────────────────────────────

def test_market_order_fills_at_the_tick():
    assert fill_price_for(_Order("BUY"), 101.5) == 101.5
    assert fill_price_for(_Order("SELL"), 101.5) == 101.5


def test_limit_buy_fills_only_at_or_below_its_price():
    o = _Order("BUY", "LIMIT", 100.0)
    assert fill_price_for(o, 101.0) is None      # market above limit
    assert fill_price_for(o, 100.0) == 100.0     # touched
    assert fill_price_for(o, 98.0) == 100.0      # gapped through — no free money


def test_limit_sell_fills_only_at_or_above_its_price():
    o = _Order("SELL", "LIMIT", 100.0)
    assert fill_price_for(o, 99.0) is None
    assert fill_price_for(o, 100.0) == 100.0
    assert fill_price_for(o, 102.0) == 100.0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_unknown_price_never_fills(bad):
    """reference_price() returns 0.0 for 'unknown' — that must not become a
    fill at zero, which would book a fictional 100% gain."""
    assert fill_price_for(_Order("BUY"), bad) is None
    assert fill_price_for(_Order("BUY", "LIMIT", 100.0), bad) is None


def test_limit_order_without_a_price_does_not_fill():
    assert fill_price_for(_Order("BUY", "LIMIT", None), 100.0) is None


# ── Exits ──────────────────────────────────────────────────────────────

def test_long_exits_on_stop_and_target():
    t = _Trade("BUY", stop_loss=95.0, target=110.0)
    assert exit_for(t, 100.0) is None
    assert exit_for(t, 95.0) == (95.0, "SL_HIT")
    assert exit_for(t, 94.0) == (95.0, "SL_HIT")     # gap — fills at the stop
    assert exit_for(t, 110.0) == (110.0, "TARGET_HIT")


def test_short_exits_are_inverted():
    t = _Trade("SELL", stop_loss=110.0, target=95.0)
    assert exit_for(t, 100.0) is None
    assert exit_for(t, 110.0) == (110.0, "SL_HIT")
    assert exit_for(t, 95.0) == (95.0, "TARGET_HIT")


def test_stop_wins_when_both_are_touched():
    """One tick cannot tell us which came first inside the bar. Assuming the
    target would flatter every ambiguous trade."""
    t = _Trade("BUY", stop_loss=95.0, target=110.0)
    # A tick that is somehow beyond both (wide gap) must resolve to the stop.
    assert exit_for(t, 94.0)[1] == "SL_HIT"


def test_unknown_price_never_exits():
    t = _Trade("BUY", stop_loss=95.0, target=110.0)
    assert exit_for(t, 0.0) is None


def test_missing_levels_do_not_exit():
    class _Bare:
        side = "BUY"
        stop_loss = None
        target = None

    assert exit_for(_Bare(), 100.0) is None
