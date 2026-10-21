"""Breakeven-then-1.5×ATR trailing stop (TRADING_RULES.md §2) — pure."""
from __future__ import annotations

from apps.trading.services.trailing_stop import (
    BREAKEVEN, INITIAL, TRAILING, next_stop, normalize_symbol, wilder_atr,
)


def long(best, atr=2.0, cur=None):
    return next_stop(side="BUY", entry=100, initial_stop=95, best_price=best, atr=atr,
                     current_stop=cur)


def short(best, atr=2.0, cur=None):
    return next_stop(side="SELL", entry=100, initial_stop=105, best_price=best, atr=atr,
                     current_stop=cur)


def test_long_holds_initial_stop_below_1r():
    d = long(104.9)
    assert d.stop == 95 and d.phase == INITIAL and not d.moved


def test_long_moves_to_breakeven_at_exactly_1r():
    d = long(105, atr=5.0)          # 105 - 7.5 = 97.5 < entry → breakeven
    assert d.stop == 100 and d.phase == BREAKEVEN and d.moved


def test_long_trails_1_5_atr_behind_best():
    d = long(110, atr=2.0)          # 110 - 3 = 107
    assert d.stop == 107 and d.phase == TRAILING


def test_long_never_loosens():
    d = long(106, atr=2.0, cur=107)  # candidate 103 < current 107
    assert d.stop == 107 and not d.moved


def test_long_without_atr_only_breakeven():
    d = long(120, atr=None)
    assert d.stop == 100 and d.phase == BREAKEVEN


def test_short_breakeven_and_trail():
    assert short(95.1).stop == 105                     # < 1R
    be = short(95, atr=5.0)                            # 95 + 7.5 = 102.5 > entry → BE
    assert be.stop == 100 and be.phase == BREAKEVEN
    tr = short(90, atr=2.0)                            # 90 + 3 = 93
    assert tr.stop == 93 and tr.phase == TRAILING


def test_short_never_loosens():
    d = short(94, atr=2.0, cur=93)                     # candidate 97 > 93 → keep 93
    assert d.stop == 93 and not d.moved


def test_ratchet_sequence_monotonic():
    stop = None
    seen = []
    for best, atr in [(101, 2), (105, 2), (108, 2), (107, 4), (112, 1), (111, 9)]:
        d = long(best, atr=atr, cur=stop)
        stop = d.stop
        seen.append(stop)
    assert seen == sorted(seen)


def test_zero_risk_trade_untouched():
    d = next_stop(side="BUY", entry=100, initial_stop=100, best_price=150, atr=1)
    assert d.stop == 100 and not d.moved


def test_wilder_atr():
    bars = [(10, 8, 9)] * 20
    assert wilder_atr(bars, 14) == 2.0
    assert wilder_atr(bars[:10], 14) is None


def test_normalize_symbol():
    assert normalize_symbol("NSE:RELIANCE-EQ") == "RELIANCE"
    assert normalize_symbol("m&m") == "M&M"
    assert normalize_symbol("BAJAJ-AUTO") == "BAJAJ-AUTO"
