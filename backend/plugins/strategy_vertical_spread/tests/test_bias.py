"""Unit tests for the 5-rule bias classifier.

Includes regression tests replaying the April 2026 backtest days that
matter — the gap-override on Apr 1 and the RANGE on Apr 22 (where the
old EMA20 classifier wrongly fired an entry that lost ₹3,131).
"""
from __future__ import annotations

import pytest

from plugins.strategy_vertical_spread.bias import (
    Bias, BIAS_TO_MODE, classify,
)


def test_gap_override_up():
    """+2.54% gap (Apr 1 2026) — must classify UP regardless of momentum."""
    r = classify(today_open=23250, prev_close=22680,
                  close_3d_ago=22500, spot_now=22680,
                  vix_today=25.0, vix_yesterday=25.0)
    assert r.bias == Bias.UP
    assert "gap" in r.reason.lower()


def test_gap_override_down():
    """−1.31% gap (Apr 2 2026) — must classify DOWN."""
    r = classify(today_open=22600, prev_close=22900,
                  close_3d_ago=22850, spot_now=22713,
                  vix_today=25.5, vix_yesterday=25.0)
    assert r.bias == Bias.DOWN


def test_three_day_momentum_up_with_vix_fall():
    """Apr 10 setup — gap small, 3d mom +927, VIX falling. Must be UP."""
    r = classify(today_open=24000, prev_close=23900,
                  close_3d_ago=23123, spot_now=24050,
                  vix_today=18.85, vix_yesterday=20.43)
    assert r.bias == Bias.UP
    assert "3d mom" in r.reason


def test_three_day_momentum_down_with_vix_rise():
    """Symmetric DOWN — mom -800, VIX rising."""
    r = classify(today_open=23000, prev_close=23000,
                  close_3d_ago=23800, spot_now=23000,
                  vix_today=22.0, vix_yesterday=19.0)
    assert r.bias == Bias.DOWN


def test_range_when_mom_stalls():
    """Apr 22 2026 — gap -0.43%, mom +25 (below 50pt threshold), VIX +0.77.
    All four rules miss → RANGE. (This is the trade v2 wrongly took
    and v3 correctly skipped.)"""
    r = classify(today_open=24500, prev_close=24606,
                  close_3d_ago=24353, spot_now=24378,
                  vix_today=18.30, vix_yesterday=17.53)
    assert r.bias == Bias.RANGE
    assert "no edge" in r.reason.lower()


def test_range_when_mixed_signals():
    """Momentum up but VIX up too (mixed) — RANGE."""
    r = classify(today_open=24000, prev_close=24000,
                  close_3d_ago=23500, spot_now=24050,
                  vix_today=22.0, vix_yesterday=20.0)
    # mom +550 with VIX rising → no confluence → RANGE
    assert r.bias == Bias.RANGE


def test_zero_vix_yesterday_does_not_crash():
    """Missing yesterday's VIX should degrade to RANGE, not crash."""
    r = classify(today_open=24000, prev_close=24000,
                  close_3d_ago=24000, spot_now=24000,
                  vix_today=18.0, vix_yesterday=0.0)
    assert r.bias in (Bias.UP, Bias.DOWN, Bias.RANGE)


def test_bias_to_mode_mapping():
    """Every Bias enum value maps to a structure."""
    assert BIAS_TO_MODE[Bias.UP] == "BULL_PUT"
    assert BIAS_TO_MODE[Bias.DOWN] == "BEAR_CALL"
    assert BIAS_TO_MODE[Bias.RANGE] == "IRON_CONDOR"
