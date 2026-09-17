"""Unit tests for Black-Scholes pricing primitives.

Validates against put-call parity, IV round-trip, and known-good values
from the PoC's BSM calculations.
"""
from __future__ import annotations

import math

import pytest

from plugins.strategy_vertical_spread.pricing import (
    bsm_call, bsm_put, delta_call, delta_put, iv_from_price,
)


def test_put_call_parity():
    """C − P = S − K·e^(−rT)  (forward arbitrage relation, for European options)."""
    S, K, T, r, sigma = 24000, 24000, 30 / 365, 0.07, 0.18
    C = bsm_call(S, K, T, r, sigma)
    P = bsm_put(S, K, T, r, sigma)
    expected = S - K * math.exp(-r * T)
    assert abs((C - P) - expected) < 1e-3


def test_iv_round_trip():
    """Given a BSM price, iv_from_price should recover the input sigma."""
    S, K, T, r, sigma = 24000, 23800, 7 / 365, 0.07, 0.20
    target = bsm_put(S, K, T, r, sigma)
    recovered = iv_from_price(target, S, K, T, r, is_call=False)
    assert abs(recovered - sigma) < 0.001, f"recovered={recovered} expected={sigma}"


def test_atm_pe_call_match_for_zero_rate():
    """At zero risk-free rate, ATM call and put are identical."""
    p = bsm_put(24000, 24000, 7 / 365, 0.0, 0.18)
    c = bsm_call(24000, 24000, 7 / 365, 0.0, 0.18)
    assert abs(p - c) < 1e-6


def test_otm_put_below_intrinsic_zero():
    """A 1000-pt OTM put with low IV has near-zero value."""
    p = bsm_put(24000, 23000, 1 / 365, 0.07, 0.15)
    assert 0 <= p < 0.5


def test_deep_itm_put_approaches_intrinsic():
    """A deep-ITM put (with little time value) prices near intrinsic K − S."""
    p = bsm_put(20000, 24000, 1 / 365, 0.07, 0.10)
    intrinsic = 24000 - 20000
    assert abs(p - intrinsic) < 100   # very little time premium left


def test_delta_signs():
    """Call delta in [0,1], put delta in [-1,0]."""
    S, K, T, r, sigma = 24000, 24000, 7 / 365, 0.07, 0.18
    assert 0 < delta_call(S, K, T, r, sigma) < 1
    assert -1 < delta_put(S, K, T, r, sigma) < 0


def test_delta_relationship():
    """Δ_call − Δ_put = 1 (for European options, no dividend)."""
    S, K, T, r, sigma = 24000, 24100, 14 / 365, 0.07, 0.20
    diff = delta_call(S, K, T, r, sigma) - delta_put(S, K, T, r, sigma)
    assert abs(diff - 1.0) < 1e-6


def test_zero_time_to_expiry():
    """At T=0 options price equals intrinsic value."""
    assert bsm_put(24000, 24100, 0, 0.07, 0.18) == 100
    assert bsm_call(24000, 24100, 0, 0.07, 0.18) == 0
    assert bsm_put(24200, 24100, 0, 0.07, 0.18) == 0
    assert bsm_call(24200, 24100, 0, 0.07, 0.18) == 100
